"""
Memory Agent
============
Handles all memory operations for JARVIS:
- Store memories (short-term or long-term based on importance)
- Recall/recent memories
- Semantic search across memories
- Memory consolidation (LTM promote)
- Memory statistics

Uses DeepSeek for structured memory extraction and routing.
Coordinates STM, LTM, vector memory, and the high-level memory service.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.agents_v2.state import AgentState, ExecutionStatus
from backend.agents_v2.base import BaseAgent
from backend.agents_v2.registry import get_agent_registry
from backend.agents_v2.tools import AgentTools
from backend.llm.deepseek import deepseek
from backend.database.mongodb import mongodb
from backend.memory.short_term import stm
from backend.memory.long_term import ltm
from backend.memory.vector_memory import vector_memory
from backend.services.memory_service import memory_service
from backend.services.task_service import task_service
from backend.config.settings import settings


class MemoryAgent(BaseAgent):
    """
    Memory Agent — manages JARVIS's memory system.

    Detects the user's intent from natural language, then routes to
    the appropriate memory subsystem (STM, LTM, vector search) and
    returns a natural language response summarising the result.

    Capabilities:
    - ``store``  — extract structured memory via DeepSeek, route to STM/LTM
    - ``recall`` — semantic search on LTM, text search on STM
    - ``recent`` — sliding-window of short-term memories
    - ``search`` — full semantic + text search across all stores
    - ``consolidate`` — promote qualifying STM items into LTM
    - ``stats`` — aggregate memory counts
    - ``general`` — answer questions using memory context
    """

    # ── Intent-compiled patterns (fast-path, no LLM call) ─────

    _STORE = re.compile(
        r"(?:remember|store|save|note\s*(?:down|that)?|"
        r"don't\s*forget|keep\s*in\s*mind|take\s*note)",
        re.IGNORECASE,
    )
    _RECALL = re.compile(
        r"(?:what\s*(?:do\s*you\s*)?know\s*about|"
        r"recall|tell\s*me\s*about|"
        r"do\s*you\s*(?:remember|know)|"
        r"what\s+(?:is|was|are|were)\s+.*)",
        re.IGNORECASE,
    )
    _RECENT = re.compile(
        r"(?:what\s+happened\s+recently|recent\s+memor(?:y|ies)|"
        r"what\s+(?:did|have)\s+(?:i|we)\s+(?:do|talk|discuss).*"
        r"(?:recently|earlier|before|last)|"
        r"what\s+was\s+(?:the\s+)?last)",
        re.IGNORECASE,
    )
    _SEARCH = re.compile(
        r"(?:search\s*(?:for|my)?|find\s+(?:memor(?:y|ies)\s+)?(?:about|for|related\s+to)?|"
        r"look\s+up|semantic\s+search)",
        re.IGNORECASE,
    )
    _CONSOLIDATE = re.compile(
        r"(?:consolidate|promote\s*(?:to\s*)?ltm|"
        r"move\s*(?:to\s*)?long.?term|"
        r"optimize\s*memor)",
        re.IGNORECASE,
    )
    _STATS = re.compile(
        r"(?:memory\s*(?:stats?|statistics|summary|usage|report)|"
        r"how\s+many\s+memor(?:y|ies)|"
        r"show\s*(?:my\s*)?memor)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__(
            name="memory",
            model_name="deepseek",
            system_prompt="""You are JARVIS's Memory Agent. Your job is to manage JARVIS's memory system.

Commands you understand:
- "remember X" / "store X" → Store information in memory
- "what do you know about X" / "recall X" → Retrieve memories about X
- "what happened recently" / "recent memories" → Get recent memories
- "search for X" → Semantic search across memories
- "consolidate" → Consolidate short-term to long-term memory
- "memory stats" → Get memory statistics
- "forget X" / "delete X" → Remove memories (future feature)

Respond naturally based on the memory operation result.""",
            description="Stores, retrieves, and manages memories",
        )

    # ── Main entry point ──────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Detect memory operation, execute it, and populate state."""
        message = state["message"]
        user_id = state["user_id"]
        operation, query = self._detect_operation(message)

        handler_map: Dict[str, Any] = {
            "store": self._handle_store,
            "recall": self._handle_recall,
            "recent": self._handle_recent,
            "search": self._handle_search,
            "consolidate": self._handle_consolidate,
            "stats": self._handle_stats,
        }

        handler = handler_map.get(operation, self._handle_general)
        result = await handler(user_id, query if operation != "general" else message, state)

        # Populate shared context so downstream agents can see memories
        state.setdefault("shared_context", {})
        state["shared_context"]["memories"] = result.get("memories", [])
        state["final_response"] = result.get("response", "")

        # Log the action for observability
        await self._store_agent_log(
            state=state,
            action=f"memory_{operation}",
            input_summary=message[:200],
            output_summary=result.get("response", "")[:200],
        )

        return state

    # ── Operation detection ───────────────────────────────────

    def _detect_operation(self, message: str) -> Tuple[str, str]:
        """
        Classify the user's memory intent using regex patterns.

        Returns
            (operation_key, cleaned_query)
        """
        msg = message.strip()

        for pattern, op in [
            (self._STORE, "store"),
            (self._RECALL, "recall"),
            (self._RECENT, "recent"),
            (self._SEARCH, "search"),
            (self._CONSOLIDATE, "consolidate"),
            (self._STATS, "stats"),
        ]:
            match = pattern.search(msg)
            if match:
                # Extract everything after the matched keyword
                query = msg[match.end() :].strip().lstrip(":,;. ")
                return op, query

        return "general", msg

    # ── Store ─────────────────────────────────────────────────

    async def _handle_store(
        self, user_id: str, query: str, state: AgentState
    ) -> Dict[str, Any]:
        """Extract structured memory via DeepSeek and persist it."""
        if not query:
            return {"response": "What would you like me to remember?", "memories": []}

        try:
            analysis = await deepseek.extract_json(
                system_prompt="""You are a memory extraction system. Analyse the user's message and extract a memory to persist.

Rules:
- Extract the *key information* that is worth remembering.
- Assign an *importance score* between 0.0 and 1.0:
  0.9+ = critical facts, personal info, preferences
  0.7–0.8 = useful context, project details
  0.5–0.6 = casual observations
  < 0.5 = transient, not worth keeping
- Suggest relevant *tags* (2–4).
- Provide a one-sentence *summary*.

Respond with valid JSON **only**:
{
  "content": "...",
  "importance_score": 0.0-1.0,
  "tags": ["..."],
  "summary": "..."
}""",
                user_message=query,
            )
        except Exception:
            # Fallback: treat the raw message as content with medium importance
            analysis = {
                "content": query,
                "importance_score": 0.5,
                "tags": ["user_input"],
                "summary": query[:100],
            }

        content = str(analysis.get("content", query))
        importance = float(analysis.get("importance_score", 0.5))
        tags = list(analysis.get("tags", [])) if isinstance(analysis.get("tags"), list) else []
        summary = str(analysis.get("summary", ""))[:200]

        # Route to LTM if above threshold, else STM
        used_ltm = False
        if importance >= settings.MEMORY_LTM_IMPORTANCE_THRESHOLD:
            try:
                embedding = await vector_memory.embed(content)
                await ltm.store(
                    user_id=user_id,
                    content=content,
                    embedding=embedding,
                    tags=tags,
                    importance_score=importance,
                    summary=summary,
                    source="user_message",
                )
                used_ltm = True
            except ValueError:
                # Below threshold after all — fall through to STM
                await stm.store(
                    user_id=user_id,
                    content=content,
                    tags=tags,
                    importance_score=importance,
                    metadata={"summary": summary, "source": "user_message"},
                )
        else:
            await stm.store(
                user_id=user_id,
                content=content,
                tags=tags,
                importance_score=importance,
                metadata={"summary": summary, "source": "user_message"},
            )

        memory_type = "long-term" if used_ltm else "short-term"
        return {
            "response": (
                f"✅ I've stored that in **{memory_type} memory**."
                + (f" ({summary})" if summary else "")
            ),
            "memories": [
                {
                    "content": content,
                    "type": memory_type,
                    "importance": importance,
                    "tags": tags,
                }
            ],
        }

    # ── Recall ────────────────────────────────────────────────

    async def _handle_recall(
        self, user_id: str, query: str, state: AgentState
    ) -> Dict[str, Any]:
        """Retrieve memories semantically matching the query."""
        if not query:
            return {"response": "What would you like me to recall?", "memories": []}

        all_memories: List[Dict[str, Any]] = []

        # LTM semantic search
        try:
            embedding = await vector_memory.embed(query)
            ltm_results = await ltm.semantic_search(user_id, embedding, limit=5)
            all_memories.extend(ltm_results)
        except Exception:
            pass

        # STM text search (supplement)
        try:
            stm_results = await stm.search(user_id, query, limit=5)
            # Deduplicate by content
            existing = {m.get("content", "") for m in all_memories}
            for mem in stm_results:
                if mem.get("content", "") not in existing:
                    all_memories.append(mem)
        except Exception:
            pass

        if not all_memories:
            return {
                "response": f"I don't have any memories about *{query}* yet.",
                "memories": [],
            }

        # Build a clean summary
        lines: List[str] = [f"I found {len(all_memories)} memory/ies about *{query}*:\n"]
        for mem in all_memories[:5]:
            content = mem.get("content", "")[:200]
            imp = mem.get("importance_score", 0.0)
            mtype = mem.get("memory_type", "short_term")
            lines.append(f"- [{mtype}] (importance {imp:.1f}) {content}")

        return {"response": "\n".join(lines), "memories": all_memories}

    # ── Recent ────────────────────────────────────────────────

    async def _handle_recent(
        self, user_id: str, _query: str, state: AgentState
    ) -> Dict[str, Any]:
        """Fetch the most recent STM entries."""
        recent = await stm.get_context_window(user_id, max_items=15)

        if not recent:
            return {"response": "No recent memories found.", "memories": []}

        lines: List[str] = [f"Here are your {len(recent)} most recent memories:\n"]
        for mem in recent:
            content = mem.get("content", "")[:150]
            imp = mem.get("importance_score", 0.0)
            lines.append(f"- {content} (importance {imp:.1f})")

        return {"response": "\n".join(lines), "memories": recent}

    # ── Search ────────────────────────────────────────────────

    async def _handle_search(
        self, user_id: str, query: str, state: AgentState
    ) -> Dict[str, Any]:
        """Full semantic + text search across all memory stores."""
        if not query:
            return {"response": "What would you like me to search for?", "memories": []}

        results = await memory_service.search(user_id, query, limit=10)

        if not results:
            return {
                "response": f"No results found for *{query}*.",
                "memories": [],
            }

        lines: List[str] = [f"Search results for *{query}* ({len(results)}):\n"]
        for mem in results[:10]:
            content = mem.get("content", "")[:150]
            mtype = mem.get("memory_type", "unknown")
            imp = mem.get("importance_score", 0.0)
            lines.append(f"- [{mtype}] {content} (score {imp:.1f})")

        return {"response": "\n".join(lines), "memories": results}

    # ── Consolidate ───────────────────────────────────────────

    async def _handle_consolidate(
        self, user_id: str, _query: str, state: AgentState
    ) -> Dict[str, Any]:
        """Promote qualifying short-term memories into LTM."""
        count = await memory_service.consolidate(user_id)

        if count > 0:
            return {
                "response": f"✅ Consolidated {count} short-term memories into long-term memory.",
                "memories": [],
            }
        return {
            "response": "No memories needed consolidation at this time.",
            "memories": [],
        }

    # ── Stats ─────────────────────────────────────────────────

    async def _handle_stats(
        self, user_id: str, _query: str, state: AgentState
    ) -> Dict[str, Any]:
        """Aggregate memory statistics."""
        stats = await memory_service.get_stats(user_id)

        response = (
            "📊 **Memory Statistics**\n\n"
            f"• Short-term memories:  **{stats.get('short_term', 0):,}**\n"
            f"• Long-term memories:   **{stats.get('long_term', 0):,}**\n"
            f"• High-importance (>0.8): **{stats.get('high_importance', 0):,}**"
        )
        return {"response": response, "memories": []}

    # ── General / fallback ────────────────────────────────────

    async def _handle_general(
        self, user_id: str, message: str, state: AgentState
    ) -> Dict[str, Any]:
        """
        Answer a question by injecting memory context into an LLM call.

        This is the default handler when no explicit memory operation
        keyword is detected.
        """
        # Load recent STM context
        await self._load_memory_context(state)

        # Search LTM for semantically relevant items
        ltm_memories: List[Dict[str, Any]] = []
        try:
            embedding = await vector_memory.embed(message)
            ltm_memories = await ltm.semantic_search(user_id, embedding, limit=5)
        except Exception:
            pass

        # Build a compact context string
        context_parts: List[str] = []
        if ltm_memories:
            context_parts.append("## Relevant Long-Term Memories")
            for m in ltm_memories:
                context_parts.append(f"- {m.get('content', '')[:200]}")
        if state.get("memory_context"):
            context_parts.append("## Recent Interactions")
            for m in state["memory_context"][:5]:
                context_parts.append(f"- {m.get('content', '')[:150]}")

        context_str = "\n".join(context_parts) if context_parts else None
        messages = self._build_system_messages(message, context=context_str)

        response_text, _tokens = await self._call_llm(messages, temperature=0.5)

        state["total_tokens"] += _tokens

        return {"response": response_text, "memories": ltm_memories}
