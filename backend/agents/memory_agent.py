"""
Memory Agent — manages all memory operations including storage, retrieval,
consolidation, and context building for the JARVIS system.
"""

from __future__ import annotations

import time
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc, serialize_doc
from backend.memory.long_term import ltm
from backend.memory.short_term import stm
from backend.memory.vector_memory import vector_memory
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryAgent:
    """Specialized agent for memory operations.

    Handles:
    - Storing new memories (STM → LTM pipeline)
    - Semantic memory retrieval
    - Context building for conversations
    - Memory consolidation
    - Forgetting and cleanup
    """

    def __init__(self) -> None:
        self.name = "memory_agent"

    async def process(
        self,
        user_id: str,
        message: str,
        context: list[dict[str, str]] | None = None,
        attachments: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a memory-related request."""
        start = time.monotonic()
        session_id = session_id or f"memory_{int(start)}"

        task_type = self._detect_task_type(message)
        logger.info(
            "Memory agent processing",
            extra={"task_type": task_type, "session_id": session_id, "user_id": user_id},
        )

        try:
            if task_type == "store":
                result = await self._store_memory(user_id, message)
            elif task_type == "recall":
                result = await self._recall(user_id, message)
            elif task_type == "search":
                result = await self._search(user_id, message)
            elif task_type == "stats":
                result = await self._stats(user_id)
            elif task_type == "consolidate":
                result = await self._consolidate(user_id)
            else:
                result = await self._general_memory_response(user_id, message)

            elapsed = (time.monotonic() - start) * 1000
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary=result.get("content", "")[:200],
                    duration_ms=elapsed,
                    status="success",
                )
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Memory agent failed", extra={"error": str(exc), "session_id": session_id})
            return {"content": f"Memory operation failed: {str(exc)}", "agent": self.name, "error": str(exc)}

    async def _store_memory(self, user_id: str, message: str) -> dict[str, Any]:
        """Store information in memory."""
        # Extract what to remember
        content = message
        for prefix in ["remember ", "remember that ", "save this: ", "store: ", "don't forget "]:
            if message.lower().startswith(prefix):
                content = message[len(prefix):]
                break

        # Get embedding for LTM
        try:
            embedding = await vector_memory.embed(content)
        except Exception:
            embedding = None

        # Store in LTM
        try:
            doc = await ltm.store(
                user_id=user_id,
                content=content,
                embedding=embedding,
                tags=["user_saved"],
                importance_score=0.7,
                source="user_explicit",
            )
            return {
                "content": f"I've stored that in my long-term memory. I'll remember: \"{content[:100]}{'...' if len(content) > 100 else ''}\"",
                "agent": self.name,
                "metadata": {"memory_id": doc.get("id"), "memory_type": "long_term"},
            }
        except ValueError:
            # Below threshold, store in STM instead
            doc = await stm.store(
                user_id=user_id,
                content=content,
                tags=["user_saved"],
                importance_score=0.5,
            )
            return {
                "content": f"I've noted that in my short-term memory: \"{content[:100]}{'...' if len(content) > 100 else ''}\"",
                "agent": self.name,
                "metadata": {"memory_id": doc.get("id"), "memory_type": "short_term"},
            }

    async def _recall(self, user_id: str, message: str) -> dict[str, Any]:
        """Recall information from memory."""
        # Extract what to recall
        query = message
        for prefix in ["do you remember ", "recall ", "what did i say about ", "what do you know about "]:
            if message.lower().startswith(prefix):
                query = message[len(prefix):]
                break

        # Generate query embedding
        try:
            query_embedding = await vector_memory.embed(query)
        except Exception:
            query_embedding = None

        # Search LTM
        memories: list[dict[str, Any]] = []
        if query_embedding:
            try:
                memories = await ltm.semantic_search(
                    user_id, query_embedding, limit=5, threshold=0.4
                )
            except Exception:
                pass

        # Fall back to STM text search
        if not memories:
            try:
                memories = await stm.search(user_id, query, limit=5)
            except Exception:
                pass

        if memories:
            content = "Here's what I remember:\n\n"
            for mem in memories:
                importance = mem.get("importance_score", 0)
                stars = "⭐" * max(1, int(importance * 5))
                content += f"- {mem.get('content', '')[:300]} {stars}\n"
            return {"content": content, "agent": self.name, "metadata": {"memories_found": len(memories)}}
        else:
            return {
                "content": "I don't have any memories related to that. Would you like me to remember something specific?",
                "agent": self.name,
                "metadata": {"memories_found": 0},
            }

    async def _search(self, user_id: str, message: str) -> dict[str, Any]:
        """Semantic search across all memories."""
        query = message
        for prefix in ["search ", "find ", "look for "]:
            if message.lower().startswith(prefix):
                query = message[len(prefix):]
                break

        try:
            query_embedding = await vector_memory.embed(query)
        except Exception:
            query_embedding = None

        results = []
        if query_embedding:
            try:
                results = await ltm.semantic_search(user_id, query_embedding, limit=10)
            except Exception:
                pass

        if results:
            content = f"## Memory Search Results for: \"{query}\"\n\n"
            for i, mem in enumerate(results, 1):
                score = mem.get("score", 0)
                content += f"{i}. **{mem.get('memory_type', 'memory')}** (relevance: {score:.2f})\n"
                mtype = mem.get('memory_type', '')
                content += f"   - Score: {mem.get('importance_score', 0):.1f}/1.0\n"
                content += f"   - {mem.get('content', '')[:200]}...\n\n"
            return {"content": content, "agent": self.name, "metadata": {"results_count": len(results)}}
        else:
            return {
                "content": "No memories found matching your search.",
                "agent": self.name,
                "metadata": {"results_count": 0},
            }

    async def _stats(self, user_id: str) -> dict[str, Any]:
        """Get memory statistics."""
        stm_count = await stm.count_active(user_id)
        ltm_stats = await ltm.get_stats(user_id)
        content = f"""## Memory Statistics

- **Short-term memories:** {stm_count} active
- **Long-term memories:** {ltm_stats.get('total', 0)} total
- **High-importance memories:** {ltm_stats.get('high_importance', 0)}"""
        return {"content": content, "agent": self.name, "metadata": {"stats": {"stm": stm_count, "ltm": ltm_stats}}}

    async def _consolidate(self, user_id: str) -> dict[str, Any]:
        """Trigger STM to LTM consolidation."""
        count = await ltm.consolidate_from_stm(
            user_id, embedding_func=vector_memory.embed if hasattr(vector_memory, 'embed') else None
        )
        if count > 0:
            return {
                "content": f"Consolidated {count} short-term memories into long-term storage.",
                "agent": self.name,
                "metadata": {"consolidated_count": count},
            }
        return {
            "content": "No memories needed consolidation at this time.",
            "agent": self.name,
            "metadata": {"consolidated_count": 0},
        }

    async def _general_memory_response(self, user_id: str, message: str) -> dict[str, Any]:
        """Handle general memory-related queries."""
        # Try to recall relevant information
        try:
            embedding = await vector_memory.embed(message)
            memories = await ltm.semantic_search(user_id, embedding, limit=3)
        except Exception:
            memories = []

        if memories:
            context = "\n".join(f"- {m['content'][:200]}" for m in memories)
            return {
                "content": f"Based on what I know:\n\n{context}\n\nIs there anything specific you'd like to remember or store?",
                "agent": self.name,
                "metadata": {"memories_found": len(memories)},
            }
        return {
            "content": "I can help you remember things! Try:\n- \"Remember that my favorite color is blue\"\n- \"What do you know about Python?\"\n- \"Search memories for project details\"",
            "agent": self.name,
        }

    def _detect_task_type(self, message: str) -> str:
        """Detect the memory task type."""
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["remember ", "remember that", "store this", "save this", "don't forget"]):
            return "store"
        if any(kw in msg_lower for kw in ["do you remember", "recall", "what did i say"]):
            return "recall"
        if any(kw in msg_lower for kw in ["search ", "find ", "look for"]):
            return "search"
        if any(kw in msg_lower for kw in ["memory stats", "memory statistics", "how many memories"]):
            return "stats"
        if any(kw in msg_lower for kw in ["consolidate", "optimize memory"]):
            return "consolidate"
        return "general"


# Global singleton
memory_agent = MemoryAgent()
