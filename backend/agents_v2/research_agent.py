"""
Research Agent
==============
Handles web search, deep research, summarization, and quick-answer lookup
using the Minimax M2.1 model backed by the SearchService abstraction.

Capabilities:
  - **search**:   Web search with result synthesis via Minimax.
  - **deep**:     Multi-source deep research producing a structured report.
  - **summarize**:Condense long text into a concise summary.
  - **quick**:    Short answer lookup (chat-style) for factual questions.

Results are optionally persisted to MongoDB's ``knowledge`` collection for
future reuse across agents.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from agents_v2.state import AgentState, ExecutionStatus
from agents_v2.base import BaseAgent
from agents_v2.registry import get_agent_registry
from agents_v2.tools import AgentTools
from llm.codex import codex
from llm.minimax import minimax
from llm.mimo import mimo
from llm.router import llm_router
from services.search_service import search_service
from database.mongodb import mongodb
from database.schemas import new_knowledge_doc
from config.settings import settings


RESEARCH_KEYWORDS: Dict[str, List[str]] = {
    "search": [
        "search the", "search for", "look up", "google", "web search",
        "find information", "latest", "news about", "find out",
    ],
    "deep": [
        "deep research", "research", "investigate", "report on",
        "comprehensive", "in-depth", "thorough", "detailed analysis",
        "write a report", "study", "compare and contrast",
    ],
    "summarize": [
        "summarize", "summary", "tl;dr", "tldr", "summarise",
        "condense", "brief me", "key points", "recap",
    ],
    "quick": [
        "what is", "who is", "when did", "where is", "define",
        "explain briefly", "tell me about", "meaning of",
        "how to", "why does",
    ],
}


class ResearchAgent(BaseAgent):
    """Agent that performs web research, deep-dive reports, summarization
    and quick fact lookup via Minimax M2.1 and the backend SearchService.

    Agents register themselves with the global ``AgentRegistry`` on
    initialisation so they are discoverable by the LangGraph router.
    """

    def __init__(self) -> None:
        super().__init__(
            name="research",
            model_name="minimax",
            system_prompt=(
                "You are JARVIS's Research Agent, an expert research analyst. "
                "You search the web, gather sources, synthesise information, "
                "and produce well-structured, factual reports.\n\n"
                "Rules:\n"
                "1. Always cite sources where possible.\n"
                "2. Distinguish between established facts and speculation.\n"
                "3. When information is contradictory, note both sides.\n"
                "4. Keep responses concise unless a detailed report is requested.\n"
                "5. Use bullet points and headings for readability.\n"
                "6. Store important findings for future reference."
            ),
            description=(
                "Searches the web, gathers sources, and produces research "
                "reports using Minimax"
            ),
        )
        get_agent_registry().register(self)

    # ── Public API ──────────────────────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Execute the research agent's logic on *state*.

        Steps:
        1. Read ``message`` and ``shared_context`` from state.
        2. Load memory context.
        3. Detect research type (search / deep / summarise / quick).
        4. Execute the appropriate research pipeline.
        5. Store knowledge in MongoDB for future retrieval.
        6. Update ``final_response`` and ``shared_context``.
        7. Log execution.
        """
        message: str = state.get("message", "") or ""
        context: Dict[str, Any] = state.get("shared_context", {})
        await self._load_memory_context(state)

        research_type = self._detect_research_type(message)
        state.setdefault("shared_context", {})
        state["shared_context"]["research_type"] = research_type

        try:
            final: str = ""

            if research_type == "search":
                final = await self._handle_search(message, context)
            elif research_type == "deep":
                final = await self._handle_deep(message, context)
            elif research_type == "summarize":
                final = await self._handle_summarize(message, context)
            elif research_type == "quick":
                final = await self._handle_quick(message, context)
            else:
                # Fallback: quick answer.
                final = await self._handle_quick(message, context)

            state["final_response"] = final
            state["response_agent"] = self.name
            state["shared_context"]["research_result"] = final

            # Persist knowledge for cross-agent reuse.
            await self._store_knowledge(
                title=self._extract_title(message, final),
                content=final,
                source=f"research_agent:{research_type}",
                tags=[research_type, "auto-generated"],
                url=None,
                metadata={
                    "research_type": research_type,
                    "user_id": state.get("user_id"),
                },
            )

            await self._store_agent_log(
                state,
                action=f"research_{research_type}",
                input_summary=message,
                output_summary=final,
                status="success",
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            state.setdefault("errors", []).append({
                "step_id": f"{self.name}_{int(time.time() * 1000)}",
                "agent": self.name,
                "error": error_msg,
                "retry_count": state.get("retry_count", 0),
                "timestamp": time.time(),
            })
            state["final_response"] = (
                f"I encountered an error while processing your research request "
                f"(type: {research_type}).\n\n{error_msg}"
            )
            state["response_agent"] = self.name

            await self._store_agent_log(
                state,
                action=f"research_{research_type}",
                input_summary=message,
                output_summary="",
                status="failed",
                error=error_msg,
            )

        return state

    # ── Research type detection ─────────────────────────────────────────

    @staticmethod
    def _detect_research_type(message: str) -> str:
        """Return ``"search"``, ``"deep"``, ``"summarize"``, or ``"quick"``.

        Uses word-boundary-aware keyword scoring so that a word like
        ``"search"`` does not accidentally match inside ``"research"``.
        The group with the highest match count wins.
        """
        msg_lower = message.lower()
        scores: Dict[str, int] = {}
        for rtype, keywords in RESEARCH_KEYWORDS.items():
            count = 0
            for kw in keywords:
                # Use word-boundary matching so "search" does not match
                # inside "research" or "searched".
                if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                    count += 1
            scores[rtype] = count

        # "summarize" requires an explicit keyword — short messages without
        # a strong signal default to "quick".
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "quick"

    # ── Task handlers ───────────────────────────────────────────────────

    async def _handle_search(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> str:
        """Perform a web search and synthesise results via Minimax.

        1. Generate optimised search queries from the user message.
        2. Execute the search via ``search_service.search``.
        3. Ask Minimax to synthesise the results into a coherent answer.
        """
        # Step 1: Generate good search queries.
        raw_queries = await minimax.web_search_query(message)
        queries = [q.strip("- \n") for q in raw_queries.split("\n") if q.strip()]
        queries = queries[:3]  # Keep top 3.

        # Include the original message as the first query.
        all_queries = [message, *queries]

        # Step 2: Execute searches.
        raw_results: List[Dict[str, Any]] = []
        seen_urls: set = set()
        for query in all_queries:
            results = await search_service.search(query, num_results=5)
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_results.append(r)

        # Step 3: Synthesise with Minimax.
        if not raw_results:
            return (
                "I searched the web but couldn't find relevant results. "
                "Try rephrasing your query or providing more context."
            )

        sources_text = "\n\n".join(
            f"Source {i + 1}: {r.get('title', 'Untitled')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Snippet: {r.get('snippet', '')}"
            for i, r in enumerate(raw_results[:15])
        )

        context_str = self._build_context_str(context)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research synthesiser. Based *only* on the "
                    "web search results provided below, answer the user's "
                    "question. Cite sources by number [1], [2], etc.\n\n"
                    "If the results don't fully answer the question, say so.\n"
                    "Format your answer with headings and bullet points."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_str}\n\n"
                    f"Question: {message}\n\n"
                    f"Search Results:\n{sources_text}"
                ),
            },
        ]
        response = await minimax.chat(messages, temperature=0.3)
        return response["choices"][0]["message"]["content"]

    async def _handle_deep(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> str:
        """Perform deep research using Minimax's structured research pipeline.

        Gathers supplementary web context first, then delegates the actual
        report generation to ``minimax.research()``.
        """
        # Gather broad web context first.
        web_results = await search_service.search(message, num_results=15)
        web_context: str = ""
        if web_results:
            web_context = "\n".join(
                f"- {r.get('title', '')}: {r.get('snippet', '')}"
                for r in web_results[:10]
            )

        context_str = self._build_context_str(context)
        combined_context = f"{context_str}\n\nWeb search results:\n{web_context}" if web_context else context_str

        depth = "deep" if len(message.split()) > 10 else "moderate"
        report = await minimax.research(
            topic=message,
            context=combined_context or None,
            depth=depth,
        )

        return self._format_deep_report(report)

    async def _handle_summarize(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> str:
        """Summarise content.

        If the message contains a URL, fetch the page content first.
        Otherwise summarise the message itself, enriched with any context.
        """
        # Check for URL in message.
        urls = AgentTools.extract_urls(message)
        text_to_summarize: str = message

        if urls:
            # Try to extract content from the first URL.
            fetched = await search_service.extract_content(urls[0])
            if fetched:
                text_to_summarize = (
                    f"URL: {urls[0]}\n\nContent:\n{fetched[:8000]}"
                )

        context_str = self._build_context_str(context)
        if context_str:
            text_to_summarize = f"Context:\n{context_str}\n\n{text_to_summarize}"

        # Determine desired length from message clues.
        max_words = 200
        if any(w in message.lower() for w in ["brief", "short", "one sentence"]):
            max_words = 50
        elif any(w in message.lower() for w in ["detailed", "comprehensive", "thorough"]):
            max_words = 500

        summary = await minimax.summarize(
            text=text_to_summarize,
            max_length=max_words,
            format="paragraph",
        )
        return summary

    async def _handle_quick(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> str:
        """Quick factual answer via Minimax chat, optionally augmented with
        a lightweight web search for freshness."""
        context_str = self._build_context_str(context)

        # For time-sensitive topics, fetch a couple of search snippets.
        web_context: str = ""
        if self._needs_fresh_data(message):
            results = await search_service.search(message, num_results=5)
            if results:
                web_context = "\n".join(
                    f"- {r.get('title', '')}: {r.get('snippet', '')}"
                    for r in results[:5]
                )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions concisely. "
                    "Give factual, well-structured answers. If you use web "
                    "search results, cite them. If you're unsure, say so."
                ),
            },
        ]
        user_content = message
        if context_str:
            user_content = f"Context:\n{context_str}\n\n{message}"
        if web_context:
            user_content += f"\n\nWeb search results:\n{web_context}"
        messages.append({"role": "user", "content": user_content})

        response = await minimax.chat(messages, temperature=0.4, max_tokens=2048)
        return response["choices"][0]["message"]["content"]

    # ── Knowledge persistence ───────────────────────────────────────────

    async def _store_knowledge(
        self,
        title: str,
        content: str,
        source: str,
        tags: List[str],
        url: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Persist research output to MongoDB's ``knowledge`` collection.

        Failures are silently absorbed so they never break the main flow.
        """
        try:
            doc = new_knowledge_doc(
                title=title,
                content=content[:5000],
                source=source,
                tags=tags,
                url=url,
                metadata=metadata,
            )
            await mongodb.knowledge.insert_one(doc)
        except Exception:
            pass  # Storage failure is non-critical.

    # ── Formatting ──────────────────────────────────────────────────────

    @staticmethod
    def _format_deep_report(report: Dict[str, Any]) -> str:
        """Convert the structured dict from ``minimax.research`` into
        a readable markdown report."""
        lines: List[str] = []

        title = report.get("title") or report.get("topic", "Research Report")
        lines.append(f"# {title}\n")

        summary = report.get("executive_summary")
        if summary:
            lines.append(f"**Executive Summary:** {summary}\n")

        findings = report.get("key_findings", [])
        if findings:
            lines.append("## Key Findings\n")
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")

        analysis = report.get("detailed_analysis")
        if analysis:
            lines.append("\n## Detailed Analysis\n")
            lines.append(analysis)

        sources = report.get("sources", [])
        if sources:
            lines.append("\n## Sources\n")
            for s in sources:
                title_s = s.get("title", "Untitled")
                url_s = s.get("url", "")
                relevance = s.get("relevance", "")
                source_line = f"- **{title_s}**"
                if url_s:
                    source_line += f" ({url_s})"
                if relevance:
                    source_line += f" — {relevance}"
                lines.append(source_line)

        conclusions = report.get("conclusions")
        if conclusions:
            lines.append(f"\n## Conclusions\n{conclusions}")

        further = report.get("further_reading", [])
        if further:
            lines.append("\n## Further Reading\n")
            for item in further:
                lines.append(f"- {item}")

        if not (summary or findings or analysis):
            # Fallback for non-JSON / unstructured responses.
            lines.append(str(report.get("raw", report)))

        return "\n".join(lines)

    # ── Utility helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_context_str(context: Dict[str, Any]) -> str:
        """Flatten shared context into a plain-text string."""
        if not context:
            return ""
        skip_keys = {"research_result", "research_type"}
        parts: List[str] = []
        for k, v in context.items():
            if k in skip_keys:
                continue
            v_str = str(v)[:600]
            parts.append(f"{k}: {v_str}")
        return "\n".join(parts)

    @staticmethod
    def _extract_title(message: str, response: str) -> str:
        """Derive a concise title from the user message."""
        # Use first sentence of the message, capped at 80 chars.
        title = message.strip().split("\n")[0].split(".")[0]
        if len(title) > 80:
            title = title[:77] + "..."
        return title or "Research Result"

    @staticmethod
    def _needs_fresh_data(message: str) -> bool:
        """Heuristic: does *message* likely need up-to-date web data?"""
        msg_lower = message.lower()
        freshness_indicators = [
            "latest", "recent", "current", "today", "this week", "this month",
            "news", "update", "price", "stock", "weather", "forecast",
            "election", "score", "result", "2024", "2025", "2026",
        ]
        return any(indicator in msg_lower for indicator in freshness_indicators)

    def __repr__(self) -> str:
        return f"<ResearchAgent model={self.model_name}>"
