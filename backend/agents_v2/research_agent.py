"""
Research Agent
==============
Production-grade LangGraph agent for internet research, deep research,
source verification, fact checking, report generation, and knowledge extraction.

Pipeline: Plan -> Search -> Collect -> Rank -> Verify -> Synthesize -> Report -> Respond

Models:
  - Minimax M2.1: Search query generation, synthesis, web search
  - DeepSeek V4 Flash: Fact verification, structured report generation

The agent delegates all pipeline logic to ``ResearchService`` and focuses on:
  1. Research type detection (7 types)
  2. State management (context, results, errors)
  3. Knowledge persistence
  4. Execution logging
"""

from __future__ import annotations

import re
import time
from typing import Any

from backend.agents_v2.state import AgentState
from backend.agents_v2.base import BaseAgent
from backend.agents_v2.registry import get_agent_registry
from backend.services.research_service import research_service, RESEARCH_TYPE_KEYWORDS
from backend.database.mongodb import mongodb
from backend.database.schemas import new_knowledge_doc
from backend.config.settings import settings


class ResearchAgent(BaseAgent):
    """Agent that performs web research, deep-dive reports, source verification,
    fact checking, and knowledge extraction.

    Replaces the previous basic agent with a full 8-stage research pipeline.
    """

    def __init__(self) -> None:
        super().__init__(
            name="research",
            model_name="minimax",
            system_prompt=(
                "You are JARVIS's Research Agent, an expert research analyst. "
                "You search the web, gather sources, verify facts, synthesise "
                "information, and produce well-structured reports.\n\n"
                "Rules:\n"
                "1. Always cite sources where possible.\n"
                "2. Distinguish between established facts and speculation.\n"
                "3. When information is contradictory, note both sides.\n"
                "4. Keep responses concise unless a detailed report is requested.\n"
                "5. Use bullet points and headings for readability.\n"
                "6. Store important findings for future reference."
            ),
            description=(
                "Searches the web, gathers sources, verifies facts, and produces "
                "research reports using Minimax M2.1 and DeepSeek V4 Flash"
            ),
        )
        get_agent_registry().register(self)

    # ── Public API ──────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Execute the research agent's logic on *state*.

        Pipeline:
        1. Read message and context from state
        2. Load memory context
        3. Detect research type (7 types)
        4. Execute the appropriate research pipeline via ResearchService
        5. Store knowledge in MongoDB for future retrieval
        6. Update final_response and shared_context
        7. Log execution
        """
        message: str = state.get("message", "") or ""
        context: dict[str, Any] = state.get("shared_context", {})
        await self._load_memory_context(state)

        research_type = await self.detect_research_type(message)
        state.setdefault("shared_context", {})
        state["shared_context"]["research_type"] = research_type

        try:
            # Determine depth
            depth = self._determine_depth(message, research_type)

            if research_type == "quick":
                final = await self._handle_quick(message, context)
            elif research_type in ("deep", "comparative", "technical",
                                   "market", "product", "architecture"):
                final = await self._handle_deep(message, research_type, depth)
            else:
                final = await self._handle_quick(message, context)

            state["final_response"] = final
            state["response_agent"] = self.name
            state["shared_context"]["research_result"] = final

            # Persist knowledge for cross-agent reuse
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

    # ── Research type detection ─────────────────────────

    @staticmethod
    async def detect_research_type(message: str) -> str:
        """Detect the research type using keyword scoring.

        Returns one of: quick, deep, comparative, technical, market,
        product, architecture.
        """
        msg_lower = message.lower()
        scores: dict[str, int] = {}
        for rtype, keywords in RESEARCH_TYPE_KEYWORDS.items():
            count = 0
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                    count += 1
            scores[rtype] = count

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "quick"

    # ── Task handlers ───────────────────────────────────

    async def _handle_quick(
        self,
        message: str,
        context: dict[str, Any],
    ) -> str:
        """Quick web search with synthesis — delegates to ResearchService."""
        result = await research_service.quick_search(message, max_sources=10)
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        if not answer:
            return "I searched the web but couldn't find relevant results."

        # Append sources
        if sources:
            source_lines = []
            for i, s in enumerate(sources[:5], 1):
                title = s.get("title", "Untitled")
                url = s.get("url", "")
                source_lines.append(f"{i}. [{title}]({url})")
            answer += "\n\n**Sources:**\n" + "\n".join(source_lines)

        return answer

    async def _handle_deep(
        self,
        message: str,
        research_type: str,
        depth: str,
    ) -> str:
        """Full deep research pipeline — delegates to ResearchService."""
        result = await research_service.deep_research(
            query=message,
            research_type=research_type,
            depth=depth,
            max_sources=20,
        )

        return self._format_deep_report(result)

    # ── Formatting ──────────────────────────────────────

    @staticmethod
    def _format_deep_report(result: dict[str, Any]) -> str:
        """Convert the structured report dict to a readable markdown string."""
        lines: list[str] = []

        title = result.get("title", "Research Report")
        lines.append(f"# {title}\n")

        summary = result.get("executive_summary", "")
        if summary:
            lines.append(f"**Summary:** {summary}\n")

        findings = result.get("key_findings", [])
        if findings:
            lines.append("## Key Findings\n")
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")
            lines.append("")

        analysis = result.get("detailed_analysis")
        if analysis:
            lines.append("## Detailed Analysis\n")
            lines.append(analysis)
            lines.append("")

        pros = result.get("pros")
        cons = result.get("cons")
        if pros or cons:
            lines.append("## Pros & Cons\n")
            if pros:
                lines.append("**Pros:**")
                for p in pros:
                    lines.append(f"- {p}")
            if cons:
                lines.append("**Cons:**")
                for c in cons:
                    lines.append(f"- {c}")
            lines.append("")

        recommendations = result.get("recommendations")
        if recommendations:
            lines.append("## Recommendations\n")
            for r in recommendations:
                lines.append(f"- {r}")
            lines.append("")

        conclusions = result.get("conclusions")
        if conclusions:
            lines.append(f"## Conclusions\n{conclusions}\n")

        fact_check = result.get("fact_check")
        if fact_check and fact_check.get("verified_claims"):
            lines.append("## Fact Check\n")
            for claim in fact_check.get("verified_claims", []):
                lines.append(f"- {claim}")
            for claim in fact_check.get("contradicted_claims", []):
                lines.append(f"- {claim}")
            for claim in fact_check.get("unverifiable_claims", []):
                lines.append(f"- {claim}")
            confidence = fact_check.get("overall_confidence", 0)
            lines.append(f"\n*Fact-check confidence: {confidence:.0%}*")
            lines.append("")

        sources = result.get("sources", [])
        if sources:
            lines.append("## Sources\n")
            for s in sources[:10]:
                title_s = s.get("title", "Untitled")
                url_s = s.get("url", "")
                score = s.get("overall_score", 0)
                if url_s:
                    lines.append(f"- **[{title_s}]({url_s})** (score: {score:.2f})")
                else:
                    lines.append(f"- **{title_s}** (score: {score:.2f})")

        return "\n".join(lines)

    # ── Knowledge persistence ───────────────────────────

    async def _store_knowledge(
        self,
        title: str,
        content: str,
        source: str,
        tags: list[str],
        url: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Persist research output to MongoDB's knowledge collection."""
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
            pass  # Storage failure is non-critical

    # ── Utility helpers ─────────────────────────────────

    @staticmethod
    def _determine_depth(message: str, research_type: str) -> str:
        """Determine research depth from message length and type."""
        if research_type == "quick":
            return "quick"
        word_count = len(message.split())
        if word_count > 30:
            return "deep"
        if word_count > 15:
            return "moderate"
        return settings.RESEARCH_DEFAULT_DEPTH

    @staticmethod
    def _extract_title(message: str, response: str) -> str:
        """Derive a concise title from the user message."""
        title = message.strip().split("\n")[0].split(".")[0]
        if len(title) > 80:
            title = title[:77] + "..."
        return title or "Research Result"

    def __repr__(self) -> str:
        return f"<ResearchAgent model={self.model_name}>"
