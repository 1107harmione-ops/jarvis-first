"""
Research Agent — web search, knowledge extraction, summarization, report generation.
Powered by Minimax M2.1 for long-context research and synthesis.
"""

from __future__ import annotations

import time
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc, new_knowledge_doc, serialize_doc
from backend.llm.minimax import minimax
from backend.services.search_service import search_service
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ResearchAgent:
    """Specialized agent for research, analysis, and knowledge tasks."""

    def __init__(self) -> None:
        self.name = "research_agent"

    async def process(
        self,
        user_id: str,
        message: str,
        context: list[dict[str, str]] | None = None,
        attachments: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a research request."""
        start = time.monotonic()
        session_id = session_id or f"research_{int(start)}"

        task_type = self._detect_task_type(message)
        logger.info(
            "Research agent processing",
            extra={"task_type": task_type, "session_id": session_id, "user_id": user_id},
        )

        try:
            if task_type == "web_search":
                result = await self._web_search(message, context)
            elif task_type == "summarization":
                result = await self._summarize(message, context)
            elif task_type == "deep_research":
                result = await self._deep_research(message, context)
            else:
                result = await self._quick_research(message, context)

            # Store extracted knowledge
            if task_type in ("deep_research", "web_search"):
                await self._store_knowledge(user_id, message, result.get("content", ""))

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
                    tokens_used=result.get("tokens_used", 0),
                    status="success",
                )
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Research agent failed", extra={"error": str(exc), "session_id": session_id})
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary="",
                    duration_ms=elapsed,
                    status="error",
                    error=str(exc),
                )
            )
            return {"content": f"Research failed: {str(exc)}", "agent": self.name, "error": str(exc)}

    async def _web_search(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Search the web and synthesize results."""
        # Generate optimized search queries
        search_queries = await minimax.web_search_query(message)
        queries = [q.strip() for q in search_queries.split("\n") if q.strip()]

        # Execute searches
        all_results: list[dict[str, Any]] = []
        for query in queries[:3]:  # Limit to 3 queries
            try:
                results = await search_service.search(query, num_results=5)
                all_results.extend(results)
            except Exception as exc:
                logger.warning("Search query failed", extra={"query": query, "error": str(exc)})

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[dict[str, Any]] = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Synthesize results
        sources_text = "\n\n".join(
            f"Source: {r.get('title', 'Untitled')}\nURL: {r.get('url', '')}\n{r.get('snippet', '')}"
            for r in unique_results[:10]
        )

        messages = context or []
        messages = [m for m in messages if m["role"] != "system"]
        messages.insert(0, {
            "role": "system",
            "content": f"""You are a research analyst. Synthesize the following web search results into a
comprehensive answer. Cite sources inline. If information is insufficient, say so.

Web Search Results:
{sources_text}""",
        })
        messages.append({"role": "user", "content": message})

        response = await minimax.chat(messages, temperature=0.3, max_tokens=4096)
        content = response["choices"][0]["message"]["content"]

        # Build source list
        sources = [
            {"title": r.get("title", "Untitled"), "url": r.get("url", "")}
            for r in unique_results[:5]
        ]

        return {
            "content": content,
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"task": "web_search", "sources": sources, "queries_used": queries},
        }

    async def _deep_research(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Perform deep research with structured report."""
        # First, gather context from web
        search_results = await search_service.search(message, num_results=10)
        context_text = "\n\n".join(
            f"{r.get('title', '')}: {r.get('snippet', '')}"
            for r in search_results
        )

        # Generate structured research report
        report = await minimax.research(message, context=context_text, depth="deep")

        content = f"""# {report.get('title', 'Research Report')}

## Executive Summary
{report.get('executive_summary', '')}

## Key Findings
{chr(10).join(f'- {f}' for f in report.get('key_findings', []))}

## Detailed Analysis
{report.get('detailed_analysis', '')}

## Conclusions
{report.get('conclusions', '')}

## Sources
{chr(10).join(f'- {s.get("title", "")}: {s.get("url", "")}' for s in report.get('sources', []))}
"""

        return {
            "content": content,
            "agent": self.name,
            "metadata": {"task": "deep_research", "sources": report.get("sources", [])},
        }

    async def _summarize(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Summarize provided text."""
        # Extract the text to summarize from the message or context
        text_to_summarize = message
        if context:
            last_user = next((m["content"] for m in reversed(context) if m["role"] == "user"), "")
            if len(last_user) > len(message):
                text_to_summarize = last_user

        summary = await minimax.summarize(text_to_summarize)
        return {
            "content": summary,
            "agent": self.name,
            "metadata": {"task": "summarization"},
        }

    async def _quick_research(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Quick research using Minimax's built-in knowledge."""
        messages = context or []
        messages = [m for m in messages if m["role"] != "system"]
        messages.insert(0, {
            "role": "system",
            "content": "You are a knowledgeable research assistant. Provide accurate, well-structured answers. Include relevant facts, data, and context. If you're unsure, acknowledge uncertainty.",
        })
        messages.append({"role": "user", "content": message})
        response = await minimax.chat(messages, temperature=0.3, max_tokens=4096)

        return {
            "content": response["choices"][0]["message"]["content"],
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"task": "quick_research"},
        }

    async def _store_knowledge(self, user_id: str, query: str, content: str) -> None:
        """Store research results as knowledge."""
        try:
            doc = new_knowledge_doc(
                title=f"Research: {query[:100]}",
                content=content[:5000],
                source="research_agent",
                tags=["research", "web"],
            )
            await mongodb.knowledge.insert_one(doc)
        except Exception as exc:
            logger.warning("Knowledge storage failed", extra={"error": str(exc)})

    def _detect_task_type(self, message: str) -> str:
        """Detect the research task type."""
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["search for", "search web", "look up", "find online", "google"]):
            return "web_search"
        if any(kw in msg_lower for kw in ["summarize", "summary", "tl;dr", "tldr", "condense"]):
            return "summarization"
        if any(kw in msg_lower for kw in ["deep research", "comprehensive report", "thorough analysis", "in-depth"]):
            return "deep_research"
        return "quick_research"


# Global singleton
research_agent = ResearchAgent()
