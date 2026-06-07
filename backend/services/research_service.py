"""
Research Service — core research pipeline.
Handles planning, searching, scoring, verification, and report generation.

Public interface:
    research_service = ResearchService()

Typical workflow:
    1. plan_research(message) -> dict (type, queries, depth, sources_needed)
    2. execute_search(plan) -> list[dict] (raw search results)
    3. collect_sources(raw_results) -> list[dict] (with content)
    4. rank_sources(collected, query) -> list[dict] (sorted by score)
    5. verify_facts(sources, message) -> dict (fact-check results)
    6. _synthesize(query, top_sources) -> str (LLM synthesis)
    7. _generate_report(synthesis, verification, type, depth) -> dict

Shortcuts:
    quick_search(query, max_sources) -> dict
    deep_research(query, type, depth, max_sources) -> dict
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId

from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.database.schemas import new_research_cache_doc, new_research_report_doc
from backend.llm.deepseek import deepseek
from backend.llm.minimax import minimax
from backend.services.search_service import search_service
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Research Type Keywords ─────────────────────────────────────

RESEARCH_TYPE_KEYWORDS: dict[str, list[str]] = {
    "quick": [
        "what is",
        "who is",
        "when",
        "where",
        "define",
        "tell me",
        "hello",
        "hi",
        "how are you",
        "thanks",
        "good morning",
    ],
    "deep": [
        "comprehensive",
        "in-depth",
        "deep research",
        "thorough",
        "detailed analysis",
        "extensive",
        "complete analysis",
        "full report",
        "in depth",
    ],
    "comparative": [
        "compare",
        "contrast",
        "vs",
        "versus",
        "differences",
        "similarities",
        "comparison",
        "pros and cons",
        "compared to",
        "versus",
    ],
    "technical": [
        "technical",
        "specification",
        "how does",
        "technical details",
        "implementation",
        "specs",
        "technical specs",
        "how it works",
        "under the hood",
    ],
    "market": [
        "market",
        "market size",
        "market share",
        "industry",
        "trends",
        "competitive landscape",
        "market analysis",
        "industry analysis",
        "market research",
    ],
    "product": [
        "product",
        "features",
        "pricing",
        "review",
        "best",
        "top",
        "recommended",
        "buy",
        "price",
        "product review",
        "rating",
    ],
    "architecture": [
        "system design",
        "architecture",
        "data flow",
        "components",
        "tech stack",
        "microservices",
        "system architecture",
        "software architecture",
        "design pattern",
    ],
}


class ResearchService:
    """Core research pipeline service.

    Handles the complete research workflow:
    1. **Planning** — detect type, generate queries, determine depth
    2. **Searching** — execute web searches, collect results
    3. **Collecting** — extract full content from sources
    4. **Scoring** — rank sources by quality metrics
    5. **Verification** — fact-check claims against sources
    6. **Synthesis** — combine findings into coherent answer
    7. **Report generation** — produce structured research report
    """

    def __init__(self) -> None:
        self._known_domains: dict[str, float] = {
            "nature.com": 0.2,
            "sciencemag.org": 0.2,
            "sciencedirect.com": 0.2,
            "reuters.com": 0.15,
            "bloomberg.com": 0.15,
            "wsj.com": 0.15,
            "wikipedia.org": 0.1,
            "blogspot.com": -0.1,
            "medium.com": -0.1,
            "wordpress.com": -0.1,
        }

    # ── Public Methods ────────────────────────────────────────────

    async def plan_research(
        self, message: str, context: str | None = None
    ) -> dict[str, Any]:
        """Plan a research task: determine type, depth, and generate search queries.

        Args:
            message: The research query or topic.
            context: Optional additional context for planning.

        Returns:
            Dict with ``queries``, ``depth``, ``research_type``, ``sources_needed``.
        """
        research_type = await self.detect_research_type(message)
        queries = await self._generate_queries(message, research_type)
        depth = self._determine_depth(message, research_type)
        sources_needed = self._sources_for_depth(depth)

        return {
            "queries": queries,
            "depth": depth,
            "research_type": research_type,
            "sources_needed": sources_needed,
        }

    async def detect_research_type(self, message: str) -> str:
        """Detect the research type from a message using keyword matching.

        Returns one of: ``quick``, ``deep``, ``comparative``, ``technical``,
        ``market``, ``product``, ``architecture``.  Defaults to ``quick``.
        """
        message_lower = message.lower()
        best_type = "quick"
        best_score = 0

        for rtype, keywords in RESEARCH_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > best_score:
                best_score = score
                best_type = rtype

        return best_type

    async def execute_search(
        self, plan: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute web searches for each query in the plan.

        Results are deduplicated by URL.  Stops when ``sources_needed``
        results have been collected.

        Returns:
            List of result dicts with ``title``, ``url``, ``snippet`` keys.
        """
        all_results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        queries = plan.get("queries", [plan.get("query", "")])
        sources_needed = plan.get("sources_needed", 10)

        for query in queries:
            try:
                results = await search_service.search(
                    query, num_results=min(sources_needed, 10)
                )
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
            except Exception as exc:
                logger.warning(
                    "Search failed for query",
                    extra={"query": query, "error": str(exc)},
                )

            if len(all_results) >= sources_needed:
                break

        return all_results[:sources_needed]

    async def collect_sources(
        self, raw_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract full content and metadata from raw search results.

        Each source gets a ``domain`` field and full-page ``content``
        (falling back to the snippet if extraction fails).
        """
        collected: list[dict[str, Any]] = []

        for result in raw_results:
            url = result.get("url", "")
            domain = self._extract_domain(url)

            content: str | None = None
            try:
                content = await search_service.extract_content(url)
            except Exception as exc:
                logger.warning(
                    "Content extraction failed",
                    extra={"url": url, "error": str(exc)},
                )

            collected.append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
                "domain": domain,
                "content": (content or result.get("snippet", "")),
                "published_date": result.get("published_date"),
                "author": result.get("author"),
            })

        return collected

    async def rank_sources(
        self, collected: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Score and rank sources by quality metrics, highest first.

        Each source is scored by :meth:`score_source` and sorted by
        ``overall_score`` descending.
        """
        scored_sources: list[dict[str, Any]] = []

        for source in collected:
            scored = await self.score_source(source, query)
            scored_sources.append(scored)

        scored_sources.sort(key=lambda s: s.get("overall_score", 0), reverse=True)
        return scored_sources

    async def score_source(
        self, source: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Score a single source on multiple quality dimensions.

        Dimension scores (0.0–1.0):
        - ``authority_score`` (30 % weight)
        - ``freshness_score`` (20 %)
        - ``accuracy_score`` (25 %)
        - ``relevance_score`` (15 %)
        - ``popularity_score`` (10 %)

        Returns:
            Source dict with all dimension scores and ``overall_score``.
        """
        domain = source.get(
            "domain", self._extract_domain(source.get("url", ""))
        )
        author = source.get("author")
        content = source.get("content", "")
        snippet = source.get("snippet", "")
        title = source.get("title", "")
        published_date = source.get("published_date")

        authority = self._compute_authority(domain, author)
        freshness = self._compute_freshness(published_date)
        accuracy = self._compute_accuracy(content, snippet)
        relevance = self._compute_relevance(query, title, snippet)
        popularity = self._compute_popularity(source)

        overall = (
            authority * 0.30
            + freshness * 0.20
            + accuracy * 0.25
            + relevance * 0.15
            + popularity * 0.10
        )

        return {
            "title": title,
            "url": source.get("url", ""),
            "snippet": snippet,
            "domain": domain,
            "authority_score": round(authority, 4),
            "freshness_score": round(freshness, 4),
            "accuracy_score": round(accuracy, 4),
            "relevance_score": round(relevance, 4),
            "popularity_score": round(popularity, 4),
            "overall_score": round(overall, 4),
            "content": content,
            "published_date": published_date,
            "author": author,
        }

    async def verify_facts(
        self, sources: list[dict[str, Any]], message: str
    ) -> dict[str, Any]:
        """Verify factual claims in a message against the provided sources.

        Uses DeepSeek ``extract_json`` to extract:
        - ``verified_claims`` — claims supported by sources
        - ``contradicted_claims`` — claims contradicted by sources
        - ``unverifiable_claims`` — claims neither supported nor contradicted
        - ``overall_confidence`` — float 0.0–1.0

        When no sources are provided, returns zero-confidence empty result.
        """
        if not sources:
            return {
                "verified_claims": [],
                "contradicted_claims": [],
                "unverifiable_claims": [],
                "overall_confidence": 0.0,
            }

        source_texts = "\n\n".join(
            f"Source {i + 1}: {s.get('title', '')}\n"
            f"URL: {s.get('url', '')}\n"
            f"{s.get('snippet', '')[:500]}"
            for i, s in enumerate(sources[:10])
        )

        system_prompt = (
            "You are a fact-checking analyst. Given a query and source texts, extract:\n"
            "1. verified_claims — claims SUPPORTED by sources\n"
            "2. contradicted_claims — claims CONTRADICTED by sources\n"
            "3. unverifiable_claims — claims neither supported nor contradicted\n"
            "4. overall_confidence — float 0.0 to 1.0\n\n"
            "Return JSON:\n"
            "{\n"
            '  "verified_claims": ["..."],\n'
            '  "contradicted_claims": ["..."],\n'
            '  "unverifiable_claims": ["..."],\n'
            '  "overall_confidence": 0.0\n'
            "}"
        )

        user_message = f"Query: {message}\n\nSources:\n{source_texts}"

        try:
            result = await deepseek.extract_json(system_prompt, user_message)
            return {
                "verified_claims": result.get("verified_claims", []),
                "contradicted_claims": result.get("contradicted_claims", []),
                "unverifiable_claims": result.get("unverifiable_claims", []),
                "overall_confidence": float(
                    result.get("overall_confidence", 0.0)
                ),
            }
        except Exception as exc:
            logger.error(
                "Fact verification failed", extra={"error": str(exc)}
            )
            return {
                "verified_claims": [],
                "contradicted_claims": [],
                "unverifiable_claims": [],
                "overall_confidence": 0.0,
            }

    async def quick_search(
        self, query: str, max_sources: int = 10
    ) -> dict[str, Any]:
        """Execute a quick search with minimal processing.

        Shortcut that runs the full pipeline (plan → search → collect → rank
        → synthesise) and returns a lightweight result.  Suitable for simple
        questions where a deep report is unnecessary.

        Returns:
            Dict with ``query``, ``answer``, ``sources``,
            ``source_count``, ``processing_time_ms``.
        """
        start_time = time.monotonic()

        plan = await self.plan_research(query)
        raw_results = await self.execute_search(
            {**plan, "sources_needed": max_sources}
        )
        collected = await self.collect_sources(raw_results)
        ranked = await self.rank_sources(collected, query)
        top_sources = ranked[:max_sources]

        try:
            answer = await self._synthesize(query, top_sources)
        except Exception:
            answer = self._fallback_synthesis(query, top_sources)

        elapsed = round((time.monotonic() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": answer,
            "sources": top_sources,
            "source_count": len(top_sources),
            "processing_time_ms": elapsed,
        }

    async def deep_research(
        self,
        query: str,
        research_type: str,
        depth: str,
        max_sources: int = 20,
    ) -> dict[str, Any]:
        """Execute a deep research pipeline with full report generation.

        Full workflow including caching, fact checking, structured report
        generation, and MongoDB persistence.  Checks the cache first and
        returns a cached result if one exists and is still valid.

        Returns:
            Dict with report fields (title, executive_summary, key_findings,
            …), sources, fact_check, and metadata.
        """
        start_time = time.monotonic()

        # ── Cache check ───────────────────────────────────────────
        cache_key = self._build_cache_key(query, depth, research_type)
        cached = await self._check_cache(cache_key)
        if cached is not None:
            return cached

        # ── Plan & search ─────────────────────────────────────────
        plan = await self.plan_research(query)
        plan["depth"] = depth
        plan["research_type"] = research_type
        plan["sources_needed"] = max_sources

        raw_results = await self.execute_search(plan)
        collected = await self.collect_sources(raw_results)
        ranked = await self.rank_sources(collected, query)
        top_sources = ranked[:max_sources]

        # ── Verify facts ──────────────────────────────────────────
        verification = await self.verify_facts(top_sources, query)

        # ── Synthesise ────────────────────────────────────────────
        synthesis = await self._synthesize(query, top_sources)

        # ── Generate report ───────────────────────────────────────
        report = await self._generate_report(
            synthesis, verification, research_type, depth
        )

        elapsed = round((time.monotonic() - start_time) * 1000, 2)

        result: dict[str, Any] = {
            **report,
            "research_type": research_type,
            "depth": depth,
            "sources": top_sources,
            "source_count": len(top_sources),
            "fact_check": verification,
            "processing_time_ms": elapsed,
        }

        # ── Persist ───────────────────────────────────────────────
        report_id = await self._persist_report(
            result, query, depth, research_type
        )
        result["report_id"] = report_id

        # ── Cache ─────────────────────────────────────────────────
        await self._set_cache(cache_key, result)

        return result

    # ── Private: Query Generation ──────────────────────────────────

    async def _generate_queries(
        self, message: str, research_type: str
    ) -> list[str]:
        """Generate optimised web search queries using Minimax.

        Falls back to the original message as a single query when the
        LLM call fails.
        """
        try:
            raw = await minimax.web_search_query(message)
            queries = [q.strip() for q in raw.split("\n") if q.strip()]
            if queries:
                return queries
        except Exception as exc:
            logger.warning(
                "Query generation via Minimax failed",
                extra={"error": str(exc)},
            )

        return [message]

    # ── Private: Depth ─────────────────────────────────────────────

    def _determine_depth(self, message: str, research_type: str) -> str:
        """Determine research depth based on message keywords and type."""
        message_lower = message.lower()

        depth_keywords: dict[str, list[str]] = {
            "comprehensive": [
                "comprehensive",
                "exhaustive",
                "extremely detailed",
                "max",
            ],
            "deep": ["deep", "in-depth", "thorough", "detailed", "extensive", "in depth"],
            "moderate": ["moderate", "somewhat", "basic research", "overview"],
        }

        for depth, keywords in depth_keywords.items():
            if any(kw in message_lower for kw in keywords):
                return depth

        type_depth_map: dict[str, str] = {
            "quick": "quick",
            "deep": "deep",
            "comparative": "deep",
            "technical": "moderate",
            "market": "moderate",
            "product": "moderate",
            "architecture": "deep",
        }

        return type_depth_map.get(
            research_type, settings.RESEARCH_DEFAULT_DEPTH
        )

    @staticmethod
    def _sources_for_depth(depth: str) -> int:
        """Return the number of sources needed for a given depth level."""
        depth_map: dict[str, int] = {
            "quick": 5,
            "moderate": 10,
            "deep": 20,
            "comprehensive": 30,
        }
        return depth_map.get(depth, 10)

    # ── Private: Domain parsing ────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain (with subdomain) from a URL.

        Examples::

            >>> ResearchService._extract_domain("https://www.example.com/path")
            "www.example.com"
            >>> ResearchService._extract_domain("http://example.edu")
            "example.edu"
        """
        url = url.replace("http://", "").replace("https://", "")
        if "/" in url:
            url = url.split("/")[0]
        return url

    # ── Private: Scoring helpers ───────────────────────────────────

    def _compute_authority(
        self, domain: str, author: str | None = None
    ) -> float:
        """Compute authority score (0.0–1.0) for a source domain.

        Base 0.3, with bonuses:
        - .edu / .gov / .org TLD → +0.2
        - Known authoritative domains → +0.2 / +0.15 / +0.1
        - Blog / low-credibility domains → −0.1
        - Author present → +0.05

        Clamped to [0, 1].
        """
        score = 0.3

        # TLD bonuses
        if domain.endswith(".edu"):
            score += 0.2
        elif domain.endswith(".gov"):
            score += 0.2
        elif domain.endswith(".org"):
            score += 0.2

        # Known-domain bonuses / penalties (check once)
        for known_domain, bonus in self._known_domains.items():
            if known_domain in domain or domain.endswith(f".{known_domain}"):
                score += bonus
                break

        # Author present
        if author:
            score += 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _compute_freshness(published_date: Any) -> float:
        """Compute freshness score (0.0–1.0) based on publication date.

        - No date → 0.5
        - < 7 days → 1.0
        - < 30 days → 0.9
        - < 90 days → 0.8
        - < 365 days → 0.6
        - < 730 days → 0.4
        - Otherwise → max(0.1, 1.0 − days / 3650)
        """
        if not published_date:
            return 0.5

        if isinstance(published_date, str):
            try:
                published_date = datetime.fromisoformat(published_date)
            except (ValueError, TypeError):
                return 0.5

        if isinstance(published_date, datetime):
            now = datetime.now(timezone.utc)
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)
            days_ago = (now - published_date).days
        else:
            return 0.5

        if days_ago < 0:
            return 1.0
        if days_ago < 7:
            return 1.0
        if days_ago < 30:
            return 0.9
        if days_ago < 90:
            return 0.8
        if days_ago < 365:
            return 0.6
        if days_ago < 730:
            return 0.4

        return max(0.1, 1.0 - days_ago / 3650)

    @staticmethod
    def _compute_accuracy(content: str, snippet: str) -> float:
        """Compute accuracy score (0.0–1.0) based on content quality signals.

        Base 0.5:
        - Content > 500 chars → +0.1
        - Content > 2000 chars → +0.1
        - Contains "citation" or "reference" → +0.1
        - Contains "doi.org" or "arxiv" → +0.1
        - Contains "according to" → +0.05
        """
        score = 0.5

        if len(content) > 500:
            score += 0.1
        if len(content) > 2000:
            score += 0.1

        content_lower = content.lower()
        if "citation" in content_lower or "reference" in content_lower:
            score += 0.1
        if "doi.org" in content_lower or "arxiv" in content_lower:
            score += 0.1
        if "according to" in content_lower:
            score += 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _compute_relevance(query: str, title: str, snippet: str) -> float:
        """Compute relevance score (0.0–1.0) based on keyword overlap.

        - Keyword overlap fraction × 0.8
        - Title contains exact query phrase → +0.2 bonus
        """
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        combined_text = (title.lower() + " " + snippet.lower())
        overlap = sum(1 for w in query_words if w in combined_text)

        score = (overlap / len(query_words)) * 0.8

        # Bonus for exact phrase match in title
        if query.lower() in title.lower():
            score += 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def _compute_popularity(source: dict[str, Any]) -> float:
        """Compute popularity score (0.0–1.0) for a source.

        Base 0.3:
        - Snippet > 100 chars → +0.1
        - URL path depth > 3 → +0.1
        """
        score = 0.3

        snippet = source.get("snippet", "")
        url = source.get("url", "")

        if len(snippet) > 100:
            score += 0.1

        path_depth = len([p for p in url.split("/") if p])
        if path_depth > 3:
            score += 0.1

        return max(0.0, min(1.0, score))

    # ── Private: Synthesis ─────────────────────────────────────────

    async def _synthesize(
        self, query: str, top_sources: list[dict[str, Any]]
    ) -> str:
        """Synthesise findings from top sources using Minimax.

        Falls back to a simple snippet-based summary when the LLM is
        unavailable or returns an error.
        """
        if not top_sources:
            return f"No sources found for '{query}'."

        context = "\n\n".join(
            f"Source {i + 1}: {s.get('title', '')}\n"
            f"URL: {s.get('url', '')}\n"
            f"{s.get('snippet', '')[:1000]}"
            for i, s in enumerate(top_sources[:10])
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Synthesise the following "
                    "search results into a coherent, informative answer. "
                    "Be accurate and cite sources by number where relevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Search Results:\n{context}\n\n"
                    "Provide a comprehensive answer synthesising the "
                    "information above."
                ),
            },
        ]

        try:
            response = await minimax.chat(messages, temperature=0.3)
            return response["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning(
                "Synthesis via Minimax failed", extra={"error": str(exc)}
            )
            return self._fallback_synthesis(query, top_sources)

    def _fallback_synthesis(
        self, query: str, top_sources: list[dict[str, Any]]
    ) -> str:
        """Construct a simple answer from source snippets (no LLM)."""
        if not top_sources:
            return f"Could not find information about '{query}'."

        parts = [f"Here is what I found about '{query}':\n"]
        for i, s in enumerate(top_sources[:5], 1):
            title = s.get("title", "Untitled")
            snippet = s.get("snippet", "")
            url = s.get("url", "")
            parts.append(f"{i}. {title}")
            if snippet:
                parts.append(f"   {snippet[:300]}")
            parts.append(f"   Source: {url}")
            parts.append("")

        return "\n".join(parts)

    # ── Private: Report generation ─────────────────────────────────

    async def _generate_report(
        self,
        synthesis: str,
        verification: dict[str, Any],
        research_type: str,
        depth: str,
    ) -> dict[str, Any]:
        """Generate a structured research report using DeepSeek.

        The system prompt is tailored to the *research_type* so that the
        report sections match the expected format.
        """
        section_descriptions: dict[str, str] = {
            "quick": "concise answer only",
            "deep": (
                "executive summary, key findings, detailed analysis, conclusions"
            ),
            "comparative": (
                "side-by-side comparison, pros/cons, recommendation"
            ),
            "technical": (
                "overview, specs, trade-offs, implementation notes"
            ),
            "market": (
                "trends, competitive landscape, key players, outlook"
            ),
            "product": (
                "features, pricing, reviews, pros/cons, verdict"
            ),
            "architecture": (
                "system design, components, data flow, tech stack"
            ),
        }
        section_desc = section_descriptions.get(
            research_type,
            "executive summary, key findings, detailed analysis, conclusions",
        )

        system_prompt = (
            f"You are a research report generator. Generate a structured "
            f"JSON report.\n\n"
            f"Research type: {research_type}\n"
            f"Depth: {depth}\n"
            f"Sections: {section_desc}\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '    "title": "Report title",\n'
            '    "executive_summary": "Brief summary of findings",\n'
            '    "key_findings": ["Finding 1", "Finding 2"],\n'
            '    "detailed_analysis": "In-depth analysis",\n'
            '    "pros": ["Pro 1"] | null,\n'
            '    "cons": ["Con 1"] | null,\n'
            '    "recommendations": ["Recommendation 1"] | null,\n'
            '    "conclusions": "Final conclusions"\n'
            "}"
        )

        user_message = (
            f"Synthesis:\n{synthesis}\n\n"
            f"Fact Check:\n{verification}"
        )

        try:
            result = await deepseek.extract_json(system_prompt, user_message)
            return {
                "title": result.get(
                    "title", f"Research Report ({research_type})"
                ),
                "executive_summary": result.get("executive_summary", ""),
                "key_findings": result.get("key_findings", []),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "pros": result.get("pros"),
                "cons": result.get("cons"),
                "recommendations": result.get("recommendations"),
                "conclusions": result.get("conclusions", ""),
            }
        except Exception as exc:
            logger.error(
                "Report generation failed", extra={"error": str(exc)}
            )
            return {
                "title": f"Research Report ({research_type})",
                "executive_summary": "",
                "key_findings": [],
                "detailed_analysis": synthesis if synthesis else "",
                "pros": None,
                "cons": None,
                "recommendations": None,
                "conclusions": "",
            }

    # ── Private: Caching ───────────────────────────────────────────

    @staticmethod
    def _build_cache_key(query: str, depth: str, research_type: str) -> str:
        """Build a deterministic SHA-256 cache key."""
        raw = f"{query}:{depth}:{research_type}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _check_cache(
        self, cache_key: str
    ) -> dict[str, Any] | None:
        """Check the research cache for a valid (non-expired) result."""
        try:
            doc = await mongodb.research_cache.find_one(
                {"cache_key": cache_key}
            )
            if doc is not None:
                ttl = doc.get("ttl")
                if ttl and ttl > datetime.now(timezone.utc):
                    logger.debug(
                        "Cache hit",
                        extra={"cache_key": cache_key[:16]},
                    )
                    report_id = doc.get("report_id")
                    if report_id:
                        report = await mongodb.research_reports.find_one(
                            {"_id": ObjectId(report_id)}
                        )
                        if report is not None:
                            report.pop("_id", None)
                            return report
                    return doc.get("result")
        except Exception as exc:
            logger.warning(
                "Cache check failed", extra={"error": str(exc)}
            )

        return None

    async def _set_cache(
        self, cache_key: str, result: dict[str, Any]
    ) -> None:
        """Store a research result in the cache."""
        try:
            ttl = datetime.now(timezone.utc) + timedelta(
                hours=settings.RESEARCH_CACHE_TTL_HOURS
            )
            source_urls = [
                s.get("url", "") for s in result.get("sources", [])
            ]

            cache_doc = new_research_cache_doc(
                cache_key=cache_key,
                query=result.get("query", ""),
                research_type=result.get("research_type", "deep"),
                ttl=ttl,
                report_id=result.get("report_id"),
                synthesis=result.get("executive_summary", ""),
                source_urls=source_urls,
            )
            await mongodb.research_cache.insert_one(cache_doc)
            logger.debug(
                "Cache set", extra={"cache_key": cache_key[:16]}
            )
        except Exception as exc:
            logger.warning(
                "Cache set failed", extra={"error": str(exc)}
            )

    # ── Private: Persistence ───────────────────────────────────────

    async def _persist_report(
        self,
        result: dict[str, Any],
        query: str,
        depth: str,
        research_type: str,
    ) -> str:
        """Persist a research report to MongoDB and return its ``_id``."""
        try:
            sources_for_doc = []
            for s in result.get("sources", []):
                sources_for_doc.append({
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet", ""),
                    "domain": s.get("domain", ""),
                    "overall_score": s.get("overall_score", 0.0),
                    "authority_score": s.get("authority_score", 0.0),
                    "freshness_score": s.get("freshness_score", 0.0),
                    "relevance_score": s.get("relevance_score", 0.0),
                })

            report_doc = new_research_report_doc(
                user_id="system",
                session_id="research-service",
                query=query,
                research_type=research_type,
                depth=depth,
                executive_summary=result.get("executive_summary", ""),
                key_findings=result.get("key_findings", []),
                detailed_analysis=result.get("detailed_analysis"),
                pros=result.get("pros"),
                cons=result.get("cons"),
                recommendations=result.get("recommendations"),
                conclusions=result.get("conclusions"),
                sources=sources_for_doc,
                fact_check=result.get("fact_check"),
            )

            insert_result = await mongodb.research_reports.insert_one(
                report_doc
            )
            report_id = str(insert_result.inserted_id)
            logger.info(
                "Research report persisted",
                extra={
                    "report_id": report_id,
                    "query": query[:50],
                },
            )
            return report_id
        except Exception as exc:
            logger.error(
                "Report persistence failed", extra={"error": str(exc)}
            )
            return ""


# ── Global Singleton ─────────────────────────────────────────

research_service = ResearchService()
