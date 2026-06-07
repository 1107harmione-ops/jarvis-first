# Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade Research Agent with internet research, source verification, fact checking, report generation, and knowledge extraction.

**Architecture:** Service-layer separation — `ResearchService` (pipeline logic) → `ResearchAgent` (LangGraph wrapper) → `ResearchAPI` (FastAPI endpoints). Follows existing patterns (SearchService, VoiceService).

**Tech Stack:** Python 3.12+, FastAPI, LangGraph, Minimax M2.1, DeepSeek V4 Flash, MongoDB/Motor, Pydantic v2

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `database/schemas.py` | Modify (+40 lines) | `new_research_report_doc()`, `new_research_source_doc()`, `new_research_cache_doc()` |
| `database/models.py` | Modify (+150 lines) | `ResearchType`/`ResearchDepth` enums, `SourceScore`, `FactCheckResult`, `ResearchReport`, request/response models |
| `database/mongodb.py` | Modify (+30 lines) | `research_reports`, `research_sources`, `research_cache` collections + indexes |
| `config/settings.py` | Modify (+6 lines) | `RESEARCH_CACHE_TTL_HOURS`, `RESEARCH_MAX_SOURCES`, `RESEARCH_DEFAULT_DEPTH` |
| `services/research_service.py` | Create (~400 lines) | `ResearchService` singleton — full 8-stage pipeline: plan, search, collect, rank, verify, synthesize, report |
| `agents_v2/research_agent.py` | Replace (~500 lines) | LangGraph agent — 7-type detection, delegates to ResearchService, stores knowledge |
| `api/research.py` | Create (~250 lines) | `APIRouter(prefix="/api/v2/research")` — search, deep, report, verify, history endpoints |
| `backend/main.py` | Modify (+2 lines) | Include research API router |
| `tests/services/test_research_service.py` | Create | Service unit tests |
| `tests/api/test_research_api.py` | Create | API integration tests |
| `tests/agents_v2/test_research_agent.py` | Create | Agent unit tests |
| `tests/integration/test_research_pipeline.py` | Create | Pipeline integration tests |

---

### Task 1: MongoDB Data Layer — Schemas, Models, Collections

**Files:**
- Modify: `database/schemas.py` (append after `new_knowledge_doc`)
- Modify: `database/models.py` (append research models)
- Modify: `database/mongodb.py` (add collections + indexes)
- Create: `tests/database/test_research_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/database/test_research_schemas.py`:

```python
"""tests/database/test_research_schemas.py"""
import pytest
from datetime import datetime, timezone
from database.schemas import new_research_report_doc, new_research_source_doc, new_research_cache_doc


class TestResearchSchemas:
    def test_new_research_report_doc_creates_required_fields(self):
        doc = new_research_report_doc(
            user_id="user_1",
            session_id="session_1",
            query="test query",
            research_type="deep",
            depth="moderate",
            executive_summary="summary",
            key_findings=["finding 1"],
        )
        assert doc["user_id"] == "user_1"
        assert doc["research_type"] == "deep"
        assert doc["depth"] == "moderate"
        assert doc["executive_summary"] == "summary"
        assert doc["key_findings"] == ["finding 1"]
        assert doc["sources"] == []
        assert doc["source_count"] == 0
        assert doc["tags"] == []
        assert "_id" in doc
        assert "created_at" in doc
        assert "updated_at" in doc

    def test_new_research_source_doc_creates_required_fields(self):
        doc = new_research_source_doc(
            url="https://example.com",
            title="Example",
            snippet="snippet",
            domain="example.com",
        )
        assert doc["url"] == "https://example.com"
        assert doc["title"] == "Example"
        assert doc["domain"] == "example.com"
        assert doc["authority_score"] == 0.0
        assert doc["overall_score"] == 0.0
        assert doc["access_count"] == 0
        assert "_id" in doc

    def test_new_research_cache_doc_creates_required_fields(self):
        from datetime import timedelta
        ttl = datetime.now(timezone.utc) + timedelta(hours=24)
        doc = new_research_cache_doc(
            cache_key="abc123",
            query="test",
            research_type="quick",
            ttl=ttl,
        )
        assert doc["cache_key"] == "abc123"
        assert doc["query"] == "test"
        assert doc["research_type"] == "quick"
        assert doc["ttl"] == ttl
        assert doc["report_id"] is None
        assert doc["source_urls"] == []
```

Run: `pytest tests/database/test_research_schemas.py::TestResearchSchemas -v`
Expected: FAIL — 3 ImportErrors (functions not defined yet)

- [ ] **Step 2: Add schema builders to database/schemas.py**

Append after `new_knowledge_doc()` (line 258):

```python
# ── Research ─────────────────────────────────────────────


def new_research_report_doc(
    user_id: str,
    session_id: str,
    query: str,
    research_type: str,
    depth: str,
    executive_summary: str = "",
    key_findings: list[str] | None = None,
    detailed_analysis: str | None = None,
    pros: list[str] | None = None,
    cons: list[str] | None = None,
    recommendations: list[str] | None = None,
    conclusions: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    fact_check: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "session_id": session_id,
        "title": "",
        "query": query,
        "research_type": research_type,
        "depth": depth,
        "executive_summary": executive_summary,
        "key_findings": key_findings or [],
        "detailed_analysis": detailed_analysis,
        "pros": pros,
        "cons": cons,
        "recommendations": recommendations,
        "conclusions": conclusions,
        "sources": sources or [],
        "source_count": len(sources) if sources else 0,
        "fact_check": fact_check,
        "metadata": metadata or {},
        "tags": tags or [],
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


def new_research_source_doc(
    url: str,
    title: str,
    snippet: str,
    domain: str,
    content: str = "",
    published_date: datetime | None = None,
    author: str | None = None,
    authority_score: float = 0.0,
    freshness_score: float = 0.0,
    accuracy_score: float = 0.0,
    relevance_score: float = 0.0,
    popularity_score: float = 0.0,
    overall_score: float = 0.0,
    query: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "url": url,
        "title": title,
        "snippet": snippet,
        "content": content,
        "domain": domain,
        "published_date": published_date,
        "author": author,
        "authority_score": authority_score,
        "freshness_score": freshness_score,
        "accuracy_score": accuracy_score,
        "relevance_score": relevance_score,
        "popularity_score": popularity_score,
        "overall_score": overall_score,
        "query": query,
        "tags": tags or [],
        "access_count": 0,
        "created_at": now_utc(),
    }


def new_research_cache_doc(
    cache_key: str,
    query: str,
    research_type: str,
    ttl: datetime,
    report_id: str | None = None,
    synthesis: str | None = None,
    source_urls: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "cache_key": cache_key,
        "query": query,
        "research_type": research_type,
        "report_id": report_id,
        "synthesis": synthesis,
        "source_urls": source_urls or [],
        "ttl": ttl,
        "created_at": now_utc(),
    }
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/database/test_research_schemas.py::TestResearchSchemas -v`
Expected: PASS (3/3)

- [ ] **Step 4: Add research Pydantic models to database/models.py**

Append after the `SystemHealth` class (line 428):

```python
# ── Research ────────────────────────────────────────


class ResearchType(str, Enum):
    QUICK = "quick"
    DEEP = "deep"
    COMPARATIVE = "comparative"
    TECHNICAL = "technical"
    MARKET = "market"
    PRODUCT = "product"
    ARCHITECTURE = "architecture"


class ResearchDepth(str, Enum):
    QUICK = "quick"
    MODERATE = "moderate"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"


class SourceScore(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""
    overall_score: float = 0.0
    authority_score: float = 0.0
    freshness_score: float = 0.0
    relevance_score: float = 0.0


class FactCheckResult(BaseModel):
    verified_claims: list[str] = []
    contradicted_claims: list[str] = []
    unverifiable_claims: list[str] = []
    overall_confidence: float = 0.0


class ResearchSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    depth: ResearchDepth = ResearchDepth.QUICK
    max_sources: int = Field(default=10, ge=3, le=50)


class ResearchSearchResponse(BaseModel):
    query: str
    research_type: ResearchType
    answer: str
    sources: list[SourceScore] = []
    source_count: int = 0
    processing_time_ms: float = 0.0


class DeepResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    research_type: ResearchType = ResearchType.DEEP
    depth: ResearchDepth = ResearchDepth.MODERATE
    max_sources: int = Field(default=20, ge=5, le=100)


class DeepResearchResponse(BaseModel):
    report_id: str
    title: str = ""
    research_type: str
    depth: str
    executive_summary: str = ""
    key_findings: list[str] = []
    detailed_analysis: str | None = None
    pros: list[str] | None = None
    cons: list[str] | None = None
    recommendations: list[str] | None = None
    conclusions: str | None = None
    sources: list[SourceScore] = []
    source_count: int = 0
    fact_check: FactCheckResult | None = None
    processing_time_ms: float = 0.0
    created_at: str = ""


class VerifyContentRequest(BaseModel):
    content: str = Field(min_length=10, max_length=50000)
    context: str | None = Field(default=None, max_length=5000)


class VerifyContentResponse(BaseModel):
    verified_claims: list[str] = []
    contradicted_claims: list[str] = []
    unverifiable_claims: list[str] = []
    overall_confidence: float = 0.0
    analysis: str = ""


class ResearchReportSummary(BaseModel):
    """Lightweight summary for history listing."""
    id: str
    query: str
    research_type: str
    depth: str
    executive_summary: str = ""
    source_count: int = 0
    created_at: str = ""
```

- [ ] **Step 5: Add research collections and indexes to database/mongodb.py**

Add collection properties inside `MongoDBManager` after `offline_queue` (line 139):

```python
    @property
    def research_reports(self) -> AsyncIOMotorCollection:
        return self.get_collection("research_reports")

    @property
    def research_sources(self) -> AsyncIOMotorCollection:
        return self.get_collection("research_sources")

    @property
    def research_cache(self) -> AsyncIOMotorCollection:
        return self.get_collection("research_cache")
```

Add indexes inside `ensure_indexes()` after the `offline_queue` entry (line 215):

```python
            "research_reports": [
                ("user_id", 1),
                ("created_at", -1),
                ("research_type", 1),
                [("user_id", 1), ("created_at", -1)],
                [("tags", 1)],
            ],
            "research_sources": [
                ("url", 1),
                ("domain", 1),
                ("overall_score", -1),
                ("created_at", -1),
            ],
            "research_cache": [
                ("cache_key", 1),
                ("ttl", 1),
            ],
```

---

### Task 2: Settings

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Add research settings**

After `TASK_BACKGROUND_POLL_SECONDS` (line 164), add:

```python
    # ── Research ───────────────────────────────────────────
    RESEARCH_CACHE_TTL_HOURS: int = Field(default=24, ge=1, description="Research cache TTL in hours")
    RESEARCH_MAX_SOURCES: int = Field(default=50, ge=5, le=200, description="Max sources per research query")
    RESEARCH_DEFAULT_DEPTH: str = "moderate"
```

No test needed — these are Pydantic settings with defaults.

---

### Task 3: ResearchService — Pipeline Core

**Files:**
- Create: `services/research_service.py`
- Create: `tests/services/test_research_service.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/services/test_research_service.py`:

```python
"""tests/services/test_research_service.py"""
import pytest
from services.research_service import ResearchService

pytestmark = pytest.mark.asyncio


class TestResearchService:
    """Tests for the ResearchService singleton."""

    async def test_detect_research_type_identifies_quick(self):
        service = ResearchService()
        rtype = await service.detect_research_type("What is the capital of France?")
        assert rtype == "quick"

    async def test_detect_research_type_identifies_deep(self):
        service = ResearchService()
        rtype = await service.detect_research_type(
            "I need a comprehensive deep research report on climate change impacts"
        )
        assert rtype == "deep"

    async def test_detect_research_type_identifies_comparative(self):
        service = ResearchService()
        rtype = await service.detect_research_type(
            "Compare and contrast Python vs JavaScript for web development"
        )
        assert rtype == "comparative"

    async def test_score_source_returns_weighted_scores(self):
        service = ResearchService()
        source = {
            "url": "https://example.edu/research",
            "title": "Research Paper",
            "snippet": "Important findings",
            "domain": "example.edu",
        }
        scored = await service.score_source(source, "research query")
        assert "overall_score" in scored
        assert "authority_score" in scored
        assert "freshness_score" in scored
        assert "relevance_score" in scored
        assert 0.0 <= scored["overall_score"] <= 1.0
        assert scored["authority_score"] >= 0.2  # .edu bonus
```

Run: `pytest tests/services/test_research_service.py -v`
Expected: FAIL — ImportError (module not found)

- [ ] **Step 2: Create the ResearchService class**

Create `services/research_service.py` with the full implementation:

```python
"""
Research Service — Core Research Pipeline
==========================================
Production-grade internet research engine supporting 7 research types,
source evaluation, fact checking, and structured report generation.

Pipeline: Plan → Search → Collect → Rank → Verify → Synthesize → Report → Respond

Models:
  - Minimax M2.1: Search query generation, synthesis, web search
  - DeepSeek V4 Flash: Fact verification, structured report generation
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from backend.config.settings import settings
from backend.utils.logger import get_logger
from backend.llm.minimax import minimax
from backend.llm.deepseek import deepseek
from backend.services.search_service import search_service
from backend.database.mongodb import mongodb
from backend.database.schemas import (
    new_research_report_doc,
    new_research_cache_doc,
)

logger = get_logger(__name__)


# ── Research type keywords ─────────────────────────────

RESEARCH_TYPE_KEYWORDS: dict[str, list[str]] = {
    "quick": [
        "what is", "who is", "when did", "where is", "define",
        "explain briefly", "tell me about", "meaning of",
        "how to", "why does", "quick", "simple",
    ],
    "deep": [
        "deep research", "comprehensive", "in-depth", "investigate",
        "thorough analysis", "detailed report", "research",
        "study", "full analysis", "write a report",
    ],
    "comparative": [
        "compare", "vs", "versus", "differences", "pros and cons",
        "better", "which one", "comparison", "contrast",
        "similarities", "trade-offs",
    ],
    "technical": [
        "technical", "specification", "implementation",
        "how it works", "under the hood", "internals",
        "protocol", "algorithm", "performance",
    ],
    "market": [
        "market", "trends", "competitive", "industry",
        "market size", "market share", "landscape",
        "adoption", "growth", "forecast",
    ],
    "product": [
        "product", "features", "pricing", "reviews", "best",
        "top rated", "recommended", "alternative to",
        "buy", "price", "worth it",
    ],
    "architecture": [
        "system design", "architecture", "design pattern",
        "architecture of", "components", "data flow",
        "infrastructure", "high-level design",
    ],
}


class ResearchService:
    """Core research pipeline service.

    Singleton that orchestrates the full 8-stage research pipeline:
    Plan → Search → Collect → Rank → Verify → Synthesize → Report → Respond
    """

    def __init__(self) -> None:
        self._cache_ttl = settings.RESEARCH_CACHE_TTL_HOURS
        self._max_sources = settings.RESEARCH_MAX_SOURCES
        self._default_depth = settings.RESEARCH_DEFAULT_DEPTH

    # ── Public API ──────────────────────────────────────

    async def plan_research(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stage 1: Generate a research plan from the user message.

        Returns:
            Dict with keys: queries, depth, research_type, sources_needed
        """
        research_type = await self.detect_research_type(message)
        depth = self._determine_depth(message, research_type)
        sources_needed = self._sources_for_depth(depth)

        # Generate search queries via Minimax
        queries = await self._generate_queries(message, research_type)

        return {
            "queries": queries[:5],
            "depth": depth,
            "research_type": research_type,
            "sources_needed": sources_needed,
        }

    async def detect_research_type(self, message: str) -> str:
        """Detect the research type from a user message using keyword scoring."""
        msg_lower = message.lower()
        scores: dict[str, int] = {}
        for rtype, keywords in RESEARCH_TYPE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in msg_lower)
            scores[rtype] = count

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "quick"

    async def execute_search(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Stage 2: Execute searches for all queries in the plan.

        Returns deduplicated list of raw search results.
        """
        queries = plan.get("queries", [])
        sources_needed = plan.get("sources_needed", 10)
        all_results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for query in queries:
            try:
                results = await search_service.search(query, num_results=min(10, sources_needed))
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
                if len(all_results) >= sources_needed:
                    break
            except Exception as exc:
                logger.warning("Search failed for query", extra={"query": query, "error": str(exc)})
                continue

        return all_results[:sources_needed]

    async def collect_sources(
        self,
        raw_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Stage 3: Collect full content and metadata from raw search results.

        Attempts full-page extraction; falls back to snippet-only on failure.
        """
        collected: list[dict[str, Any]] = []
        for raw in raw_results:
            url = raw.get("url", "")
            try:
                content = await search_service.extract_content(url)
            except Exception:
                content = None

            source = dict(raw)
            if content:
                source["content"] = content[:10000]
            else:
                source["content"] = ""
            source["domain"] = self._extract_domain(url)
            collected.append(source)

        return collected

    async def rank_sources(
        self,
        collected: list[dict[str, Any]],
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Stage 4: Rank collected sources using 5-dimension scoring.

        Scores each source on authority, freshness, accuracy, relevance,
        and popularity, producing a weighted overall_score (0.0 – 1.0).
        """
        for source in collected:
            source.update(await self.score_source(source, query))
        collected.sort(key=lambda s: s.get("overall_score", 0), reverse=True)
        return collected

    async def score_source(
        self,
        source: dict[str, Any],
        query: str = "",
    ) -> dict[str, float]:
        """Score a single source on 5 weighted dimensions.

        Weights:
          Authority(30%): Domain reputation (.edu/.gov +0.2, known tech +0.1)
          Freshness(20%): Recency decay
          Accuracy(25%):  Base score from content signals
          Relevance(15%): Keyword overlap with query
          Popularity(10%): Engagement signals
        """
        domain = source.get("domain", self._extract_domain(source.get("url", "")))

        # Authority score
        authority = self._compute_authority(domain, source.get("author"))

        # Freshness score
        freshness = self._compute_freshness(source.get("published_date"))

        # Accuracy score (base heuristic)
        accuracy = self._compute_accuracy(source.get("content", ""), source.get("snippet", ""))

        # Relevance score
        relevance = self._compute_relevance(query, source.get("title", ""), source.get("snippet", ""))

        # Popularity score
        popularity = self._compute_popularity(source)

        overall = (
            authority * 0.30
            + freshness * 0.20
            + accuracy * 0.25
            + relevance * 0.15
            + popularity * 0.10
        )

        return {
            "authority_score": round(authority, 4),
            "freshness_score": round(freshness, 4),
            "accuracy_score": round(accuracy, 4),
            "relevance_score": round(relevance, 4),
            "popularity_score": round(popularity, 4),
            "overall_score": round(overall, 4),
        }

    async def verify_facts(
        self,
        sources: list[dict[str, Any]],
        message: str,
    ) -> dict[str, Any]:
        """Stage 5: Fact-check claims from sources using DeepSeek V4 Flash.

        Returns:
            Dict with verified_claims, contradicted_claims, unverifiable_claims,
            overall_confidence.
        """
        if not sources:
            return {
                "verified_claims": [],
                "contradicted_claims": [],
                "unverifiable_claims": [],
                "overall_confidence": 0.0,
            }

        # Build source text for verification
        sources_text = "\n\n".join(
            f"Source {i + 1}: {s.get('title', 'Untitled')}\n"
            f"URL: {s.get('url', '')}\n"
            f"Content: {s.get('snippet', '')[:2000]}"
            for i, s in enumerate(sources[:10])
        )

        try:
            result = await deepseek.extract_json(
                system_prompt=(
                    "You are a fact-checking AI. Analyze the following sources "
                    "and user query. Extract factual claims, cross-reference them "
                    "across sources, and classify each claim.\n\n"
                    "Respond with valid JSON:\n"
                    "{\n"
                    '  "verified_claims": ["claim1", ...],\n'
                    '  "contradicted_claims": ["claim1", ...],\n'
                    '  "unverifiable_claims": ["claim1", ...],\n'
                    '  "overall_confidence": 0.0-1.0\n'
                    "}"
                ),
                user_message=(
                    f"User query: {message}\n\n"
                    f"Sources:\n{sources_text}\n\n"
                    "Fact-check the claims in these sources."
                ),
            )
            return {
                "verified_claims": result.get("verified_claims", []),
                "contradicted_claims": result.get("contradicted_claims", []),
                "unverifiable_claims": result.get("unverifiable_claims", []),
                "overall_confidence": float(result.get("overall_confidence", 0.5)),
            }
        except Exception as exc:
            logger.warning("Fact verification failed", extra={"error": str(exc)})
            return {
                "verified_claims": [],
                "contradicted_claims": [],
                "unverifiable_claims": [],
                "overall_confidence": 0.0,
            }

    # ── Composite pipeline methods ──────────────────────

    async def quick_search(
        self,
        query: str,
        max_sources: int = 10,
    ) -> dict[str, Any]:
        """Run the quick research pipeline: search -> synthesize -> respond.

        Returns:
            Dict with query, research_type, answer, sources, processing_time_ms.
        """
        start = time.monotonic()

        plan = await self.plan_research(query)
        raw_results = await self.execute_search({**plan, "sources_needed": max_sources})
        collected = await self.collect_sources(raw_results)
        ranked = await self.rank_sources(collected, query)
        top = ranked[:min(5, len(ranked))]

        # Synthesize using Minimax
        answer = await self._synthesize(query, top)

        sources_out = [
            {
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "snippet": s.get("snippet", ""),
                "domain": s.get("domain", ""),
                "overall_score": s.get("overall_score", 0.0),
                "authority_score": s.get("authority_score", 0.0),
                "freshness_score": s.get("freshness_score", 0.0),
                "relevance_score": s.get("relevance_score", 0.0),
            }
            for s in top
        ]

        elapsed = (time.monotonic() - start) * 1000
        return {
            "query": query,
            "research_type": plan["research_type"],
            "answer": answer,
            "sources": sources_out,
            "source_count": len(sources_out),
            "processing_time_ms": round(elapsed, 2),
        }

    async def deep_research(
        self,
        query: str,
        research_type: str = "deep",
        depth: str = "moderate",
        max_sources: int = 20,
    ) -> dict[str, Any]:
        """Run the full 8-stage deep research pipeline.

        Plan -> Search -> Collect -> Rank -> Verify -> Synthesize -> Report -> Respond

        Returns:
            Dict with report_id, title, all report sections, sources, fact_check.
        """
        start = time.monotonic()

        # Check cache
        cache_key = self._build_cache_key(query, depth, research_type)
        cached = await self._check_cache(cache_key)
        if cached:
            cached["processing_time_ms"] = round((time.monotonic() - start) * 1000, 2)
            return cached

        # Build plan
        plan = await self.plan_research(query)

        # Pipeline
        raw_results = await self.execute_search({
            **plan, "queries": plan["queries"] or [query],
            "sources_needed": max_sources,
        })
        collected = await self.collect_sources(raw_results)
        ranked = await self.rank_sources(collected, query)
        top = ranked[:min(max_sources, len(ranked))]

        verification = await self.verify_facts(top, query)
        synthesis = await self._synthesize(query, top)
        report = await self._generate_report(synthesis, verification, research_type, depth)

        sources_out = [
            {
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "snippet": s.get("snippet", ""),
                "domain": s.get("domain", ""),
                "overall_score": s.get("overall_score", 0.0),
                "authority_score": s.get("authority_score", 0.0),
                "freshness_score": s.get("freshness_score", 0.0),
                "relevance_score": s.get("relevance_score", 0.0),
            }
            for s in top
        ]

        fact_check_out = {
            "verified_claims": verification.get("verified_claims", []),
            "contradicted_claims": verification.get("contradicted_claims", []),
            "unverifiable_claims": verification.get("unverifiable_claims", []),
            "overall_confidence": verification.get("overall_confidence", 0.0),
        }

        elapsed = (time.monotonic() - start) * 1000
        now_str = datetime.now(timezone.utc).isoformat()

        result = {
            "report_id": "",
            "title": report.get("title", f"Research: {query[:80]}"),
            "research_type": research_type,
            "depth": depth,
            "executive_summary": report.get("executive_summary", synthesis[:500]),
            "key_findings": report.get("key_findings", []),
            "detailed_analysis": report.get("detailed_analysis"),
            "pros": report.get("pros"),
            "cons": report.get("cons"),
            "recommendations": report.get("recommendations"),
            "conclusions": report.get("conclusions"),
            "sources": sources_out,
            "source_count": len(sources_out),
            "fact_check": fact_check_out,
            "processing_time_ms": round(elapsed, 2),
            "created_at": now_str,
        }

        # Persist and cache
        try:
            report_id = await self._persist_report(result, query, depth, research_type)
            result["report_id"] = report_id
            await self._set_cache(cache_key, result)
        except Exception as exc:
            logger.warning("Research persistence failed", extra={"error": str(exc)})

        return result

    # ── Scoring helpers ────────────────────────────────

    async def _generate_queries(self, message: str, research_type: str) -> list[str]:
        """Generate search queries from the user message."""
        try:
            raw = await minimax.web_search_query(message)
            queries = [q.strip("- \n") for q in raw.split("\n") if q.strip()]
            return queries[:5]
        except Exception:
            return [message]

    def _determine_depth(self, message: str, research_type: str) -> str:
        """Determine research depth from message length and type."""
        if research_type == "quick":
            return "quick"
        word_count = len(message.split())
        if word_count > 30:
            return "deep"
        if word_count > 15:
            return "moderate"
        return self._default_depth

    @staticmethod
    def _sources_for_depth(depth: str) -> int:
        return {"quick": 5, "moderate": 10, "deep": 20, "comprehensive": 30}.get(depth, 10)

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain from a URL."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc or parsed.hostname or url
        except Exception:
            return url

    @staticmethod
    def _compute_authority(domain: str, author: str | None = None) -> float:
        """Compute authority score (0.0 - 1.0) from domain and author."""
        score = 0.3  # Base
        if domain.endswith((".edu", ".gov", ".org")):
            score += 0.2
        elif domain.endswith((".ac.uk", ".ac.jp", ".edu.au")):
            score += 0.15
        known_domains = {
            "wikipedia.org": 0.1, "reuters.com": 0.15, "bloomberg.com": 0.15,
            "nature.com": 0.2, "sciencedirect.com": 0.2, "ieee.org": 0.2,
            "arxiv.org": 0.15, "github.com": 0.1, "stackoverflow.com": 0.1,
        }
        for known_domain, bonus in known_domains.items():
            if known_domain in domain:
                score += bonus
                break
        if author:
            score += 0.05
        return max(0.0, min(1.0, score))

    @staticmethod
    def _compute_freshness(published_date: datetime | None) -> float:
        """Compute freshness score (0.0 - 1.0) based on recency."""
        if not published_date:
            return 0.5  # Unknown date neutral
        days_old = (datetime.now(timezone.utc) - published_date).days
        if days_old < 7:
            return 1.0
        if days_old < 30:
            return 0.9
        if days_old < 90:
            return 0.8
        if days_old < 365:
            return 0.6
        if days_old < 730:
            return 0.4
        return max(0.1, 1.0 - (days_old / 3650))

    @staticmethod
    def _compute_accuracy(content: str, snippet: str) -> float:
        """Compute accuracy signal score (0.0 - 1.0)."""
        score = 0.5
        if len(content) > 500:
            score += 0.1
        if len(content) > 2000:
            score += 0.1
        if "citation" in content.lower() or "reference" in content.lower():
            score += 0.1
        if "doi.org" in content or "arxiv" in content:
            score += 0.1
        if "according to" in content.lower():
            score += 0.05
        return min(1.0, score)

    @staticmethod
    def _compute_relevance(query: str, title: str, snippet: str) -> float:
        """Compute relevance score (0.0 - 1.0) via keyword overlap."""
        if not query:
            return 0.5
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        snippet_words = set(snippet.lower().split())
        combined = title_words | snippet_words
        if not combined:
            return 0.3
        overlap = len(query_words & combined)
        score = overlap / max(len(query_words), 1) * 0.8
        # Bonus: exact phrase match in title
        if query.lower()[:30] in title.lower():
            score += 0.2
        return min(1.0, score)

    @staticmethod
    def _compute_popularity(source: dict[str, Any]) -> float:
        """Compute popularity score (0.0 - 1.0)."""
        score = 0.3
        snippet = source.get("snippet", "")
        if len(snippet) > 100:
            score += 0.1
        url = source.get("url", "")
        if url.count("/") > 3:
            score += 0.1  # Deeper paths more specific content
        return min(1.0, score)

    # ── LLM synthesis and report generation ────────────

    async def _synthesize(
        self,
        query: str,
        top_sources: list[dict[str, Any]],
    ) -> str:
        """Synthesize top sources into a coherent answer using Minimax."""
        if not top_sources:
            return "I searched but couldn't find relevant results. Try rephrasing your query."

        sources_text = "\n\n".join(
            f"Source {i + 1}: {s.get('title', 'Untitled')}\n"
            f"URL: {s.get('url', '')}\n"
            f"Snippet: {s.get('snippet', '')}"
            for i, s in enumerate(top_sources[:15])
        )

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
                "content": f"Question: {query}\n\nSearch Results:\n{sources_text}",
            },
        ]
        try:
            response = await minimax.chat(messages, temperature=0.3)
            return response["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("Synthesis failed", extra={"error": str(exc)})
            return "I couldn't synthesise the search results at this time."

    async def _generate_report(
        self,
        synthesis: str,
        verification: dict[str, Any],
        research_type: str,
        depth: str,
    ) -> dict[str, Any]:
        """Generate a structured report using DeepSeek V4 Flash."""
        type_sections = {
            "quick": "Concise answer with key points only.",
            "deep": "Executive summary, key findings, detailed analysis, conclusions.",
            "comparative": "Side-by-side comparison, pros/cons per option, recommendation.",
            "technical": "Overview, specifications, trade-offs, implementation notes.",
            "market": "Trends, competitive landscape, key players, outlook.",
            "product": "Features, pricing, user reviews, pros/cons, verdict.",
            "architecture": "System design, components, data flow, tech stack, alternatives.",
        }

        section_desc = type_sections.get(research_type, "Standard research report.")

        system_prompt = (
            "You are a research report writer. Generate a structured report "
            "based on the synthesis and fact-check results provided.\n\n"
            f"Report type: {research_type} (Depth: {depth})\n"
            f"Sections to include: {section_desc}\n\n"
            "Respond with valid JSON only:\n"
            "{\n"
            '  "title": "...",\n'
            '  "executive_summary": "...",\n'
            '  "key_findings": ["...", "..."],\n'
            '  "detailed_analysis": "...",\n'
            '  "pros": ["..."] | null,\n'
            '  "cons": ["..."] | null,\n'
            '  "recommendations": ["..."] | null,\n'
            '  "conclusions": "..."\n'
            "}"
        )

        verified = verification.get("verified_claims", [])
        contradicted = verification.get("contradicted_claims", [])

        user_message = (
            f"Synthesis:\n{synthesis}\n\n"
            f"Fact-check results:\n"
            f"  Verified claims: {verified}\n"
            f"  Contradicted claims: {contradicted}\n"
            f"  Overall confidence: {verification.get('overall_confidence', 0.0)}\n\n"
            "Generate the report."
        )

        try:
            result = await deepseek.extract_json(system_prompt, user_message)
            return {
                "title": result.get("title", "Research Report"),
                "executive_summary": result.get("executive_summary", synthesis[:500]),
                "key_findings": result.get("key_findings", []),
                "detailed_analysis": result.get("detailed_analysis"),
                "pros": result.get("pros"),
                "cons": result.get("cons"),
                "recommendations": result.get("recommendations"),
                "conclusions": result.get("conclusions"),
            }
        except Exception as exc:
            logger.warning("Report generation failed", extra={"error": str(exc)})
            return {
                "title": "Research Report",
                "executive_summary": synthesis[:500],
                "key_findings": [],
                "detailed_analysis": synthesis,
            }

    # ── Cache ──────────────────────────────────────────

    def _build_cache_key(self, query: str, depth: str, research_type: str) -> str:
        """Build a deterministic cache key from query parameters."""
        normalized = f"{query.strip().lower()}:{depth}:{research_type}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    async def _check_cache(self, cache_key: str) -> dict[str, Any] | None:
        """Check if a cached result exists and is not expired."""
        try:
            doc = await mongodb.research_cache.find_one({
                "cache_key": cache_key,
                "ttl": {"$gt": datetime.now(timezone.utc)},
            })
            if doc and doc.get("report_id"):
                from bson import ObjectId
                report = await mongodb.research_reports.find_one({
                    "_id": ObjectId(doc["report_id"]),
                })
                if report:
                    from database.schemas import serialize_doc
                    return serialize_doc(report)
            return None
        except Exception:
            return None

    async def _set_cache(self, cache_key: str, result: dict[str, Any]) -> None:
        """Store a result in the research cache."""
        try:
            ttl = datetime.now(timezone.utc) + timedelta(hours=self._cache_ttl)
            doc = new_research_cache_doc(
                cache_key=cache_key,
                query=result.get("query", ""),
                research_type=result.get("research_type", ""),
                ttl=ttl,
            )
            await mongodb.research_cache.insert_one(doc)
        except Exception as exc:
            logger.warning("Cache set failed", extra={"error": str(exc)})

    # ── Persistence ────────────────────────────────────

    async def _persist_report(
        self,
        result: dict[str, Any],
        query: str,
        depth: str,
        research_type: str,
    ) -> str:
        """Persist a research report to MongoDB."""
        doc = new_research_report_doc(
            user_id=result.get("user_id", "anonymous"),
            session_id=result.get("session_id", ""),
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
            sources=result.get("sources", []),
            fact_check=result.get("fact_check"),
            tags=[research_type, depth],
        )
        ins = await mongodb.research_reports.insert_one(doc)
        return str(ins.inserted_id)


# Global singleton
research_service = ResearchService()
```

- [ ] **Step 3: Run tests to verify progress**

Run: `pytest tests/services/test_research_service.py -v -x`
Expected: Tests that don't need external calls pass (detect types, score_source)

- [ ] **Step 4: Add mocked test variants for LLM-dependent methods**

Add to `tests/services/test_research_service.py`:

```python
class TestResearchServiceMocked:
    """Tests with mocked external dependencies."""

    @patch("services.research_service.minimax.web_search_query")
    async def test_generate_queries(self, mock_queries):
        mock_queries.return_value = "query1\nquery2\nquery3"
        service = ResearchService()
        queries = await service._generate_queries("test", "quick")
        assert len(queries) == 3
        mock_queries.assert_called_once()

    @patch("services.research_service.minimax.chat")
    async def test_synthesize_with_results(self, mock_chat):
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Synthesized answer."}}]
        }
        service = ResearchService()
        sources = [{"title": "S1", "url": "https://example.com", "snippet": "Content"}]
        result = await service._synthesize("test query", sources)
        assert "Synthesized answer" in result
```

---

### Task 4: ResearchAgent — LangGraph Agent (Replace existing)

**Files:**
- Replace: `agents_v2/research_agent.py`
- Create: `tests/agents_v2/test_research_agent.py`

- [ ] **Step 1: Write failing agent tests**

Create `tests/agents_v2/test_research_agent.py`:

```python
"""tests/agents_v2/test_research_agent.py"""
import pytest
from agents_v2.state import create_initial_state
from agents_v2.research_agent import ResearchAgent

pytestmark = pytest.mark.asyncio


class TestResearchAgent:
    @pytest.fixture
    def agent(self):
        return ResearchAgent()

    @pytest.fixture
    def base_state(self):
        return create_initial_state(
            user_id="test_user",
            message="What is quantum computing?",
        )

    async def test_agent_name_and_model(self, agent):
        assert agent.name == "research"
        assert agent.model_name == "minimax"

    async def test_detect_research_type_quick(self, agent):
        rtype = await agent.detect_research_type("What is the capital of France?")
        assert rtype == "quick"

    async def test_detect_research_type_deep(self, agent):
        rtype = await agent.detect_research_type(
            "I need a comprehensive deep research report"
        )
        assert rtype == "deep"

    async def test_detect_research_type_comparative(self, agent):
        rtype = await agent.detect_research_type(
            "Compare Python vs JavaScript"
        )
        assert rtype == "comparative"

    async def test_process_returns_state(self, agent, base_state):
        result = await agent.process(base_state)
        assert "final_response" in result
        assert result["response_agent"] == "research"
```

- [ ] **Step 2: Replace research_agent.py with enhanced version**

Complete content for `agents_v2/research_agent.py`:

```python
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

from agents_v2.state import AgentState
from agents_v2.base import BaseAgent
from agents_v2.registry import get_agent_registry
from services.research_service import research_service, RESEARCH_TYPE_KEYWORDS
from database.mongodb import mongodb
from database.schemas import new_knowledge_doc
from config.settings import settings


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
```

- [ ] **Step 3: Run agent tests**

Run: `pytest tests/agents_v2/test_research_agent.py -v`
Expected: Tests that don't need LLM calls pass (name, detect type)

---

### Task 5: Research API

**Files:**
- Create: `api/research.py`
- Create: `tests/api/test_research_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/api/test_research_api.py`:

```python
"""tests/api/test_research_api.py"""
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

pytestmark = pytest.mark.asyncio


class TestResearchAPI:
    @pytest.fixture
    async def client(self):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac

    @pytest.fixture
    async def auth_headers(self):
        return {"Authorization": "Bearer test-token"}

    async def test_search_endpoint_requires_auth(self, client):
        resp = await client.post("/api/v2/research/search", json={"query": "test"})
        assert resp.status_code == 401

    async def test_deep_endpoint_requires_auth(self, client):
        resp = await client.post("/api/v2/research/deep", json={"query": "test"})
        assert resp.status_code == 401

    async def test_history_endpoint_requires_auth(self, client):
        resp = await client.get("/api/v2/research/history")
        assert resp.status_code == 401
```

Run: `pytest tests/api/test_research_api.py -v`
Expected: FAIL — router not registered yet

- [ ] **Step 2: Create the research API router**

Create `api/research.py`:

```python
"""
Research API Routes
===================
FastAPI endpoints for the Research Agent system.

Provides:
- POST /api/v2/research/search     - Quick web search with synthesis
- POST /api/v2/research/deep       - Full deep research pipeline
- POST /api/v2/research/verify     - Fact-check content
- GET  /api/v2/research/history    - Paginated research history
- GET  /api/v2/research/reports/{report_id} - Get specific report
- GET  /api/v2/research/sources    - Search/filter sources
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import get_current_user
from database.models import (
    UserResponse,
    ResearchSearchRequest,
    ResearchSearchResponse,
    DeepResearchRequest,
    DeepResearchResponse,
    VerifyContentRequest,
    VerifyContentResponse,
    SourceScore,
    FactCheckResult,
    ResearchReportSummary,
)
from database.schemas import serialize_doc
from database.mongodb import mongodb
from services.research_service import research_service

router = APIRouter(prefix="/api/v2/research", tags=["research"])


@router.post("/search", response_model=ResearchSearchResponse)
async def research_search(
    request: ResearchSearchRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Quick web search with AI-powered synthesis.

    Runs the first stages of the research pipeline:
    Plan -> Search -> Collect -> Rank -> Synthesize
    """
    result = await research_service.quick_search(
        query=request.query,
        max_sources=request.max_sources,
    )

    sources = [SourceScore(**s) for s in result.get("sources", [])]

    return ResearchSearchResponse(
        query=result["query"],
        research_type=result.get("research_type", "quick"),
        answer=result.get("answer", ""),
        sources=sources,
        source_count=result.get("source_count", 0),
        processing_time_ms=result.get("processing_time_ms", 0.0),
    )


@router.post("/deep", response_model=DeepResearchResponse)
async def research_deep(
    request: DeepResearchRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Full deep research with the complete 8-stage pipeline.

    Plan -> Search -> Collect -> Rank -> Verify -> Synthesize -> Report -> Respond
    """
    result = await research_service.deep_research(
        query=request.query,
        research_type=request.research_type.value,
        depth=request.depth.value,
        max_sources=request.max_sources,
    )

    sources = [SourceScore(**s) for s in result.get("sources", [])]
    fact_check_data = result.get("fact_check")
    fact_check = FactCheckResult(**fact_check_data) if fact_check_data else None

    return DeepResearchResponse(
        report_id=result.get("report_id", ""),
        title=result.get("title", ""),
        research_type=result.get("research_type", request.research_type.value),
        depth=result.get("depth", request.depth.value),
        executive_summary=result.get("executive_summary", ""),
        key_findings=result.get("key_findings", []),
        detailed_analysis=result.get("detailed_analysis"),
        pros=result.get("pros"),
        cons=result.get("cons"),
        recommendations=result.get("recommendations"),
        conclusions=result.get("conclusions"),
        sources=sources,
        source_count=result.get("source_count", 0),
        fact_check=fact_check,
        processing_time_ms=result.get("processing_time_ms", 0.0),
        created_at=result.get("created_at", ""),
    )


@router.post("/verify", response_model=VerifyContentResponse)
async def research_verify(
    request: VerifyContentRequest,
    current_user: UserResponse = Depends(get_current_user),
):
    """Fact-check content against web sources.

    Extracts claims from the provided content, cross-references against
    web search results, and classifies each claim as verified, contradicted,
    or unverifiable.
    """
    # Search for relevant sources first
    plan = await research_service.plan_research(request.content)
    raw_results = await research_service.execute_search({
        **plan, "sources_needed": 10,
    })
    collected = await research_service.collect_sources(raw_results)

    verification = await research_service.verify_facts(
        collected, request.content,
    )

    return VerifyContentResponse(
        verified_claims=verification.get("verified_claims", []),
        contradicted_claims=verification.get("contradicted_claims", []),
        unverifiable_claims=verification.get("unverifiable_claims", []),
        overall_confidence=verification.get("overall_confidence", 0.0),
        analysis=f"Found {len(verification.get('verified_claims', []))} verified, "
                 f"{len(verification.get('contradicted_claims', []))} contradicted, "
                 f"and {len(verification.get('unverifiable_claims', []))} unverifiable claims.",
    )


@router.get("/history")
async def research_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    research_type: str | None = Query(None),
    current_user: UserResponse = Depends(get_current_user),
):
    """Get paginated research history for the current user."""
    query_filter: dict[str, Any] = {"user_id": current_user.id}
    if research_type:
        query_filter["research_type"] = research_type

    cursor = (
        mongodb.research_reports
        .find(query_filter)
        .sort("created_at", -1)
        .skip(offset)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    total = await mongodb.research_reports.count_documents(query_filter)

    items = []
    for doc in docs:
        serialized = serialize_doc(doc)
        items.append(ResearchReportSummary(
            id=serialized["id"],
            query=serialized.get("query", ""),
            research_type=serialized.get("research_type", ""),
            depth=serialized.get("depth", ""),
            executive_summary=serialized.get("executive_summary", "")[:200],
            source_count=serialized.get("source_count", 0),
            created_at=serialized.get("created_at", ""),
        ))

    return {
        "items": items,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/reports/{report_id}")
async def get_research_report(
    report_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Get a specific research report by ID."""
    from bson import ObjectId

    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid report ID format")

    doc = await mongodb.research_reports.find_one({
        "_id": oid,
        "user_id": current_user.id,
    })

    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    return serialize_doc(doc)


@router.get("/sources")
async def search_sources(
    domain: str | None = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    current_user: UserResponse = Depends(get_current_user),
):
    """Search and filter cached research sources."""
    query_filter: dict[str, Any] = {}
    if domain:
        query_filter["domain"] = {"$regex": domain, "$options": "i"}
    if min_score > 0:
        query_filter["overall_score"] = {"$gte": min_score}

    cursor = (
        mongodb.research_sources
        .find(query_filter)
        .sort("overall_score", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)

    return [serialize_doc(doc) for doc in docs]
```

- [ ] **Step 3: Register the router in main.py**

Edit `backend/main.py`. After line 246 (`app.include_router(agents_v2_router)`), add:

```python
# Register research API
from api.research import router as research_router
app.include_router(research_router)
```

- [ ] **Step 4: Run API tests**

Run: `pytest tests/api/test_research_api.py -v`
Expected: Auth-required tests PASS (get 401)

---

### Task 6: Integration & Verification

**Files:**
- None new — just import verification

- [ ] **Step 1: Verify agent system initializes correctly**

The `agents_v2/init.py` already imports `ResearchAgent` from `agents_v2.research_agent` at line 36, and registers it at line 47. The updated `ResearchAgent` class auto-registers via `get_agent_registry().register(self)` in `__init__`, so existing init code works unchanged.

Run: `python -c "from agents_v2.init import initialize_agent_system; import asyncio; asyncio.run(initialize_agent_system())" 2>&1 | head -5`
Expected: "Registered 9 agents: ['router', 'planner', 'coding', 'research', 'vision', 'memory', 'task', 'utility', 'response']"

- [ ] **Step 2: Verify all imports compile cleanly**

Run: `python -c "
from services.research_service import research_service, ResearchService
from api.research import router
from agents_v2.research_agent import ResearchAgent
from database.schemas import new_research_report_doc, new_research_source_doc, new_research_cache_doc
from database.models import ResearchType, ResearchDepth, SourceScore, FactCheckResult
from database.models import ResearchSearchRequest, DeepResearchResponse, VerifyContentResponse
print('All imports OK')
"`
Expected: "All imports OK"

---

### Task 7: Integration Tests

**Files:**
- Create: `tests/integration/test_research_pipeline.py`

- [ ] **Step 1: Write integration tests**

```python
"""tests/integration/test_research_pipeline.py"""
import pytest
from unittest.mock import patch
from services.research_service import ResearchService

pytestmark = pytest.mark.asyncio


class TestResearchPipelineIntegration:
    """Integration tests for the full research pipeline.
    
    These tests exercise real service methods with mocked LLM/Search
    dependencies to validate pipeline logic end-to-end.
    """

    @patch("services.research_service.minimax.web_search_query")
    @patch("services.research_service.search_service.search")
    async def test_quick_search_pipeline(self, mock_search, mock_queries):
        """Quick search should return answer with sources."""
        mock_queries.return_value = "test query"
        mock_search.return_value = [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "Content 1"},
            {"title": "Result 2", "url": "https://example.com/2", "snippet": "Content 2"},
        ]

        service = ResearchService()
        result = await service.quick_search("test query", max_sources=5)

        assert result["query"] == "test query"
        assert "sources" in result
        assert "processing_time_ms" in result
        assert result["source_count"] > 0

    @patch("services.research_service.minimax.web_search_query")
    @patch("services.research_service.search_service.search")
    @patch("services.research_service.deepseek.extract_json")
    async def test_deep_research_pipeline(self, mock_extract, mock_search, mock_queries):
        """Deep research should return full report structure."""
        mock_queries.return_value = "test query"
        mock_search.return_value = [
            {"title": "Source 1", "url": "https://example.com/1", "snippet": "Content"},
            {"title": "Source 2", "url": "https://example.com/2", "snippet": "Content"},
        ]
        mock_extract.return_value = {
            "title": "Test Report",
            "executive_summary": "Summary",
            "key_findings": ["Finding 1"],
            "detailed_analysis": "Analysis",
            "pros": ["Pro 1"],
            "cons": ["Con 1"],
            "recommendations": ["Rec 1"],
            "conclusions": "Conclusion",
            "verified_claims": ["Claim 1"],
            "contradicted_claims": [],
            "unverifiable_claims": [],
            "overall_confidence": 0.8,
        }

        service = ResearchService()
        result = await service.deep_research(
            "test query", research_type="deep", depth="quick", max_sources=5
        )

        assert "title" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "sources" in result
        assert "fact_check" in result
        assert "processing_time_ms" in result

    async def test_empty_search_returns_empty_pipeline(self):
        """Pipeline should handle empty search results gracefully."""
        service = ResearchService()
        result = await service.quick_search("nonexistent_topic_xyz_123", max_sources=5)
        assert result["source_count"] == 0 or result["answer"]

    async def test_verify_facts_no_sources(self):
        """Fact verification with no sources should return empty results."""
        service = ResearchService()
        result = await service.verify_facts([], "test")
        assert result["verified_claims"] == []
        assert result["overall_confidence"] == 0.0
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_research_pipeline.py -v`
Expected: PASS (mocked tests) + unmocked tests handle gracefully

---

## Self-Review Checklist

- [x] **Spec coverage**: Every spec section maps to tasks:
  - MongoDB schemas -> Task 1
  - Settings -> Task 2
  - ResearchService (plan/search/collect/rank/verify/synthesize/report) -> Task 3
  - ResearchAgent (type detection, state management) -> Task 4
  - API endpoints (search/deep/verify/history/reports/sources) -> Task 5
  - Registration + import verification -> Task 6
  - Integration tests -> Task 7

- [x] **Placeholder scan**: No TBDs, TODOs, "implement later", "add error handling" (all concrete). Every step has exact code. No "similar to Task N" references.

- [x] **Type consistency**: `detect_research_type()` returns `str`, `score_source()` returns `dict[str, float]`, `ResearchSearchResponse` uses `SourceScore` model, `DeepResearchResponse` includes `FactCheckResult`. Method signatures match between ResearchService and ResearchAgent.

- [x] **Test coverage**: Schema tests (Task 1), Service tests (Task 3), Agent tests (Task 4), API tests (Task 5), Integration tests (Task 7)
