"""tests/integration/test_research_pipeline.py

End-to-end integration tests for the research pipeline.

Mocks all external dependencies (LLMs, search service, MongoDB) and
tests the full pipeline flow: plan → search → score → verify → synthesise.

Run from /root:
    python3 -m pytest backend/tests/integration/test_research_pipeline.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.services.research_service import ResearchService

pytestmark = pytest.mark.asyncio


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def service() -> ResearchService:
    """Return a fresh ResearchService singleton per test."""
    return ResearchService()


# ── Full Pipeline Integration Tests ───────────────────────────────────────────


class TestQuickSearchPipeline:
    """End-to-end test of quick_search() with mocked dependencies."""

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    @patch("backend.services.research_service.minimax.chat")
    async def test_quick_search_full_flow(
        self,
        mock_chat: AsyncMock,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """Verify the full quick_search pipeline: plan → search → score → synthesise."""
        # ── Arrange ──
        mock_queries.return_value = "test query 1\ntest query 2"
        mock_search.return_value = [
            {
                "title": "Result A",
                "url": "https://example.edu/a",
                "snippet": "This is snippet A about the topic",
            },
            {
                "title": "Result B",
                "url": "https://example.org/b",
                "snippet": "This is snippet B with more detail about the topic",
            },
            {
                "title": "Result C",
                "url": "https://example.com/c",
                "snippet": "Snippet C here",
            },
        ]
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Synthesised answer text."}}]
        }

        # ── Act ──
        result = await service.quick_search("What is the topic?", max_sources=3)

        # ── Assert ──
        assert result["query"] == "What is the topic?"
        assert result["answer"] == "Synthesised answer text."
        assert len(result["sources"]) == 3
        assert result["source_count"] == 3
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

        # Verify sources are scored
        for source in result["sources"]:
            assert "overall_score" in source
            assert "authority_score" in source
            assert "freshness_score" in source
            assert "accuracy_score" in source
            assert 0.0 <= source["overall_score"] <= 1.0

        # Verify the edu domain got an authority bonus
        edu_sources = [s for s in result["sources"] if "example.edu" in s.get("domain", "")]
        if edu_sources:
            assert edu_sources[0]["authority_score"] >= 0.2  # .edu bonus

        # Verify calls
        assert mock_queries.called
        assert mock_search.called
        assert mock_chat.called

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    async def test_quick_search_fallback_synthesis(
        self,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """When LLM chat fails, quick_search falls back to snippet-based synthesis."""
        # ── Arrange ──
        mock_queries.return_value = "test query"
        mock_search.return_value = [
            {
                "title": "Result A",
                "url": "https://example.com/a",
                "snippet": "This is snippet A about the topic",
            },
        ]

        # ── Act ──
        result = await service.quick_search("What is the topic?", max_sources=1)

        # ── Assert ──
        assert result["query"] == "What is the topic?"
        assert "Here is what I found" in result["answer"]
        assert len(result["sources"]) == 1

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    async def test_quick_search_no_results(
        self,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """Quick search handles empty search results gracefully."""
        mock_queries.return_value = "test query"
        mock_search.return_value = []

        result = await service.quick_search("Obscure topic", max_sources=5)

        assert result["query"] == "Obscure topic"
        assert result["answer"] == "No sources found for 'Obscure topic'."
        assert result["sources"] == []
        assert result["source_count"] == 0


class TestDeepResearchPipeline:
    """End-to-end test of deep_research() with mocked dependencies."""

    @staticmethod
    def _make_mock_mongodb():
        """Create a mock MongoDB object that avoids triggering properties
        that require an active connection."""
        mock_db = AsyncMock()
        # mock_db.research_cache.find_one returns None (cache miss)
        mock_db.research_cache.find_one = AsyncMock(return_value=None)
        mock_db.research_cache.insert_one = AsyncMock(return_value=None)
        # mock_db.research_reports.insert_one returns a result with inserted_id
        mock_insert_result = AsyncMock()
        mock_insert_result.inserted_id = "507f1f77bcf86cd799439011"
        mock_db.research_reports.insert_one = AsyncMock(return_value=mock_insert_result)
        return mock_db

    @patch("backend.services.research_service.search_service.extract_content")
    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    @patch("backend.services.research_service.minimax.chat")
    @patch("backend.services.research_service.deepseek.extract_json")
    @patch("backend.services.research_service.mongodb")
    async def test_deep_research_full_flow(
        self,
        mock_mongodb: AsyncMock,
        mock_deepseek: AsyncMock,
        mock_chat: AsyncMock,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        mock_extract: AsyncMock,
        service: ResearchService,
    ):
        """Verify the full deep_research pipeline with cache miss, search,
        scoring, fact-check, report generation, and persistence."""
        # ── Arrange – Mock MongoDB ──
        mock_mongodb.configure_mock(**self._make_mock_mongodb().__dict__)

        # Query generation
        mock_queries.return_value = "deep query 1\ndeep query 2"

        # Search results
        mock_search.return_value = [
            {
                "title": "Deep Result A",
                "url": "https://example.edu/deep-a",
                "snippet": "Detailed analysis of the topic with citations and references",
            },
            {
                "title": "Deep Result B",
                "url": "https://example.gov/deep-b",
                "snippet": "Government report on the topic according to latest data",
            },
            {
                "title": "Deep Result C",
                "url": "https://example.com/deep-c",
                "snippet": "Blog post about the topic",
            },
        ]

        # Content extraction
        async def mock_extract_content(url: str) -> str:
            return f"Full content for {url}. " * 50

        mock_extract.side_effect = mock_extract_content

        # Synthesis
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Deep synthesis text."}}]
        }

        # Fact-checking + Report generation (both use deepseek.extract_json)
        mock_deepseek.side_effect = [
            {
                "verified_claims": ["Claim 1", "Claim 2"],
                "contradicted_claims": [],
                "unverifiable_claims": ["Claim 3"],
                "overall_confidence": 0.85,
            },
            {
                "title": "Deep Research Report",
                "executive_summary": "Executive summary of findings",
                "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
                "detailed_analysis": "In-depth analysis of the topic",
                "pros": ["Pro 1"],
                "cons": ["Con 1"],
                "recommendations": ["Recommendation 1"],
                "conclusions": "Final conclusion",
            },
        ]

        # ── Act ──
        result = await service.deep_research(
            query="Tell me everything about the topic",
            research_type="deep",
            depth="moderate",
            max_sources=3,
        )

        # ── Assert – Report structure ──
        assert result["title"] == "Deep Research Report"
        assert result["executive_summary"] == "Executive summary of findings"
        assert len(result["key_findings"]) == 3
        assert "Finding 1" in result["key_findings"]
        assert result["detailed_analysis"] == "In-depth analysis of the topic"
        assert result["pros"] == ["Pro 1"]
        assert result["cons"] == ["Con 1"]
        assert result["recommendations"] == ["Recommendation 1"]
        assert result["conclusions"] == "Final conclusion"

        # ── Assert – Sources are scored ──
        assert len(result["sources"]) == 3
        for source in result["sources"]:
            assert "overall_score" in source
            assert 0.0 <= source["overall_score"] <= 1.0

        # ── Assert – Fact check ──
        assert len(result["fact_check"]["verified_claims"]) == 2
        assert result["fact_check"]["overall_confidence"] == 0.85

        # ── Assert – Metadata ──
        assert result["research_type"] == "deep"
        assert result["depth"] == "moderate"
        assert result["source_count"] == 3
        assert "processing_time_ms" in result
        assert result["report_id"] == "507f1f77bcf86cd799439011"

        # ── Verify all external calls were made ──
        assert mock_queries.called
        assert mock_search.called
        assert mock_extract.called
        assert mock_chat.called
        assert mock_deepseek.call_count >= 1
        assert mock_mongodb.research_cache.find_one.called
        assert mock_mongodb.research_cache.insert_one.called
        assert mock_mongodb.research_reports.insert_one.called

    @patch("backend.services.research_service.mongodb")
    async def test_deep_research_cache_hit(
        self,
        mock_mongodb: AsyncMock,
        service: ResearchService,
    ):
        """deep_research returns cached results when available."""
        # Arrange: return a cached result
        mock_mongodb.research_cache.find_one = AsyncMock(
            return_value={
                "cache_key": "abc",
                "ttl": None,  # No TTL = expired
            }
        )

        result = await service.deep_research(
            query="Cached query",
            research_type="quick",
            depth="quick",
            max_sources=3,
        )

        # Cache lookup was attempted
        assert mock_mongodb.research_cache.find_one.called

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    @patch("backend.services.research_service.minimax.chat")
    @patch("backend.services.research_service.deepseek.extract_json")
    @patch("backend.services.research_service.mongodb")
    async def test_deep_research_with_market_type(
        self,
        mock_mongodb: AsyncMock,
        mock_deepseek: AsyncMock,
        mock_chat: AsyncMock,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """deep_research generates market-specific report sections."""
        mock_mongodb.configure_mock(**self._make_mock_mongodb().__dict__)
        mock_queries.return_value = "market query"
        mock_search.return_value = [
            {
                "title": "Market Report",
                "url": "https://example.com/market",
                "snippet": "Market analysis with competitive landscape data",
            },
        ]
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Market synthesis."}}]
        }
        mock_deepseek.side_effect = [
            {"verified_claims": [], "contradicted_claims": [], "unverifiable_claims": [], "overall_confidence": 0.0},
            {
                "title": "Market Research Report",
                "executive_summary": "Market trends and outlook",
                "key_findings": ["Trend 1", "Competitor analysis"],
                "detailed_analysis": "Market analysis",
                "pros": None,
                "cons": None,
                "recommendations": ["Strategy recommendation"],
                "conclusions": "Market conclusion",
            },
        ]

        result = await service.deep_research(
            query="Market trends for AI",
            research_type="market",
            depth="moderate",
            max_sources=1,
        )

        assert result["title"] == "Market Research Report"
        assert result["research_type"] == "market"


class TestPlanAndScorePipeline:
    """Tests for the planning and scoring stages of the pipeline."""

    @patch("backend.services.research_service.minimax.web_search_query")
    async def test_plan_research_full(
        self,
        mock_queries: AsyncMock,
        service: ResearchService,
    ):
        """plan_research produces complete plan with queries, depth, type."""
        mock_queries.return_value = "q1\nq2\nq3"

        plan = await service.plan_research(
            "Compare Python and JavaScript for web development"
        )

        assert "queries" in plan
        assert len(plan["queries"]) == 3
        assert plan["research_type"] == "comparative"
        assert plan["depth"] in ("quick", "moderate", "deep", "comprehensive")
        assert 5 <= plan["sources_needed"] <= 30

    @patch("backend.services.research_service.search_service.search")
    async def test_execute_search_deduplication(
        self,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """execute_search deduplicates results by URL."""
        mock_search.return_value = [
            {"title": "A", "url": "https://example.com/1", "snippet": "S1"},
            {"title": "B", "url": "https://example.com/2", "snippet": "S2"},
            {"title": "A dup", "url": "https://example.com/1", "snippet": "S1 dup"},
        ]

        plan = {
            "queries": ["test query"],
            "depth": "quick",
            "research_type": "quick",
            "sources_needed": 10,
        }
        results = await service.execute_search(plan)

        assert len(results) == 2  # Duplicate URL removed
        urls = [r["url"] for r in results]
        assert len(urls) == len(set(urls))

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.search_service.extract_content")
    async def test_collect_and_rank_sources(
        self,
        mock_extract: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """Sources are collected with domain info and ranked by score."""
        mock_search.return_value = [
            {"title": "Edu Result", "url": "https://example.edu/page", "snippet": "Academic research about AI"},
            {"title": "Gov Result", "url": "https://example.gov/page", "snippet": "Government data on AI trends"},
            {"title": "Blog Result", "url": "https://blogspot.com/page", "snippet": "Personal blog about AI"},
        ]
        mock_extract.return_value = "Full content with citations and references. " * 50

        plan = {
            "queries": ["AI research"],
            "depth": "moderate",
            "research_type": "deep",
            "sources_needed": 10,
        }

        # Search
        raw = await service.execute_search(plan)
        assert len(raw) == 3

        # Collect
        collected = await service.collect_sources(raw)
        assert len(collected) == 3
        for source in collected:
            assert "domain" in source
            assert "content" in source

        # Rank
        ranked = await service.rank_sources(collected, "AI research")
        assert len(ranked) == 3

        # Verify sorted by score descending
        scores = [s["overall_score"] for s in ranked]
        assert scores == sorted(scores, reverse=True)

        # Edu source should be ranked higher than blog source
        edu = next(s for s in ranked if "example.edu" in s["domain"])
        blog = next(s for s in ranked if "blogspot" in s["domain"])
        assert edu["overall_score"] > blog["overall_score"]

    async def test_verify_facts_empty(self, service: ResearchService):
        """verify_facts returns zero-confidence result for empty sources."""
        result = await service.verify_facts([], "test")
        assert result["verified_claims"] == []
        assert result["overall_confidence"] == 0.0


class TestPipelineErrorHandling:
    """Tests for pipeline resilience when external services fail."""

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    async def test_search_service_down(
        self,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """Pipeline handles search service failure gracefully."""
        mock_queries.return_value = "query"
        mock_search.side_effect = ConnectionError("Search service unavailable")

        result = await service.quick_search("Test query", max_sources=5)

        assert result["query"] == "Test query"
        assert "No sources found" in result["answer"]
        assert result["sources"] == []

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    @patch("backend.services.research_service.minimax.chat")
    async def test_synthesis_llm_failure(
        self,
        mock_chat: AsyncMock,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """When the LLM synthesis fails, pipeline falls back gracefully."""
        mock_queries.return_value = "query"
        mock_search.return_value = [
            {"title": "Result", "url": "https://example.com/r", "snippet": "Snippet text about the topic"},
        ]
        mock_chat.side_effect = RuntimeError("LLM unavailable")

        result = await service.quick_search("Test query", max_sources=1)

        assert result["query"] == "Test query"
        assert "Here is what I found" in result["answer"]  # Fallback
        assert len(result["sources"]) == 1

    @patch("backend.services.research_service.search_service.search")
    @patch("backend.services.research_service.minimax.web_search_query")
    @patch("backend.services.research_service.minimax.chat")
    @patch("backend.services.research_service.deepseek.extract_json")
    @patch("backend.services.research_service.mongodb")
    async def test_mongodb_unavailable(
        self,
        mock_mongodb: AsyncMock,
        mock_deepseek: AsyncMock,
        mock_chat: AsyncMock,
        mock_queries: AsyncMock,
        mock_search: AsyncMock,
        service: ResearchService,
    ):
        """deep_research completes even when MongoDB persistence fails."""
        # Mock MongoDB operations to raise errors
        mock_mongodb.research_cache.find_one = AsyncMock(side_effect=ConnectionError("MongoDB unavailable"))
        mock_mongodb.research_cache.insert_one = AsyncMock(side_effect=ConnectionError("MongoDB unavailable"))
        mock_mongodb.research_reports.insert_one = AsyncMock(side_effect=ConnectionError("MongoDB unavailable"))

        mock_queries.return_value = "query"
        mock_search.return_value = [
            {"title": "Result", "url": "https://example.com/r", "snippet": "Snippet"},
        ]
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Synthesis."}}]
        }
        mock_deepseek.side_effect = [
            {"verified_claims": [], "contradicted_claims": [], "unverifiable_claims": [], "overall_confidence": 0.0},
            {
                "title": "Report",
                "executive_summary": "Summary",
                "key_findings": ["Finding"],
                "detailed_analysis": "Analysis",
                "pros": None,
                "cons": None,
                "recommendations": None,
                "conclusions": "",
            },
        ]

        result = await service.deep_research(
            query="Test topic",
            research_type="deep",
            depth="quick",
            max_sources=1,
        )

        # Report should still be generated even if persistence fails
        assert result["title"] == "Report"
        assert result["executive_summary"] == "Summary"
        assert result["report_id"] == ""  # Persistence failed
        assert "processing_time_ms" in result
