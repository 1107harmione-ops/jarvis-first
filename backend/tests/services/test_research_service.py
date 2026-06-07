"""tests/services/test_research_service.py

Tests for the ResearchService — core research pipeline.
Uses ``pytest`` with asyncio mark for all async tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.services.research_service import ResearchService

pytestmark = pytest.mark.asyncio


class TestResearchService:
    """Tests for the ResearchService singleton — keyword detection, scoring,
    and logic that does not depend on external LLM / search calls."""

    # ── Research Type Detection ───────────────────────────────────

    async def test_detect_research_type_identifies_quick(self):
        service = ResearchService()
        rtype = await service.detect_research_type(
            "What is the capital of France?"
        )
        assert rtype == "quick"

    async def test_detect_research_type_identifies_deep(self):
        service = ResearchService()
        rtype = await service.detect_research_type(
            "I need a comprehensive deep research report on "
            "climate change impacts"
        )
        assert rtype == "deep"

    async def test_detect_research_type_identifies_comparative(self):
        service = ResearchService()
        rtype = await service.detect_research_type(
            "Compare and contrast Python vs JavaScript for "
            "web development"
        )
        assert rtype == "comparative"

    async def test_detect_research_type_identifies_technical(self):
        service = ResearchService()
        rtype = await service.detect_research_type(
            "What are the technical specifications of the new "
            "ARM architecture?"
        )
        assert rtype == "technical"

    async def test_detect_research_type_defaults_to_quick(self):
        service = ResearchService()
        rtype = await service.detect_research_type("Hello there")
        assert rtype == "quick"

    # ── Source Scoring ────────────────────────────────────────────

    async def test_score_source_edu_domain_gets_authority_bonus(self):
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

    async def test_score_source_unknown_domain_base_score(self):
        service = ResearchService()
        source = {
            "url": "https://random-blog.example.com/page",
            "title": "Some Title",
            "snippet": "Short",
            "domain": "random-blog.example.com",
        }
        scored = await service.score_source(source, "")
        assert scored["authority_score"] == 0.3  # Base only

    async def test_score_source_freshness_no_date(self):
        service = ResearchService()
        scored = await service.score_source(
            {
                "url": "https://example.com",
                "title": "T",
                "snippet": "S",
                "domain": "example.com",
            },
            "",
        )
        assert scored["freshness_score"] == 0.5  # Unknown date

    # ── Fact Verification ─────────────────────────────────────────

    async def test_verify_facts_no_sources(self):
        service = ResearchService()
        result = await service.verify_facts([], "test")
        assert result["verified_claims"] == []
        assert result["overall_confidence"] == 0.0

    # ── Domain Extraction ─────────────────────────────────────────

    async def test_extract_domain_from_url(self):
        service = ResearchService()
        assert (
            service._extract_domain("https://www.example.com/path")
            == "www.example.com"
        )
        assert (
            service._extract_domain("http://example.edu") == "example.edu"
        )

    # ── Sources For Depth ─────────────────────────────────────────

    async def test_sources_for_depth(self):
        service = ResearchService()
        assert service._sources_for_depth("quick") == 5
        assert service._sources_for_depth("moderate") == 10
        assert service._sources_for_depth("deep") == 20
        assert service._sources_for_depth("comprehensive") == 30
        assert service._sources_for_depth("unknown") == 10

    # ── Quick Search (no mocks — handles failures gracefully) ─────

    async def test_quick_search_handles_no_results(self):
        service = ResearchService()
        result = await service.quick_search(
            "xyznonexistent12345", max_sources=5
        )
        assert "query" in result
        assert "answer" in result
        assert "sources" in result


class TestResearchServiceMocked:
    """Tests with mocked external dependencies (LLMs, search service)."""

    # ── Query Generation ──────────────────────────────────────────

    @patch("backend.services.research_service.minimax.web_search_query")
    async def test_generate_queries(self, mock_queries):
        mock_queries.return_value = "query1\nquery2\nquery3"
        service = ResearchService()
        queries = await service._generate_queries("test", "quick")
        assert len(queries) == 3
        mock_queries.assert_called_once()

    # ── Search Execution ──────────────────────────────────────────

    @patch("backend.services.research_service.search_service.search")
    async def test_execute_search_returns_deduplicated(
        self, mock_search
    ):
        mock_search.return_value = [
            {
                "title": "A",
                "url": "https://example.com/1",
                "snippet": "S1",
            },
            {
                "title": "B",
                "url": "https://example.com/2",
                "snippet": "S2",
            },
        ]
        service = ResearchService()
        plan = {
            "queries": ["test"],
            "depth": "quick",
            "research_type": "quick",
            "sources_needed": 5,
        }
        results = await service.execute_search(plan)
        assert len(results) == 2
        assert results[0]["title"] == "A"

    # ── Fact Verification (mocked) ────────────────────────────────

    @patch("backend.services.research_service.deepseek.extract_json")
    async def test_verify_facts_with_sources(self, mock_extract):
        mock_extract.return_value = {
            "verified_claims": ["Claim 1"],
            "contradicted_claims": [],
            "unverifiable_claims": ["Claim 2"],
            "overall_confidence": 0.8,
        }
        service = ResearchService()
        sources = [
            {
                "title": "S1",
                "url": "https://example.com/1",
                "snippet": "Content 1",
                "domain": "example.com",
            },
        ]
        result = await service.verify_facts(sources, "test query")
        assert result["verified_claims"] == ["Claim 1"]
        assert result["overall_confidence"] == 0.8

    # ── Synthesis (mocked) ────────────────────────────────────────

    @patch("backend.services.research_service.minimax.chat")
    async def test_synthesize_with_results(self, mock_chat):
        mock_chat.return_value = {
            "choices": [{"message": {"content": "Synthesized answer."}}]
        }
        service = ResearchService()
        sources = [
            {
                "title": "S1",
                "url": "https://example.com",
                "snippet": "Content",
            }
        ]
        result = await service._synthesize("test query", sources)
        assert "Synthesized answer" in result

    # ── Report Generation (mocked) ────────────────────────────────

    @patch("backend.services.research_service.deepseek.extract_json")
    async def test_generate_report(self, mock_extract):
        mock_extract.return_value = {
            "title": "Report Title",
            "executive_summary": "Summary",
            "key_findings": ["Finding 1"],
            "detailed_analysis": "Analysis",
            "pros": ["Pro 1"],
            "cons": ["Con 1"],
            "recommendations": ["Rec 1"],
            "conclusions": "Conclusion",
        }
        service = ResearchService()
        verification = {
            "verified_claims": [],
            "contradicted_claims": [],
            "unverifiable_claims": [],
            "overall_confidence": 0.5,
        }
        result = await service._generate_report(
            "synthesis", verification, "deep", "moderate"
        )
        assert result["title"] == "Report Title"
        assert result["key_findings"] == ["Finding 1"]
