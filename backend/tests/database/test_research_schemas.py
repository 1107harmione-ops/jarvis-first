"""tests/database/test_research_schemas.py"""
import pytest
from datetime import datetime, timezone
from backend.database.schemas import new_research_report_doc, new_research_source_doc, new_research_cache_doc


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
