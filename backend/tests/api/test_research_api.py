"""tests/api/test_research_api.py"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from backend.main import app

pytestmark = pytest.mark.asyncio


class TestResearchAPI:
    @pytest_asyncio.fixture
    async def client(self):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac

    async def test_search_endpoint_requires_auth(self, client):
        resp = await client.post("/api/v2/research/search", json={"query": "test"})
        assert resp.status_code == 401

    async def test_deep_endpoint_requires_auth(self, client):
        resp = await client.post("/api/v2/research/deep", json={"query": "test"})
        assert resp.status_code == 401

    async def test_verify_endpoint_requires_auth(self, client):
        resp = await client.post("/api/v2/research/verify", json={"content": "test content for verification purposes here"})
        assert resp.status_code == 401

    async def test_history_endpoint_requires_auth(self, client):
        resp = await client.get("/api/v2/research/history")
        assert resp.status_code == 401

    async def test_reports_endpoint_requires_auth(self, client):
        resp = await client.get("/api/v2/research/reports/someid")
        assert resp.status_code == 401

    async def test_sources_endpoint_requires_auth(self, client):
        resp = await client.get("/api/v2/research/sources")
        assert resp.status_code == 401
