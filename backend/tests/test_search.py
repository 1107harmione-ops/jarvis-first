"""Search API tests."""
from __future__ import annotations

from httpx import AsyncClient


class TestSearchAPI:
    async def test_search_empty(self, client: AsyncClient):
        response = await client.get("/api/search?q=python")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["results"] == []

    async def test_search_tasks(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Learn Python"})
        await client.post("/api/tasks", json={"title": "Write Rust code"})

        response = await client.get("/api/search?q=python")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        titles = [r["title"] for r in data["results"]]
        assert "Learn Python" in titles

    async def test_search_notes(self, client: AsyncClient):
        await client.post("/api/notes", json={"title": "Meeting notes"})
        await client.post("/api/notes", json={"title": "Shopping list"})

        response = await client.get("/api/search?q=meeting")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any("Meeting" in r["title"] for r in data["results"])

    async def test_search_memory(self, client: AsyncClient):
        await client.post("/api/memory", json={"fact": "I love Python programming"})

        response = await client.get("/api/search?q=python")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        types = [r["type"] for r in data["results"]]
        assert "memory" in types

    async def test_search_filter_by_type(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Python task"})
        await client.post("/api/notes", json={"title": "Python notes"})

        response = await client.get("/api/search?q=python&type=task")
        assert response.status_code == 200
        data = response.json()
        assert all(r["type"] == "task" for r in data["results"])

    async def test_search_cross_entity(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Buy groceries"})
        await client.post("/api/notes", json={"title": "Groceries list", "content": "milk, eggs, bread"})
        await client.post("/api/memory", json={"fact": "Weekly groceries on Saturday"})

        response = await client.get("/api/search?q=groceries")
        assert response.status_code == 200
        data = response.json()
        types = {r["type"] for r in data["results"]}
        assert "task" in types
        assert "note" in types
        assert "memory" in types

    async def test_search_voice_global(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Review Python PR"})
        response = await client.post("/api/voice/command", json={
            "text": "search for python",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "GLOBAL_SEARCH"
        assert "python" in data["spoken_response"].lower()
