"""Memory API tests."""
from __future__ import annotations

from httpx import AsyncClient


class TestMemoryAPI:
    """Test suite for memory CRUD endpoints."""

    async def test_store_memory(self, client: AsyncClient):
        response = await client.post("/api/memory", json={
            "fact": "My name is John",
            "category": "personal",
            "importance": 5,
        })
        assert response.status_code == 201
        data = response.json()
        assert "John" in data["fact"]
        assert data["importance"] == 5

    async def test_store_memory_skips_greetings(self, client: AsyncClient):
        response = await client.post("/api/memory", json={
            "fact": "hello",
        })
        assert response.status_code == 400

    async def test_list_memory_empty(self, client: AsyncClient):
        response = await client.get("/api/memory")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["entries"] == []

    async def test_list_memory_with_data(self, client: AsyncClient):
        await client.post("/api/memory", json={"fact": "I like Python"})
        await client.post("/api/memory", json={"fact": "I use FastAPI"})

        response = await client.get("/api/memory")
        assert response.status_code == 200
        assert response.json()["total"] == 2

    async def test_search_memory(self, client: AsyncClient):
        await client.post("/api/memory", json={"fact": "Learning Go"})
        await client.post("/api/memory", json={"fact": "Learning Rust"})
        await client.post("/api/memory", json={"fact": "Write more tests"})

        response = await client.get("/api/memory/search?q=Learning")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    async def test_forget_memory(self, client: AsyncClient):
        create_resp = await client.post("/api/memory", json={"fact": "Delete me"})
        entry_id = create_resp.json()["id"]

        response = await client.delete(f"/api/memory/{entry_id}")
        assert response.status_code == 204

    async def test_voice_remember(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "remember that I love Python",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "MEMORY_SAVE"

    async def test_voice_recall(self, client: AsyncClient):
        await client.post("/api/memory", json={"fact": "I love Python"})
        response = await client.post("/api/voice/command", json={
            "text": "what do you know about Python",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "MEMORY_RECALL"

    async def test_voice_forget(self, client: AsyncClient):
        await client.post("/api/memory", json={"fact": "I love Python"})
        response = await client.post("/api/voice/command", json={
            "text": "forget that Python",
        })
        assert response.status_code == 200
        data = response.json()
        assert "forgot" in data["spoken_response"].lower()
        assert "python" in data["spoken_response"].lower()
