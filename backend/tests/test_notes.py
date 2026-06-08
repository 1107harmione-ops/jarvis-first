"""Note API tests."""
from __future__ import annotations

from httpx import AsyncClient


class TestNoteAPI:
    """Test suite for note CRUD endpoints."""

    async def test_create_note(self, client: AsyncClient):
        response = await client.post("/api/notes", json={
            "title": "FastAPI Notes",
            "content": "FastAPI is great for building APIs",
            "category": "learning",
            "priority": "high",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "FastAPI Notes"
        assert data["category"] == "learning"
        assert data["priority"] == "high"
        assert "id" in data

    async def test_create_note_empty_title(self, client: AsyncClient):
        response = await client.post("/api/notes", json={"title": ""})
        assert response.status_code == 422

    async def test_list_notes_empty(self, client: AsyncClient):
        response = await client.get("/api/notes")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["notes"] == []

    async def test_list_notes_with_data(self, client: AsyncClient):
        await client.post("/api/notes", json={"title": "Note 1"})
        await client.post("/api/notes", json={"title": "Note 2"})

        response = await client.get("/api/notes")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["notes"]) == 2

    async def test_get_note(self, client: AsyncClient):
        create_resp = await client.post("/api/notes", json={"title": "Get Me"})
        note_id = create_resp.json()["id"]

        response = await client.get(f"/api/notes/{note_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Get Me"

    async def test_get_note_not_found(self, client: AsyncClient):
        response = await client.get("/api/notes/999")
        assert response.status_code == 404

    async def test_update_note(self, client: AsyncClient):
        create_resp = await client.post("/api/notes", json={"title": "Original"})
        note_id = create_resp.json()["id"]

        response = await client.patch(f"/api/notes/{note_id}", json={
            "title": "Updated",
            "priority": "urgent",
        })
        assert response.status_code == 200
        assert response.json()["title"] == "Updated"
        assert response.json()["priority"] == "urgent"

    async def test_delete_note(self, client: AsyncClient):
        create_resp = await client.post("/api/notes", json={"title": "Delete Me"})
        note_id = create_resp.json()["id"]

        response = await client.delete(f"/api/notes/{note_id}")
        assert response.status_code == 204

        get_resp = await client.get(f"/api/notes/{note_id}")
        assert get_resp.status_code == 404

    async def test_search_notes(self, client: AsyncClient):
        await client.post("/api/notes", json={"title": "Learn Python"})
        await client.post("/api/notes", json={"title": "Learn Rust"})
        await client.post("/api/notes", json={"title": "Write Tests"})

        response = await client.get("/api/notes/search?q=Learn")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    async def test_voice_create_note(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "create a note about voice APIs",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "NOTE_CREATE"
        assert "Note created" in data["spoken_response"]

    async def test_voice_search_note(self, client: AsyncClient):
        await client.post("/api/voice/command", json={
            "text": "create a note about WebRTC",
        })

        response = await client.post("/api/voice/command", json={
            "text": "search my notes about WebRTC",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "NOTE_SEARCH"
