"""Voice command integration tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestVoiceAPI:
    """Test suite for voice command API."""

    async def test_voice_create_task(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "create a task to learn FastAPI",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_CREATE"
        assert "Task created" in data["spoken_response"]
        assert data["data"] is not None

    async def test_voice_list_tasks(self, client: AsyncClient):
        await client.post("/api/voice/command", json={
            "text": "create a task learn Python",
        })

        response = await client.post("/api/voice/command", json={
            "text": "show my tasks",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_LIST"
        assert "python" in data["spoken_response"].lower()

    async def test_voice_complete_task(self, client: AsyncClient):
        await client.post("/api/voice/command", json={
            "text": "create a task review code",
        })

        response = await client.post("/api/voice/command", json={
            "text": "complete my task review code",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_COMPLETE"
        assert "completed" in data["spoken_response"]

    async def test_voice_search_tasks(self, client: AsyncClient):
        await client.post("/api/voice/command", json={
            "text": "create a task learn WebRTC",
        })

        response = await client.post("/api/voice/command", json={
            "text": "search my tasks about WebRTC",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_SEARCH"

    async def test_voice_unknown_command(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "what is the meaning of life",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "UNKNOWN"
        assert "didn't understand" in data["spoken_response"]

    async def test_voice_empty_text(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "",
        })
        assert response.status_code == 400
