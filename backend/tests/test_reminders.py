"""Reminder API tests."""
from __future__ import annotations

import datetime

from httpx import AsyncClient


class TestReminderAPI:
    """Test suite for reminder CRUD endpoints."""

    async def test_create_reminder(self, client: AsyncClient):
        future = (datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat()
        response = await client.post("/api/reminders", json={
            "title": "Meeting",
            "reminder_time": future,
            "repeat_type": "none",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Meeting"
        assert data["status"] == "pending"
        assert data["repeat_type"] == "none"

    async def test_list_reminders_empty(self, client: AsyncClient):
        response = await client.get("/api/reminders")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    async def test_list_reminders_with_data(self, client: AsyncClient):
        future = (datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat()
        await client.post("/api/reminders", json={"title": "R1", "reminder_time": future})
        await client.post("/api/reminders", json={"title": "R2", "reminder_time": future})

        response = await client.get("/api/reminders")
        assert response.status_code == 200
        assert response.json()["total"] == 2

    async def test_get_reminder(self, client: AsyncClient):
        future = (datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat()
        create_resp = await client.post("/api/reminders", json={"title": "Get Me", "reminder_time": future})
        reminder_id = create_resp.json()["id"]

        response = await client.get(f"/api/reminders/{reminder_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Get Me"

    async def test_get_reminder_not_found(self, client: AsyncClient):
        response = await client.get("/api/reminders/999")
        assert response.status_code == 404

    async def test_delete_reminder(self, client: AsyncClient):
        future = (datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat()
        create_resp = await client.post("/api/reminders", json={"title": "Delete", "reminder_time": future})
        reminder_id = create_resp.json()["id"]

        response = await client.delete(f"/api/reminders/{reminder_id}")
        assert response.status_code == 204

        get_resp = await client.get(f"/api/reminders/{reminder_id}")
        assert get_resp.status_code == 404

    async def test_voice_create_reminder(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "remind me to call doctor tomorrow",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "REMINDER_CREATE"
        assert "Reminder set" in data["spoken_response"]
