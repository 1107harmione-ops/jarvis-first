"""Task API tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestTaskAPI:
    """Test suite for task CRUD endpoints."""

    async def test_create_task(self, client: AsyncClient):
        response = await client.post("/api/tasks", json={
            "title": "Learn FastAPI",
            "description": "Study FastAPI documentation",
            "priority": "high",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Learn FastAPI"
        assert data["status"] == "pending"
        assert data["priority"] == "high"
        assert "id" in data

    async def test_create_task_empty_title(self, client: AsyncClient):
        response = await client.post("/api/tasks", json={"title": ""})
        assert response.status_code == 422

    async def test_list_tasks_empty(self, client: AsyncClient):
        response = await client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["tasks"] == []

    async def test_list_tasks_with_data(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Task 1"})
        await client.post("/api/tasks", json={"title": "Task 2"})

        response = await client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["tasks"]) == 2

    async def test_list_tasks_filter_status(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Pending Task"})
        resp = await client.post("/api/tasks", json={"title": "Done Task"})
        task_id = resp.json()["id"]
        await client.patch(f"/api/tasks/{task_id}/complete")

        response = await client.get("/api/tasks?status=completed")
        data = response.json()
        assert data["total"] == 1
        assert data["tasks"][0]["title"] == "Done Task"

    async def test_get_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Get Me"})
        task_id = create_resp.json()["id"]

        response = await client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Get Me"

    async def test_get_task_not_found(self, client: AsyncClient):
        response = await client.get("/api/tasks/999")
        assert response.status_code == 404

    async def test_complete_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Complete Me"})
        task_id = create_resp.json()["id"]

        response = await client.patch(f"/api/tasks/{task_id}/complete")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    async def test_delete_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Delete Me"})
        task_id = create_resp.json()["id"]

        response = await client.delete(f"/api/tasks/{task_id}")
        assert response.status_code == 204

        get_resp = await client.get(f"/api/tasks/{task_id}")
        assert get_resp.status_code == 404

    async def test_search_tasks(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Learn Python"})
        await client.post("/api/tasks", json={"title": "Learn Rust"})
        await client.post("/api/tasks", json={"title": "Write Tests"})

        response = await client.get("/api/tasks/search?q=Learn")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    async def test_update_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Original"})
        task_id = create_resp.json()["id"]

        response = await client.patch(f"/api/tasks/{task_id}", json={
            "title": "Updated",
            "priority": "urgent",
        })
        assert response.status_code == 200
        assert response.json()["title"] == "Updated"
        assert response.json()["priority"] == "urgent"
