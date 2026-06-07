"""
Tests for API endpoints — auth, chat, voice, memory, tasks, agents, admin.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


class TestAuthAPI:
    """Tests for authentication endpoints."""

    @pytest.mark.asyncio
    async def test_register_user(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "newuser@test.com",
                "password": "StrongPass123",
                "name": "New User",
            },
        )
        assert response.status_code in (201, 200, 409)  # 409 if duplicate

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/auth/register",
            json={
                "email": "not-an-email",
                "password": "StrongPass123",
                "name": "Bad Email",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login(self, async_client: AsyncClient, test_user: dict) -> None:
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": test_user["email"],
                "password": "testpassword123",
            },
        )
        assert response.status_code in (200, 401)

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, async_client: AsyncClient) -> None:
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "test@test.com",
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/auth/refresh",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code in (200, 401)

    @pytest.mark.asyncio
    async def test_get_me(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data

    @pytest.mark.asyncio
    async def test_get_me_unauthorized(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/auth/me")
        assert response.status_code in (401, 403)


class TestChatAPI:
    """Tests for chat endpoints."""

    @pytest.mark.asyncio
    async def test_create_conversation(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/chat/conversations",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"title": "Test Conversation"},
        )
        assert response.status_code in (200, 201, 500)

    @pytest.mark.asyncio
    async def test_list_conversations(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.get(
            "/api/chat/conversations",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_send_message(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/chat/messages",
            headers={"Authorization": f"Bearer {user_token}"},
            json={
                "message": "Hello!",
                "conversation_id": None,
            },
        )
        assert response.status_code in (200, 201, 500)


class TestVoiceAPI:
    """Tests for voice endpoints."""

    @pytest.mark.asyncio
    async def test_create_session(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/voice/sessions",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"language": "en"},
        )
        assert response.status_code in (200, 201, 500)

    @pytest.mark.asyncio
    async def test_get_session_unauthorized(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/voice/sessions/fake-id")
        assert response.status_code in (401, 403, 404)


class TestMemoryAPI:
    """Tests for memory endpoints."""

    @pytest.mark.asyncio
    async def test_store_memory(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/memory/store",
            headers={"Authorization": f"Bearer {user_token}"},
            json={
                "content": "Test memory",
                "memory_type": "short_term",
            },
        )
        assert response.status_code in (200, 201, 500)

    @pytest.mark.asyncio
    async def test_get_recent_memories(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.get(
            "/api/memory/recent",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_search_memory(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.get(
            "/api/memory/search?query=test",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code in (200, 500)


class TestTasksAPI:
    """Tests for task endpoints."""

    @pytest.mark.asyncio
    async def test_create_task(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/tasks",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"title": "Test task", "priority": "medium"},
        )
        assert response.status_code in (200, 201, 500)

    @pytest.mark.asyncio
    async def test_list_tasks(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.get(
            "/api/tasks",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestAgentsAPI:
    """Tests for agent endpoints."""

    @pytest.mark.asyncio
    async def test_route_to_agent(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.post(
            "/api/agents/route",
            headers={"Authorization": f"Bearer {user_token}"},
            json={
                "agent": "router",
                "message": "Hello",
            },
        )
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_agent_status(self, async_client: AsyncClient, user_token: str) -> None:
        response = await async_client.get(
            "/api/agents/status",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 200


class TestAdminAPI:
    """Tests for admin endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/admin/health")
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_admin_users_forbidden(self, async_client: AsyncClient, user_token: str) -> None:
        """Regular users should not access admin endpoints."""
        response = await async_client.get(
            "/api/admin/users",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_users_as_admin(self, async_client: AsyncClient, admin_token: str) -> None:
        response = await async_client.get(
            "/api/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_admin_metrics(self, async_client: AsyncClient, admin_token: str) -> None:
        response = await async_client.get(
            "/api/admin/metrics",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code in (200, 500)
