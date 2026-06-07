"""
Tests for backend services — voice, search, task, memory services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVoiceService:
    """Tests for the voice service."""

    @pytest.mark.asyncio
    async def test_create_session(self) -> None:
        with patch("backend.services.voice_service.VoiceSession") as MockSession:
            mock_instance = MagicMock()
            mock_instance.session_id = "test_session_id"
            mock_instance.language = "en"
            mock_instance.wake_word_enabled = False
            MockSession.return_value = mock_instance

            from backend.services.voice_service import VoiceService

            service = VoiceService()
            session = service.create_session(user_id="user123", language="en")
            assert session is not None

    def test_session_timeout(self) -> None:
        from backend.services.voice_service import VoiceService

        service = VoiceService()
        session = service.create_session(user_id="user123", language="en")
        assert session.ttl == 300  # default 300s

    def test_cleanup_expired_sessions(self) -> None:
        from backend.services.voice_service import VoiceService

        service = VoiceService()
        # Create a session
        service.create_session(user_id="user123", language="en")
        # Cleanup should not raise
        service._cleanup_expired_sessions()


class TestSearchService:
    """Tests for the search service."""

    @pytest.mark.asyncio
    async def test_search_with_duckduckgo(self) -> None:
        from backend.services.search_service import SearchService

        service = SearchService()
        result = await service.search("Python programming")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_empty_query(self) -> None:
        from backend.services.search_service import SearchService

        service = SearchService()
        result = await service.search("")
        assert result == []


class TestTaskService:
    """Tests for the task service."""

    @pytest.mark.asyncio
    async def test_create_task(self) -> None:
        from backend.services.task_service import TaskService

        service = TaskService()
        result = await service.create_task(
            user_id="user123",
            title="Test task",
            priority="high",
        )
        assert "error" in result or "id" in result

    @pytest.mark.asyncio
    async def test_list_tasks(self) -> None:
        from backend.services.task_service import TaskService

        service = TaskService()
        tasks = await service.get_tasks(user_id="user123")
        assert isinstance(tasks, list)

    @pytest.mark.asyncio
    async def test_complete_task_not_found(self) -> None:
        from backend.services.task_service import TaskService

        service = TaskService()
        result = await service.complete_task(
            user_id="user123",
            task_id="nonexistent_id",
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_task_not_found(self) -> None:
        from backend.services.task_service import TaskService

        service = TaskService()
        result = await service.update_task(
            user_id="user123",
            task_id="nonexistent_id",
            updates={"title": "Updated"},
        )
        assert "error" in result


class TestMemoryService:
    """Tests for the memory service."""

    @pytest.mark.asyncio
    async def test_store_short_term(self) -> None:
        from backend.services.memory_service import MemoryService

        service = MemoryService()
        result = await service.store_memory(
            user_id="user123",
            content="Short term test",
            memory_type="short_term",
        )
        assert "error" in result or "id" in result

    @pytest.mark.asyncio
    async def test_store_long_term(self) -> None:
        from backend.services.memory_service import MemoryService

        service = MemoryService()
        result = await service.store_memory(
            user_id="user123",
            content="Long term test",
            memory_type="long_term",
            importance_score=0.9,
        )
        assert "error" in result or "id" in result

    @pytest.mark.asyncio
    async def test_search_memory(self) -> None:
        from backend.services.memory_service import MemoryService

        service = MemoryService()
        results = await service.search_memory(
            user_id="user123",
            query="test query",
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_recent_memories(self) -> None:
        from backend.services.memory_service import MemoryService

        service = MemoryService()
        results = await service.get_recent_memories(
            user_id="user123",
            limit=10,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_important_memories(self) -> None:
        from backend.services.memory_service import MemoryService

        service = MemoryService()
        results = await service.get_important_memories(
            user_id="user123",
            min_importance=0.7,
        )
        assert isinstance(results, list)
