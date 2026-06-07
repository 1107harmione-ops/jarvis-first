"""
Tests for Voice Memory system — interaction history, command tracking, preferences.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVoiceMemoryService:
    """Tests for the voice memory service."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        assert len(service._cache) == 0
        assert service._cache_ttl == 300.0

    @pytest.mark.asyncio
    async def test_get_preferences_defaults(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        prefs = await service.get_preferences("unknown_user")
        assert isinstance(prefs, dict)
        assert prefs.get("language") == "en"
        assert prefs.get("voice_speed") == 1.0
        assert prefs.get("wake_word_enabled") is True
        assert prefs.get("interrupt_enabled") is True

    @pytest.mark.asyncio
    async def test_store_and_get_preferences(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        prefs = {"language": "hi", "voice_speed": 1.2, "wake_word_enabled": False}
        await service.store_preferences("user1", prefs)

        # Invalidate cache so it re-reads
        service._cache.pop(f"prefs_user1", None)
        retrieved = await service.get_preferences("user1")
        # Since no DB is connected, it will return defaults
        assert isinstance(retrieved, dict)

    @pytest.mark.asyncio
    async def test_get_history_empty(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        history = await service.get_history("user1")
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_search_history_empty(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        results = await service.search_history("user1", "test")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_history_no_query(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        results = await service.search_history("user1", "")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_frequent_commands_empty(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        commands = await service.get_frequent_commands("user1")
        assert isinstance(commands, list)

    @pytest.mark.asyncio
    async def test_get_voice_stats(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        stats = await service.get_voice_stats("user1")
        assert isinstance(stats, dict)
        assert "total_interactions" in stats
        assert "by_language" in stats
        assert "avg_confidence" in stats
        assert "interruptions" in stats

    @pytest.mark.asyncio
    async def test_get_history_count(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        count = await service.get_history_count("user1")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_get_history_with_pagination(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        history = await service.get_history("user1", limit=10, offset=0)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_store_session_metrics(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        # Should not raise even without DB
        await service.store_session_metrics(
            "user1", "session_1", {"stt_ms": 150, "tts_ms": 300}
        )

    @pytest.mark.asyncio
    async def test_get_session_metrics_empty(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        metrics = await service.get_session_metrics("user1")
        assert isinstance(metrics, list)

    @pytest.mark.asyncio
    async def test_cache_invalidation(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        service._cache["prefs_user1"] = {"language": "en"}
        service._cache_timestamps["prefs_user1"] = 0.0  # Expired

        retrieved = await service.get_preferences("user1")
        assert isinstance(retrieved, dict)

    @pytest.mark.asyncio
    async def test_store_interaction(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        # Should not raise even without DB
        result = await service.store_interaction(
            user_id="user1",
            session_id="session_1",
            transcript="hello",
            response="hi!",
            confidence=0.95,
            agent="router",
            language="en",
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_track_command(self) -> None:
        from backend.memory.voice_memory import VoiceMemoryService

        service = VoiceMemoryService()
        # Internal method, should not raise
        await service._track_command("user1", "what's the weather", "router")
