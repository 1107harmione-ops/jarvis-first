"""
Tests for Offline Mode Handler — service status monitoring, command queuing, fallback responses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOfflineHandler:
    """Tests for the offline handler."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        from backend.services.offline_handler import OfflineHandler, ServiceStatus

        handler = OfflineHandler()
        assert handler._running is False
        assert handler._processing_queue is False
        assert handler._monitor_task is None

        # All services start as UNKNOWN
        for status in handler._status.values():
            assert status == ServiceStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_service_status_enum(self) -> None:
        from backend.services.offline_handler import ServiceStatus

        assert ServiceStatus.ONLINE.value == "online"
        assert ServiceStatus.OFFLINE.value == "offline"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNKNOWN.value == "unknown"

    @pytest.mark.asyncio
    async def test_is_fully_online_when_unknown(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        # All UNKNOWN — not fully online
        assert handler.is_fully_online is False

    @pytest.mark.asyncio
    async def test_is_online_with_db(self) -> None:
        from backend.services.offline_handler import OfflineHandler, ServiceStatus

        handler = OfflineHandler()
        handler._status["database"] = ServiceStatus.ONLINE
        assert handler.is_online is True

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        await handler.start_monitoring()
        assert handler._running is True
        assert handler._monitor_task is not None

        await handler.stop_monitoring()
        assert handler._running is False

    @pytest.mark.asyncio
    async def test_offline_response_greeting(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("hello")
        assert "offline" in response.lower() or "hello" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_stop(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("stop")
        assert "stop" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_help(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("help")
        assert "offline" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_status(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("what is my status")
        assert "offline" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_queue(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("queue my request")
        assert "queue" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_time(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("what time is it")
        assert "time" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_generic(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        response = await handler.get_offline_response("do something random")
        assert "offline" in response.lower()

    @pytest.mark.asyncio
    async def test_offline_response_hindi(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        # Even with Hindi, should respond in English (basic fallback)
        response = await handler.get_offline_response("नमस्ते", language="hi")
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_queue_command(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        # Should not raise even without DB
        queue_id = await handler.queue_command("user1", "test command")
        assert isinstance(queue_id, str)

    @pytest.mark.asyncio
    async def test_get_queued_commands_empty(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        commands = await handler.get_queued_commands("user1")
        assert isinstance(commands, list)

    @pytest.mark.asyncio
    async def test_status_summary(self) -> None:
        from backend.services.offline_handler import OfflineHandler, ServiceStatus

        handler = OfflineHandler()
        summary = handler.status_summary
        assert isinstance(summary, dict)
        assert "llm" in summary
        assert "stt" in summary
        assert "database" in summary

    @pytest.mark.asyncio
    async def test_is_stt_available(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        # UNKNOWN — considered available
        assert handler.is_stt_available is True

    @pytest.mark.asyncio
    async def test_is_tts_available_default(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        # Piper may be enabled by default
        assert isinstance(handler.is_tts_available, bool)

    @pytest.mark.asyncio
    async def test_process_queued_commands_empty(self) -> None:
        from backend.services.offline_handler import OfflineHandler

        handler = OfflineHandler()
        count = await handler.process_queued_commands("user1")
        assert count == 0
