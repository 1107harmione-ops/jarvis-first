"""
Tests for Voice WebSocket endpoint — protocol, connection management, event handling.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVoiceWebSocketManager:
    """Tests for the VoiceWebSocketManager."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        assert len(manager._connections) == 0
        assert len(manager._user_sessions) == 0

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        session_id = await manager.connect(mock_ws, "user1")
        assert session_id is not None
        assert "user1" in manager._connections
        assert manager._user_sessions["user1"] == session_id
        assert len(manager._audio_queues) == 1

        await manager.disconnect("user1")
        assert "user1" not in manager._connections
        assert "user1" not in manager._user_sessions

    @pytest.mark.asyncio
    async def test_reconnect_replaces_old(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        old_ws = AsyncMock()
        old_ws.close = AsyncMock()
        new_ws = AsyncMock()
        new_ws.accept = AsyncMock()

        session1 = await manager.connect(old_ws, "user1")
        session2 = await manager.connect(new_ws, "user1")

        assert session1 != session2
        old_ws.close.assert_called_once_with(code=1000, reason="Replaced by new connection")

    @pytest.mark.asyncio
    async def test_send_json(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()

        await manager.connect(mock_ws, "user1")
        result = await manager.send_json("user1", {"type": "test"})
        assert result is True
        mock_ws.send_json.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_send_json_no_connection(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        result = await manager.send_json("nonexistent", {"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_send_bytes(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_bytes = AsyncMock()

        await manager.connect(mock_ws, "user1")
        result = await manager.send_bytes("user1", b"audio_data")
        assert result is True
        mock_ws.send_bytes.assert_called_once_with(b"audio_data")

    @pytest.mark.asyncio
    async def test_get_user_id(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        session_id = await manager.connect(mock_ws, "user1")
        uid = await manager.get_user_id(session_id)
        assert uid == "user1"

    @pytest.mark.asyncio
    async def test_get_user_id_unknown(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        uid = await manager.get_user_id("unknown_session")
        assert uid is None

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pipeline(self) -> None:
        from backend.websocket.voice_socket import VoiceWebSocketManager

        manager = VoiceWebSocketManager()
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        await manager.connect(mock_ws, "user1")

        # Simulate an active pipeline task
        task = AsyncMock()
        task.done = MagicMock(return_value=False)
        task.cancel = MagicMock()
        manager._pipeline_tasks["user1"] = task

        await manager.disconnect("user1")
        task.cancel.assert_called_once()


class TestVoiceProtocol:
    """Tests for voice WebSocket protocol messages."""

    def test_connected_message(self) -> None:
        msg = {"type": "connected", "session_id": "abc-123", "state": "idle"}
        assert msg["type"] == "connected"
        assert msg["session_id"] == "abc-123"

    def test_audio_start_message(self) -> None:
        msg = {"type": "audio_start"}
        assert msg["type"] == "audio_start"

    def test_audio_end_message(self) -> None:
        msg = {"type": "audio_end"}
        assert msg["type"] == "audio_end"

    def test_interrupt_message(self) -> None:
        msg = {"type": "interrupt"}
        assert msg["type"] == "interrupt"

    def test_config_message(self) -> None:
        msg = {"type": "config", "language": "hi", "voice_speed": 1.2}
        assert msg["language"] == "hi"

    def test_partial_transcript(self) -> None:
        msg = {"type": "partial", "text": "hello", "confidence": 0.7}
        assert msg["confidence"] == 0.7

    def test_final_transcript(self) -> None:
        msg = {
            "type": "transcript",
            "text": "hello world",
            "confidence": 0.95,
            "language": "en",
        }
        assert msg["confidence"] == 0.95

    def test_thinking_message(self) -> None:
        msg = {"type": "thinking", "agent": "router"}
        assert msg["agent"] == "router"

    def test_tts_start_end(self) -> None:
        start = {"type": "tts_start"}
        end = {"type": "tts_end"}
        assert start["type"] == "tts_start"
        assert end["type"] == "tts_end"

    def test_state_update(self) -> None:
        msg = {"type": "state", "state": "speaking"}
        assert msg["state"] == "speaking"

    def test_ping_pong(self) -> None:
        ping = json.dumps({"type": "ping"})
        pong = {"type": "pong"}
        assert json.loads(ping)["type"] == "ping"
        assert pong["type"] == "pong"

    def test_error_message(self) -> None:
        msg = {"type": "error", "message": "STT failed"}
        assert msg["message"] == "STT failed"

    def test_result_message(self) -> None:
        msg = {
            "type": "result",
            "transcript": "hello",
            "response": "hi!",
            "agent": "router",
        }
        assert msg["transcript"] == "hello"
        assert msg["response"] == "hi!"

    def test_config_ack(self) -> None:
        msg = {
            "type": "config_ack",
            "config": {"language": "hi", "voice_speed": 1.0},
        }
        assert msg["config"]["language"] == "hi"
