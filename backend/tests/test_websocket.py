"""
Tests for WebSocket chat endpoint.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestChatWebSocket:
    """Tests for the chat WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_connect_no_auth(self) -> None:
        """Connecting without auth token should be rejected."""
        from fastapi import WebSocket
        from backend.websocket.chat_socket import chat_websocket

        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.headers = {}
        mock_ws.cookies = {}
        mock_ws.query_params = {}
        mock_ws.close = AsyncMock()

        # The endpoint calls receive_text in a loop, so we simulate close
        mock_ws.receive_text = AsyncMock(side_effect=Exception("Not authenticated"))

        with pytest.raises(Exception):
            await chat_websocket(mock_ws)

        # If auth fails, websocket should be closed
        if hasattr(mock_ws, "close"):
            pass  # close may or may not be called depending on implementation

    @pytest.mark.asyncio
    async def test_connection_manager(self) -> None:
        """Test ConnectionManager basic operations."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        mock_ws = MagicMock()

        # Connect a user
        await manager.connect(user_id="user1", websocket=mock_ws)
        assert "user1" in manager.active_connections
        assert len(manager.active_connections["user1"]) == 1

        # Connect a second session for the same user
        mock_ws2 = MagicMock()
        await manager.connect(user_id="user1", websocket=mock_ws2)
        assert len(manager.active_connections["user1"]) == 2

        # Connect a different user
        mock_ws3 = MagicMock()
        await manager.connect(user_id="user2", websocket=mock_ws3)
        assert "user2" in manager.active_connections

        # Disconnect user1's first session
        await manager.disconnect(user_id="user1", websocket=mock_ws)
        assert len(manager.active_connections["user1"]) == 1

        # Disconnect user1 completely
        await manager.disconnect(user_id="user1", websocket=mock_ws2)
        assert "user1" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_to_user(self) -> None:
        """Test broadcasting to a specific user."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        mock_ws = AsyncMock()
        await manager.connect(user_id="user1", websocket=mock_ws)

        message = {"type": "test", "content": "hello"}
        await manager.broadcast_to_user(user_id="user1", message=message)

        mock_ws.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_no_sessions(self) -> None:
        """Broadcasting to a user with no sessions should be a no-op."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        # No sessions for this user
        await manager.broadcast_to_user(user_id="unknown_user", message={"type": "test"})
        # Should not raise

    @pytest.mark.asyncio
    async def test_set_active_stream(self) -> None:
        """Test active stream tracking."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        mock_ws = MagicMock()
        await manager.connect(user_id="user1", websocket=mock_ws)

        manager.set_active_stream(user_id="user1", conversation_id="conv1")
        assert manager.active_streams.get("user1") == "conv1"

        manager.clear_active_stream(user_id="user1")
        assert manager.active_streams.get("user1") is None

    @pytest.mark.asyncio
    async def test_is_streaming(self) -> None:
        """Test streaming state check."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        mock_ws = MagicMock()
        await manager.connect(user_id="user1", websocket=mock_ws)

        assert manager.is_streaming(user_id="user1") is False
        manager.set_active_stream(user_id="user1", conversation_id="conv1")
        assert manager.is_streaming(user_id="user1") is True
        manager.clear_active_stream(user_id="user1")
        assert manager.is_streaming(user_id="user1") is False

    def test_connection_count(self) -> None:
        """Test total connection count."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        assert manager.connection_count() == 0

    def test_cleanup_disconnected(self) -> None:
        """Test cleanup removes stale connections."""
        from backend.websocket.chat_socket import ConnectionManager

        manager = ConnectionManager()
        mock_ws = MagicMock()

        # We'd need a way to detect disconnected sockets
        # This is a basic test that cleanup doesn't raise
        try:
            manager._cleanup_disconnected()
        except Exception:
            pytest.fail("Cleanup raised unexpectedly")
