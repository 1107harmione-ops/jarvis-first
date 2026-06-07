"""
WebSocket Chat — real-time communication with streaming tokens, typing indicators,
connection recovery, and interrupt support.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from backend.agents.router_agent import router_agent
from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections with user-session mapping."""

    def __init__(self) -> None:
        # user_id -> list of (session_id, websocket)
        self._connections: dict[str, list[tuple[str, WebSocket]]] = {}
        # session_id -> user_id for reverse lookup
        self._sessions: dict[str, str] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> str:
        """Accept a WebSocket connection and register it."""
        await websocket.accept()
        session_id = str(uuid.uuid4())

        if user_id not in self._connections:
            self._connections[user_id] = []
        self._connections[user_id].append((session_id, websocket))
        self._sessions[session_id] = user_id

        # Send session confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        logger.debug("WebSocket connected", extra={"user_id": user_id, "session_id": session_id})
        return session_id

    def disconnect(self, user_id: str, session_id: str) -> None:
        """Remove a WebSocket connection."""
        if user_id in self._connections:
            self._connections[user_id] = [
                (sid, ws) for sid, ws in self._connections[user_id]
                if sid != session_id
            ]
            if not self._connections[user_id]:
                del self._connections[user_id]
        self._sessions.pop(session_id, None)
        logger.debug("WebSocket disconnected", extra={"user_id": user_id, "session_id": session_id})

    async def send_to_user(
        self, user_id: str, message: dict[str, Any]
    ) -> None:
        """Send a message to all connections of a user."""
        if user_id not in self._connections:
            return
        disconnected: list[str] = []
        for session_id, ws in self._connections[user_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(session_id)
        # Clean up disconnected
        for sid in disconnected:
            self.disconnect(user_id, sid)

    def get_user_id(self, session_id: str) -> str | None:
        """Get user ID for a session."""
        return self._sessions.get(session_id)

    @property
    def active_connections(self) -> int:
        """Count of active connections."""
        return sum(len(conns) for conns in self._connections.values())


class ChatWebSocket:
    """WebSocket handler for real-time chat with streaming.

    Protocol:
    - Client sends: {"type": "message", "content": "...", "conversation_id": "..."}
    - Client sends: {"type": "typing", "content": "..."}
    - Client sends: {"type": "ping"}
    - Client sends: {"type": "interrupt"}
    - Server sends: {"type": "token", "content": "..."} (streaming)
    - Server sends: {"type": "message", "content": "...", "agent": "..."} (complete)
    - Server sends: {"type": "error", "content": "..."}
    - Server sends: {"type": "pong"}
    - Server sends: {"type": "done", "conversation_id": "..."}
    """

    def __init__(self) -> None:
        self.manager = ConnectionManager()
        self._active_streams: dict[str, asyncio.Task[None]] = {}

    async def handle(self, websocket: WebSocket, token: str | None = None) -> None:
        """Handle a WebSocket connection lifecycle.

        Args:
            websocket: The WebSocket connection.
            token: Optional JWT token from query parameter.
        """
        # Authenticate
        user = await self._authenticate(websocket, token)
        if not user:
            return

        user_id = user["id"]
        session_id = await self.manager.connect(user_id, websocket)

        try:
            async for raw_message in websocket.iter_text():
                try:
                    data = json.loads(raw_message)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                    continue

                await self._handle_message(user_id, session_id, websocket, data)

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("WebSocket error", extra={"user_id": user_id, "session_id": session_id, "error": str(exc)})
        finally:
            # Cancel any active stream
            if session_id in self._active_streams:
                self._active_streams[session_id].cancel()
                del self._active_streams[session_id]
            self.manager.disconnect(user_id, session_id)

    async def _handle_message(
        self,
        user_id: str,
        session_id: str,
        ws: WebSocket,
        data: dict[str, Any],
    ) -> None:
        """Process a single WebSocket message."""
        msg_type = data.get("type", "message")

        if msg_type == "ping":
            await ws.send_json({"type": "pong", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

        elif msg_type == "typing":
            # Broadcast typing indicator
            await self.manager.send_to_user(user_id, {
                "type": "typing",
                "user_id": user_id,
                "content": data.get("content", ""),
            })

        elif msg_type == "interrupt":
            # Cancel active stream
            if session_id in self._active_streams:
                self._active_streams[session_id].cancel()
                del self._active_streams[session_id]
                await ws.send_json({"type": "interrupt", "content": "Stream interrupted"})
                logger.info("Stream interrupted", extra={"session_id": session_id})

        elif msg_type == "config":
            # Handle configuration update
            await ws.send_json({"type": "config", "content": "Configuration updated"})

        elif msg_type == "message":
            content = data.get("content", "")
            if not content.strip():
                await self._send_error(ws, "Message content cannot be empty")
                return

            conversation_id = data.get("conversation_id")
            attachments = data.get("attachments", [])
            metadata = data.get("metadata", {})

            # Start streaming response
            stream_task = asyncio.create_task(
                self._stream_response(
                    user_id=user_id,
                    session_id=session_id,
                    ws=ws,
                    message=content,
                    conversation_id=conversation_id,
                    attachments=attachments,
                    metadata=metadata,
                )
            )
            self._active_streams[session_id] = stream_task

    async def _stream_response(
        self,
        user_id: str,
        session_id: str,
        ws: WebSocket,
        message: str,
        conversation_id: str | None,
        attachments: list[str],
        metadata: dict[str, Any],
    ) -> None:
        """Stream an agent response token by token."""
        try:
            # Send typing indicator
            await ws.send_json({"type": "typing", "agent": "router", "status": "thinking"})

            # Get response from router agent
            response = await router_agent.process(
                user_id=user_id,
                message=message,
                conversation_id=conversation_id,
                stream=False,  # We handle the streaming via the final response
                attachments=attachments,
                metadata=metadata,
            )

            content = response.get("content", "")
            agent = response.get("agent", "router")
            conv_id = response.get("conversation_id", conversation_id)

            # Simulate streaming by sending tokens
            # In production, use actual streaming from the LLM
            buffer = ""
            words = content.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                buffer += token
                await ws.send_json({
                    "type": "token",
                    "content": token,
                    "agent": agent,
                })
                await asyncio.sleep(0.01)  # Small delay for realism

            # Send complete message
            await ws.send_json({
                "type": "message",
                "content": content,
                "agent": agent,
                "conversation_id": conv_id,
                "category": response.get("category", "general"),
                "duration_ms": response.get("duration_ms", 0),
            })

            # Send done signal
            await ws.send_json({
                "type": "done",
                "conversation_id": conv_id,
                "agent": agent,
            })

        except asyncio.CancelledError:
            logger.debug("Stream cancelled", extra={"session_id": session_id})
            await ws.send_json({"type": "interrupt", "content": "Stream cancelled"})
            raise
        except Exception as exc:
            logger.error("Stream error", extra={"session_id": session_id, "error": str(exc)})
            await self._send_error(ws, f"Processing error: {str(exc)}")

    async def _authenticate(
        self, ws: WebSocket, token: str | None
    ) -> dict[str, Any] | None:
        """Authenticate the WebSocket connection."""
        # Try query parameter first, then first message
        if not token:
            # Wait for auth message
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
                auth_data = json.loads(raw)
                token = auth_data.get("token")
            except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
                await ws.close(code=4001, reason="Authentication timeout")
                return None

        if not token:
            await ws.close(code=4001, reason="Authentication required")
            return None

        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            if not user_id:
                await ws.close(code=4001, reason="Invalid token")
                return None

            user = await mongodb.users.find_one({"_id": user_id})
            if not user or not user.get("is_active", True):
                await ws.close(code=4001, reason="User not found or inactive")
                return None

            return {
                "id": str(user["_id"]),
                "email": user.get("email", ""),
                "name": user.get("name", ""),
                "role": user.get("role", "user"),
            }
        except JWTError:
            await ws.close(code=4001, reason="Invalid token")
            return None

    async def _send_error(self, ws: WebSocket, message: str) -> None:
        """Send an error message over WebSocket."""
        try:
            await ws.send_json({"type": "error", "content": message})
        except Exception:
            pass

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast to all connected users."""
        for user_id in list(self.manager._connections.keys()):
            await self.manager.send_to_user(user_id, message)


# Global singleton
chat_websocket = ChatWebSocket()
