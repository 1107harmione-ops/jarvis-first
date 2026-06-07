"""
Voice WebSocket Endpoint — Real-time streaming voice communication.

Protocol:
  Client → Server:
    - Binary: PCM16 audio chunks (16000Hz, MONO)
    - JSON: control messages (audio_start, audio_end, interrupt, config, ping)

  Server → Client:
    - Binary: PCM16 audio chunks (22050Hz, MONO) for TTS
    - JSON: state, partials, transcripts, thinking, tts_start/end, error, pong

Supports:
  - Streaming audio capture → STT → Agent → TTS → Playback
  - Interrupt handling (user cuts off response)
  - Multi-language (English, Hindi)
  - Connection lifecycle management
  - Per-user session tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backend.config.settings import settings
from backend.services.voice_pipeline import (
    PipelineConfig,
    PipelineEvent,
    PipelineState,
    pipeline_manager,
)
from backend.services.whisper_stt import whisper_service
from backend.services.piper_tts import piper_service
from backend.utils.auth import decode_token
from backend.utils.logger import get_logger

logger = logging.getLogger("jarvis.voice_socket")


class VoiceWebSocketManager:
    """
    Manages voice WebSocket connections and their lifecycle.

    Each user can have one active voice session. Multiple connections
    from the same user are handled by replacing the old one.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}  # user_id → websocket
        self._user_sessions: dict[str, str] = {}  # user_id → session_id
        self._session_users: dict[str, str] = {}  # session_id → user_id
        self._audio_queues: dict[str, asyncio.Queue[bytes | None]] = {}
        self._pipeline_tasks: dict[str, asyncio.Task[Any]] = {}

    async def connect(
        self, websocket: WebSocket, user_id: str, session_id: str | None = None
    ) -> str:
        """Accept a new voice WebSocket connection."""
        await websocket.accept()
        sid = session_id or str(uuid.uuid4())

        # Close existing connection for this user if any
        existing = self._connections.get(user_id)
        if existing:
            try:
                await existing.close(code=1000, reason="Replaced by new connection")
            except Exception:
                pass

        self._connections[user_id] = websocket
        self._user_sessions[user_id] = sid
        self._session_users[sid] = user_id
        self._audio_queues[user_id] = asyncio.Queue()

        logger.info("Voice WS connected: user=%s session=%s", user_id, sid)
        return sid

    async def disconnect(self, user_id: str) -> None:
        """Disconnect a voice WebSocket for a user."""
        # Cancel any active pipeline
        task = self._pipeline_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Clean up audio queue
        self._audio_queues.pop(user_id, None)

        # Clean up session mappings
        sid = self._user_sessions.pop(user_id, None)
        if sid:
            self._session_users.pop(sid, None)
        self._connections.pop(user_id, None)

        # Clean up pipeline manager
        pipeline_manager.remove_pipeline(user_id)

        logger.info("Voice WS disconnected: user=%s", user_id)

    async def get_user_id(self, session_id: str) -> str | None:
        """Get user ID from session ID."""
        return self._session_users.get(session_id)

    async def send_json(self, user_id: str, message: dict[str, Any]) -> bool:
        """Send a JSON message to a user's voice WebSocket."""
        ws = self._connections.get(user_id)
        if not ws:
            return False
        try:
            await ws.send_json(message)
            return True
        except Exception:
            return False

    async def send_bytes(self, user_id: str, data: bytes) -> bool:
        """Send binary data to a user's voice WebSocket."""
        ws = self._connections.get(user_id)
        if not ws:
            return False
        try:
            await ws.send_bytes(data)
            return True
        except Exception:
            return False

    async def handle_connection(
        self, websocket: WebSocket, user_id: str, initial_config: dict[str, Any] | None = None
    ) -> None:
        """
        Handle the full voice WebSocket connection lifecycle.

        This is the main coroutine that manages a voice session.
        """
        session_id = await self.connect(websocket, user_id)
        config = PipelineConfig(
            language=initial_config.get("language", "en") if initial_config else "en",
            wake_word_enabled=initial_config.get("wake_word_enabled", True)
            if initial_config
            else True,
            voice_speed=initial_config.get("voice_speed", 1.0)
            if initial_config
            else 1.0,
            voice_pitch=initial_config.get("voice_pitch", 1.0)
            if initial_config
            else 1.0,
        )

        # Send connected acknowledgment
        await self.send_json(user_id, {
            "type": "connected",
            "session_id": session_id,
            "state": PipelineState.IDLE.value,
        })

        # Main message processing loop
        audio_chunks: list[bytes] = []
        is_collecting_audio = False

        try:
            while True:
                # Receive message (JSON or Binary)
                raw = await websocket.receive()

                if "bytes" in raw:
                    # Binary = audio chunk
                    chunk = raw["bytes"]
                    if is_collecting_audio:
                        audio_chunks.append(chunk)
                    # Also feed to audio queue for pipeline
                    queue = self._audio_queues.get(user_id)
                    if queue:
                        await queue.put(chunk)

                elif "text" in raw:
                    # Text = JSON control message
                    try:
                        msg = json.loads(raw["text"])
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type", "")

                    if msg_type == "audio_start":
                        # Start of utterance
                        audio_chunks = []
                        is_collecting_audio = True
                        await self.send_json(user_id, {
                            "type": "state",
                            "state": PipelineState.LISTENING.value,
                        })

                    elif msg_type == "audio_end":
                        # End of utterance — start pipeline processing
                        is_collecting_audio = False

                        # Signal end of audio to the queue
                        queue = self._audio_queues.get(user_id)
                        if queue:
                            await queue.put(None)  # Sentinel

                        # Start pipeline in background
                        if user_id not in self._pipeline_tasks:
                            task = asyncio.create_task(
                                self._run_pipeline(user_id, config)
                            )
                            self._pipeline_tasks[user_id] = task

                    elif msg_type == "interrupt":
                        # User wants to interrupt current response
                        pipeline_manager.interrupt_pipeline(user_id)
                        await self.send_json(user_id, {
                            "type": "interrupt",
                            "state": PipelineState.INTERRUPTED.value,
                        })

                    elif msg_type == "config":
                        # Update session config
                        if "language" in msg:
                            config.language = msg["language"]
                        if "voice_speed" in msg:
                            config.voice_speed = float(msg["voice_speed"])
                        if "voice_pitch" in msg:
                            config.voice_pitch = float(msg["voice_pitch"])
                        if "wake_word_enabled" in msg:
                            config.wake_word_enabled = bool(msg["wake_word_enabled"])
                        await self.send_json(user_id, {
                            "type": "config_ack",
                            "config": {
                                "language": config.language,
                                "voice_speed": config.voice_speed,
                                "voice_pitch": config.voice_pitch,
                                "wake_word_enabled": config.wake_word_enabled,
                            },
                        })

                    elif msg_type == "ping":
                        await self.send_json(user_id, {"type": "pong"})

                    elif msg_type == "close":
                        break

        except WebSocketDisconnect:
            logger.info("Voice WS disconnected: user=%s", user_id)
        except Exception as e:
            logger.error("Voice WS error for user %s: %s", user_id, e)
        finally:
            await self.disconnect(user_id)

    async def _run_pipeline(self, user_id: str, config: PipelineConfig) -> None:
        """
        Run the voice pipeline using audio from the user's queue.

        Reads audio chunks from the queue and feeds them into the pipeline,
        then forwards pipeline events to the WebSocket.
        """
        async def audio_generator():
            """Yield audio chunks from the queue until sentinel."""
            queue = self._audio_queues.get(user_id)
            if not queue:
                return
            while True:
                chunk = await queue.get()
                if chunk is None:  # Sentinel — end of utterance
                    break
                yield chunk

        # Forward pipeline events to WebSocket
        async def event_forwarder(event: PipelineEvent) -> None:
            if event.type == "partial":
                await self.send_json(user_id, {
                    "type": "partial",
                    "text": event.data.get("text", "") if isinstance(event.data, dict) else str(event.data),
                    "confidence": event.data.get("confidence", 0.0) if isinstance(event.data, dict) else 0.0,
                })
            elif event.type == "transcript":
                await self.send_json(user_id, {
                    "type": "transcript",
                    "text": event.data.get("text", "") if isinstance(event.data, dict) else str(event.data),
                    "confidence": event.data.get("confidence", 0.0) if isinstance(event.data, dict) else 0.0,
                    "language": event.data.get("language", "en") if isinstance(event.data, dict) else "en",
                })
            elif event.type == "thinking":
                await self.send_json(user_id, {
                    "type": "thinking",
                    "agent": event.data.get("agent", "router") if isinstance(event.data, dict) else "router",
                })
            elif event.type == "tts_start":
                await self.send_json(user_id, {"type": "tts_start"})
            elif event.type == "tts_chunk" and isinstance(event.data, bytes):
                await self.send_bytes(user_id, event.data)
            elif event.type == "tts_end":
                await self.send_json(user_id, {"type": "tts_end"})
            elif event.type == "state":
                await self.send_json(user_id, {
                    "type": "state",
                    "state": event.data if isinstance(event.data, str) else event.data.get("state", ""),
                })
            elif event.type == "result":
                await self.send_json(user_id, {
                    "type": "result",
                    **event.data,
                })
            elif event.type == "error":
                await self.send_json(user_id, {
                    "type": "error",
                    "message": event.data.get("message", str(event.data)) if isinstance(event.data, dict) else str(event.data),
                })

        try:
            pipeline = pipeline_manager.create_pipeline(
                user_id=user_id,
                config=config,
                event_callback=event_forwarder,
            )

            async for event in pipeline.process_audio_stream(audio_generator()):
                await event_forwarder(event)

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled for user %s", user_id)
        except Exception as e:
            logger.error("Pipeline error for user %s: %s", user_id, e)
            await self.send_json(user_id, {
                "type": "error",
                "message": f"Pipeline error: {e}",
            })
        finally:
            self._pipeline_tasks.pop(user_id, None)


# Singleton
voice_ws_manager = VoiceWebSocketManager()
