"""
Voice Service — manages voice sessions, STT, TTS, and voice-based interactions.
Integrates with the WebSocket voice endpoint, Piper TTS, Whisper STT,
and external speech APIs.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncGenerator

import httpx

from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc, serialize_doc
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceSessionState(str, Enum):
    """Voice session state machine."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_STT = "processing_stt"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    OFFLINE = "offline"
    ERROR = "error"


class VoiceSession:
    """Represents a single voice interaction session."""

    def __init__(self, session_id: str, user_id: str, language: str = "en") -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.language = language
        self.state = VoiceSessionState.IDLE
        self.audio_buffer: asyncio.Queue[bytes] = asyncio.Queue()
        self.created_at = datetime.now(timezone.utc)
        self.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=settings.VOICE_SESSION_TIMEOUT_SECONDS
        )
        self.last_activity = datetime.now(timezone.utc)
        self.transcript: str = ""
        self.response_text: str = ""
        self.interrupted = False
        self.wake_word_enabled: bool = True
        self.offline_mode: bool = False
        self.voice_speed: float = 1.0
        self.voice_pitch: float = 1.0
        self.metrics: dict[str, Any] = {}

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "transcript": self.transcript,
            "interrupted": self.interrupted,
            "wake_word_enabled": self.wake_word_enabled,
            "offline_mode": self.offline_mode,
        }


class VoiceService:
    """Manages voice sessions, STT/TTS processing, and voice interactions."""

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    # ── Session Management ─────────────────────────────────────

    def create_session(self, user_id: str, language: str = "en") -> VoiceSession:
        """Create a new voice session."""
        session_id = str(uuid.uuid4())
        session = VoiceSession(session_id=session_id, user_id=user_id, language=language)
        self._sessions[session_id] = session
        logger.info("Voice session created", extra={"session_id": session_id, "user_id": user_id})
        return session

    def get_session(self, session_id: str) -> VoiceSession | None:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session and session.is_expired:
            self.end_session(session_id)
            return None
        if session:
            session.last_activity = datetime.now(timezone.utc)
        return session

    def end_session(self, session_id: str) -> bool:
        """End a voice session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("Voice session ended", extra={"session_id": session_id})
            return True
        return False

    def update_state(self, session_id: str, state: VoiceSessionState) -> bool:
        """Update session state."""
        session = self.get_session(session_id)
        if session:
            session.state = state
            return True
        return False

    # ── STT (Speech-to-Text) ────────────────────────────────────

    async def transcribe(
        self, audio_data: bytes, language: str = "en", mime_type: str = "audio/webm"
    ) -> str:
        """Transcribe audio to text using Whisper (with Piper/DeepSeek fallback)."""
        from backend.services.whisper_stt import whisper_service

        # Try new Whisper STT service first
        if whisper_service.available:
            try:
                result = await whisper_service.transcribe(
                    audio_data=audio_data,
                    language=language if language != "auto" else None,
                )
                if result.text:
                    return result.text
            except Exception as e:
                logger.warning("New Whisper STT failed, falling back: %s", e)

        # Fallback: DeepSeek Whisper API
        client = await self._client()
        try:
            response = await client.post(
                f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
                files={
                    "file": ("audio.webm", io.BytesIO(audio_data), mime_type),
                    "model": (None, settings.STT_MODEL),
                    "language": (None, language),
                },
                timeout=60.0,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("text", "")
        except Exception as exc:
            logger.error("STT failed", extra={"error": str(exc), "language": language})
            raise

    # ── TTS (Text-to-Speech) ────────────────────────────────────

    async def synthesize(
        self, text: str, voice: str | None = None, speed: float = 1.0
    ) -> bytes:
        """Synthesize text to speech audio bytes (Piper preferred, API fallback)."""
        from backend.services.piper_tts import piper_service

        # Try new Piper TTS first
        if piper_service.available:
            try:
                return await piper_service.synthesize(
                    text=text,
                    speed=speed,
                )
            except Exception as e:
                logger.warning("Piper TTS failed, falling back to API: %s", e)

        # Fallback: DeepSeek TTS API
        client = await self._client()
        try:
            response = await client.post(
                f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/audio/speech",
                headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
                json={
                    "model": settings.TTS_MODEL,
                    "input": text,
                    "voice": voice or settings.TTS_VOICE,
                    "response_format": "opus",
                    "speed": speed,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.content
        except Exception as exc:
            logger.error("TTS failed", extra={"error": str(exc)})
            raise

    # ── Voice Interaction ───────────────────────────────────────

    async def process_audio(
        self,
        session_id: str,
        audio_data: bytes,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Process incoming audio: STT → respond → TTS.

        Args:
            session_id: Active voice session.
            audio_data: Raw audio bytes.
            language: Override language detection.

        Returns:
            Dict with transcript, response text, audio bytes, metrics.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")

        session.state = VoiceSessionState.PROCESSING_STT
        lang = language or session.language
        start = time.monotonic()

        # STT
        transcript = await self.transcribe(audio_data, language=lang)
        session.transcript = transcript
        stt_time = (time.monotonic() - start) * 1000

        if not transcript.strip():
            session.state = VoiceSessionState.IDLE
            return {
                "session_id": session_id,
                "transcript": "",
                "response": "",
                "audio_data": b"",
                "duration_ms": round(stt_time, 1),
                "interrupted": False,
            }

        # Check offline mode
        from backend.services.offline_handler import offline_handler

        if session.offline_mode or not offline_handler.is_fully_online:
            session.state = VoiceSessionState.IDLE
            offline_response = await offline_handler.get_offline_response(transcript, lang)
            await offline_handler.queue_command(
                user_id=session.user_id,
                transcript=transcript,
                language=lang,
            )
            return {
                "session_id": session_id,
                "transcript": transcript,
                "response": offline_response,
                "audio_data": b"",
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
                "interrupted": False,
                "offline": True,
            }

        # Process through agent pipeline (import here to avoid circular)
        from backend.agents.router_agent import router_agent

        session.state = VoiceSessionState.THINKING
        agent_response = await router_agent.process(
            user_id=session.user_id,
            message=transcript,
            stream=False,
        )
        response_text = agent_response.get("content", "")
        session.response_text = response_text
        think_time = (time.monotonic() - start) * 1000

        # TTS
        session.state = VoiceSessionState.SPEAKING
        audio_bytes = await self.synthesize(
            response_text,
            speed=session.voice_speed,
        )
        tts_time = (time.monotonic() - start) * 1000

        session.state = VoiceSessionState.IDLE

        total_time = (time.monotonic() - start) * 1000
        session.metrics = {
            "stt_ms": round(stt_time, 1),
            "think_ms": round(think_time - stt_time, 1),
            "tts_ms": round(tts_time - think_time, 1),
            "total_ms": round(total_time, 1),
        }

        # Store voice interaction history
        await self._store_interaction(session, transcript, response_text, total_time)

        # Store in voice memory
        try:
            from backend.memory.voice_memory import voice_memory_service

            await voice_memory_service.store_interaction(
                user_id=session.user_id,
                session_id=session_id,
                transcript=transcript,
                response=response_text,
                confidence=0.9,
                agent=agent_response.get("agent", "router"),
                language=lang,
                metrics=session.metrics,
                interrupted=session.interrupted,
            )
        except Exception as e:
            logger.warning("Failed to store voice memory: %s", e)

        return {
            "session_id": session_id,
            "transcript": transcript,
            "response": response_text,
            "audio_data": audio_bytes,
            "duration_ms": round(total_time, 1),
            "interrupted": session.interrupted,
            "metrics": session.metrics,
        }

    async def handle_interrupt(self, session_id: str) -> bool:
        """Handle a voice interruption."""
        session = self.get_session(session_id)
        if session:
            session.interrupted = True
            session.state = VoiceSessionState.INTERRUPTED
            logger.info("Voice session interrupted", extra={"session_id": session_id})
            return True
        return False

    # ── Cleanup ─────────────────────────────────────────────────

    def start_cleanup_task(self) -> None:
        """Periodically clean up expired sessions."""

        async def _cleanup() -> None:
            while True:
                await asyncio.sleep(60)
                now = datetime.now(timezone.utc)
                expired = [
                    sid for sid, s in self._sessions.items()
                    if now > s.expires_at
                ]
                for sid in expired:
                    self.end_session(sid)
                if expired:
                    logger.debug("Cleaned up expired sessions", extra={"count": len(expired)})

        self._cleanup_task = asyncio.create_task(_cleanup())
        logger.info("Voice session cleanup task started")

    # ── Internal ───────────────────────────────────────────────

    async def _store_interaction(
        self,
        session: VoiceSession,
        transcript: str,
        response: str,
        duration_ms: float,
    ) -> None:
        """Store voice interaction in agent_logs for history."""
        try:
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name="voice_service",
                    session_id=session.session_id,
                    user_id=session.user_id,
                    action="voice_interaction",
                    input_summary=transcript[:200],
                    output_summary=response[:200],
                    duration_ms=duration_ms,
                    status="success",
                    metadata={
                        "language": session.language,
                        "interrupted": session.interrupted,
                        "metrics": session.metrics,
                    },
                )
            )
        except Exception as exc:
            logger.warning("Failed to store voice interaction", extra={"error": str(exc)})

    async def close(self) -> None:
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._http:
            await self._http.aclose()
        self._sessions.clear()


# Global singleton
voice_service = VoiceService()
