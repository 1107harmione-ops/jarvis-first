"""VoiceSessionManager — central state machine orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Callable

from fastapi import WebSocket

from jarvis_voice.config import VoiceConfig
from jarvis_voice.models import ControlMessage, DetectionResult, VoiceEvent
from jarvis_voice.session.state import VoiceState, VoiceSession
from jarvis_voice.stt.base import BaseSTTProvider
from jarvis_voice.tts.base import BaseTTSProvider
from jarvis_voice.wakeword.base import BaseWakeWordDetector
from jarvis_voice.pipeline.audio_processor import AudioProcessor

logger = logging.getLogger("jarvis_voice.session")


class VoiceSessionManager:
    """Central orchestrator for voice sessions.

    Manages the full lifecycle of a voice session:
      IDLE → WAKE_PENDING → LISTENING → STT_PROCESSING → THINKING → SPEAKING → IDLE
    Handles interrupts, timeouts, and cleanup.
    """

    def __init__(
        self,
        config: VoiceConfig,
        stt: BaseSTTProvider,
        tts: BaseTTSProvider,
        wakeword: BaseWakeWordDetector,
    ):
        self.config = config
        self.stt = stt
        self.tts = tts
        self.wakeword = wakeword
        self._sessions: dict[str, VoiceSession] = {}
        self._timeout_task: asyncio.Task | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def active_sessions(self) -> list[VoiceSession]:
        return list(self._sessions.values())

    # ── Session lifecycle ──────────────────────────────────────────────

    async def create_session(self, websocket: WebSocket, user_id: str = "default") -> VoiceSession:
        session_id = str(uuid.uuid4())
        session = VoiceSession(
            session_id=session_id,
            user_id=user_id,
            websocket=websocket,
        )
        self._sessions[session_id] = session
        logger.info("Session created: %s (user=%s)", session_id, user_id)

        if self._timeout_task is None or self._timeout_task.done():
            self._timeout_task = asyncio.create_task(self._check_timeouts_loop())

        return session

    async def destroy_session(self, session: VoiceSession) -> None:
        logger.info("Destroying session: %s", session.session_id)
        session.state = VoiceState.OFFLINE

        # Cancel running tasks
        if session.stt_task and not session.stt_task.done():
            session.stt_task.cancel()
        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()

        # Drain buffer
        while not session.audio_buffer.empty():
            try:
                session.audio_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._sessions.pop(session.session_id, None)

        # Stop the timeout loop if no sessions remain
        if not self._sessions and self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

    # ── Audio entry point ──────────────────────────────────────────────

    async def handle_audio(self, session: VoiceSession, audio_chunk: bytes) -> None:
        """Process an incoming audio chunk based on current session state."""

        if not audio_chunk or len(audio_chunk) < 2:
            return

        session.last_activity = time.time()

        state = session.state

        if state == VoiceState.IDLE or state == VoiceState.WAKE_PENDING:
            await self._process_wake_check(session, audio_chunk)

        elif state == VoiceState.LISTENING:
            await self._process_audio_chunk(session, audio_chunk)

        elif state == VoiceState.SPEAKING:
            await self._check_interrupt(session, audio_chunk)

        elif state in (VoiceState.THINKING, VoiceState.STT_PROCESSING):
            # Buffer audio for potential next command
            pass

        elif state == VoiceState.INTERRUPTED:
            # Transition to LISTENING and process
            await self._transition(session, VoiceState.LISTENING)
            await self._process_audio_chunk(session, audio_chunk)

    # ── Control message handler ────────────────────────────────────────

    async def handle_control(self, session: VoiceSession, data: dict) -> None:
        try:
            msg = ControlMessage(**data)
        except Exception as exc:
            logger.warning("Invalid control message: %s", exc)
            return

        if msg.type == "audio_start":
            # Reset buffer for a new utterance
            session.audio_buffer = asyncio.Queue()
            session.partial_text = ""
            session.final_text = ""
            if session.state == VoiceState.IDLE:
                await self._transition(session, VoiceState.LISTENING)

        elif msg.type == "audio_end":
            if session.state == VoiceState.LISTENING:
                await self._start_stt(session)

        elif msg.type == "interrupt":
            await self._handle_interrupt(session)

        elif msg.type == "config":
            if msg.language is not None and msg.language in self.config.supported_languages:
                session.language = msg.language
                logger.info("Session %s language set to %s", session.session_id, msg.language)
            if msg.voice_speed is not None:
                session.metadata["voice_speed"] = msg.voice_speed
            if msg.sample_rate is not None:
                session.metadata["client_sample_rate"] = msg.sample_rate

    # ── State machine transitions ──────────────────────────────────────

    async def _transition(self, session: VoiceSession, new_state: VoiceState) -> None:
        old_state = session.state
        if old_state == new_state:
            return

        session.state = new_state
        logger.debug(
            "Session %s: %s → %s",
            session.session_id, old_state.value, new_state.value,
        )

        await self._send_event(session, VoiceEvent(
            type="state_change",
            state=new_state.value,
        ))

        # Perform entry actions
        if new_state == VoiceState.STT_PROCESSING:
            await self._start_stt(session)

        elif new_state == VoiceState.ERROR:
            logger.error("Session %s entered ERROR state", session.session_id)

    # ── Wake word processing ──────────────────────────────────────────

    async def _process_wake_check(self, session: VoiceSession, chunk: bytes) -> None:
        result = await self.wakeword.process_chunk(chunk, self.config.sample_rate)
        if result.detected:
            logger.info("Wake word detected (score=%.3f, source=%s)", result.score, result.source)
            session.metadata["wake_score"] = result.score
            session.metadata["wake_source"] = result.source
            await self._transition(session, VoiceState.LISTENING)
        elif session.state == VoiceState.IDLE:
            # Energy-based trigger for push-to-talk style
            rms = AudioProcessor.compute_rms(chunk)
            if rms > self.config.interrupt_energy_threshold * 2:
                await self._transition(session, VoiceState.LISTENING)

    # ── Audio accumulation + silence detection ─────────────────────────

    async def _process_audio_chunk(self, session: VoiceSession, chunk: bytes) -> None:
        # Noise gate
        gated = AudioProcessor.apply_noise_gate(chunk, self.config.interrupt_energy_threshold)
        await session.audio_buffer.put(gated)

        # Check silence timeout by looking at elapsed time since last audio with energy
        rms = AudioProcessor.compute_rms(chunk)
        if rms < self.config.interrupt_energy_threshold:
            # Check if silence has persisted long enough
            elapsed_silence = time.time() - session.last_activity
            if elapsed_silence >= self.config.silence_timeout_sec:
                await self._start_stt(session)
                return

        # Check max duration
        buffer_duration = await self._estimate_buffer_duration(session)
        if buffer_duration >= self.config.max_command_duration:
            await self._start_stt(session)

    async def _estimate_buffer_duration(self, session: VoiceSession) -> float:
        """Estimate the duration of audio in the buffer in seconds."""
        size = session.audio_buffer.qsize()
        if size == 0:
            return 0.0
        avg_chunk_size = self.config.frame_size if self.config.frame_size > 0 else 960
        total_bytes = size * avg_chunk_size
        return total_bytes / (self.config.sample_rate * self.config.bytes_per_sample)

    # ── Interrupt detection ────────────────────────────────────────────

    async def _check_interrupt(self, session: VoiceSession, chunk: bytes) -> bool:
        """Check if incoming audio during SPEAKING should trigger an interrupt."""
        if session.state != VoiceState.SPEAKING:
            return False

        rms = AudioProcessor.compute_rms(chunk)
        if rms > self.config.interrupt_energy_threshold:
            session.total_interrupts += 1
            session.interrupt_event.set()
            await self._handle_interrupt(session)
            return True
        return False

    async def _handle_interrupt(self, session: VoiceSession) -> None:
        """Handle interrupt during SPEAKING."""
        if session.state == VoiceState.SPEAKING:
            # Kill TTS task
            if session.tts_task and not session.tts_task.done():
                session.tts_task.cancel()
                session.tts_task = None
            session.interrupt_event.set()
            session.speaking_done.set()

        await self._transition(session, VoiceState.INTERRUPTED)

        # Immediately move to LISTENING for the next command
        session.audio_buffer = asyncio.Queue()
        session.partial_text = ""
        session.final_text = ""
        await self._transition(session, VoiceState.LISTENING)

    # ── STT pipeline ────────────────────────────────────────────────────

    async def _start_stt(self, session: VoiceSession) -> None:
        """Run streaming STT on accumulated audio buffer."""
        if session.state not in (VoiceState.LISTENING, VoiceState.STT_PROCESSING):
            return

        # Already processing
        if session.stt_task and not session.stt_task.done():
            return

        await self._transition(session, VoiceState.STT_PROCESSING)

        async def partial_cb(text: str, confidence: float) -> None:
            session.partial_text = text
            await self._send_event(session, VoiceEvent(
                type="partial",
                text=text,
                confidence=confidence,
            ))

        session.stt_task = asyncio.create_task(
            self._run_stt(session, partial_cb)
        )

    async def _run_stt(
        self,
        session: VoiceSession,
        partial_cb: Callable[[str, float], None],
    ) -> None:
        """Execute STT transcription in a task."""
        try:
            language = session.language if session.language != "auto" else "auto"
            text, confidence, detected_lang = await self.stt.transcribe_stream(
                audio_queue=session.audio_buffer,
                sample_rate=self.config.sample_rate,
                partial_callback=partial_cb,
                language=language,
            )

            session.final_text = text
            session.total_commands += 1

            if text.strip():
                await self._send_event(session, VoiceEvent(
                    type="transcript",
                    text=text,
                    confidence=confidence,
                ))

                # Update detected language
                if detected_lang and detected_lang != "auto":
                    session.language = detected_lang

                # Move to THINKING
                await self._transition(session, VoiceState.THINKING)

                # Call LLM placeholder
                response = await self._start_llm(session, text)

                # Move to SPEAKING with TTS
                await self._transition(session, VoiceState.SPEAKING)
                await self._start_tts(session, response)
            else:
                # No speech detected
                await self._transition(session, VoiceState.IDLE)

        except asyncio.CancelledError:
            logger.debug("STT task cancelled for session %s", session.session_id)
        except Exception as exc:
            logger.exception("STT error for session %s: %s", session.session_id, exc)
            await self._send_event(session, VoiceEvent(
                type="error",
                message=f"STT error: {exc}",
            ))
            await self._transition(session, VoiceState.ERROR)
        finally:
            session.stt_task = None

    # ── LLM placeholder ────────────────────────────────────────────────

    async def _start_llm(self, session: VoiceSession, text: str) -> str:
        """Placeholder LLM call.

        In production, this would call the main JARVIS LLM endpoint.
        Returns a mock or configurable response.
        """
        await self._send_event(session, VoiceEvent(type="thinking"))

        # Simulate brief thinking time
        await asyncio.sleep(0.3)

        # Mock response
        responses = {
            "en": f"I heard you say: {text}. How can I help you further?",
            "hi": f"आपने कहा: {text}। मैं आपकी और कैसे मदद कर सकता हूँ?",
        }
        return responses.get(session.language, responses["en"])

    # ── TTS pipeline ────────────────────────────────────────────────────

    async def _start_tts(self, session: VoiceSession, text: str) -> None:
        """Run streaming TTS on the response text."""
        if session.state != VoiceState.SPEAKING:
            return

        if session.tts_task and not session.tts_task.done():
            session.tts_task.cancel()

        session.interrupt_event.clear()
        session.speaking_done.clear()

        await self._send_event(session, VoiceEvent(type="tts_start"))

        async def chunk_cb(chunk: bytes) -> None:
            await self._send_audio(session, chunk)

        session.tts_task = asyncio.create_task(
            self._run_tts(session, text, chunk_cb)
        )

    async def _run_tts(
        self,
        session: VoiceSession,
        text: str,
        chunk_cb: Callable[[bytes], None],
    ) -> None:
        """Execute TTS synthesis in a task."""
        try:
            await self.tts.speak_stream(
                text=text,
                language=session.language,
                chunk_callback=chunk_cb,
                interrupt_event=session.interrupt_event,
            )

            await self._send_event(session, VoiceEvent(type="tts_end"))
            session.speaking_done.set()

            if not session.interrupt_event.is_set():
                await self._transition(session, VoiceState.IDLE)

        except asyncio.CancelledError:
            logger.debug("TTS task cancelled for session %s", session.session_id)
            session.speaking_done.set()
        except Exception as exc:
            logger.exception("TTS error for session %s: %s", session.session_id, exc)
            await self._send_event(session, VoiceEvent(
                type="error",
                message=f"TTS error: {exc}",
            ))
            await self._transition(session, VoiceState.ERROR)
        finally:
            session.tts_task = None

    # ── WebSocket send helpers ─────────────────────────────────────────

    async def _send_event(self, session: VoiceSession, event: VoiceEvent) -> None:
        """Send a JSON control event over the WebSocket."""
        try:
            payload = event.model_dump_json(exclude_none=True)
            await session.websocket.send_text(payload)
        except Exception as exc:
            logger.warning("Failed to send event to %s: %s", session.session_id, exc)

    async def _send_audio(self, session: VoiceSession, chunk: bytes) -> None:
        """Send binary PCM16 audio over the WebSocket."""
        try:
            await session.websocket.send_bytes(chunk)
        except Exception as exc:
            logger.warning("Failed to send audio to %s: %s", session.session_id, exc)

    # ── Timeout management ─────────────────────────────────────────────

    async def _check_timeouts_loop(self) -> None:
        """Periodic timeout check loop for all active sessions."""
        try:
            while self._sessions:
                await asyncio.sleep(1.0)
                await self._check_timeouts()
        except asyncio.CancelledError:
            pass

    async def _check_timeouts(self) -> None:
        """Check and enforce timeouts for all sessions."""
        now = time.time()
        expired: list[str] = []

        for session_id, session in list(self._sessions.items()):
            elapsed = now - session.last_activity

            # Voice timeout: no activity for too long
            if elapsed > self.config.voice_timeout_sec and session.state in (
                VoiceState.IDLE,
                VoiceState.LISTENING,
            ):
                await self._transition(session, VoiceState.IDLE)
                continue

            # Max command duration in LISTENING
            if (
                session.state == VoiceState.LISTENING
                and elapsed > self.config.silence_timeout_sec
            ):
                # Has the buffer been silent?
                await self._start_stt(session)
                continue

            # Total session timeout (30 min inactivity)
            if elapsed > 1800:
                logger.info("Session %s expired due to inactivity", session_id)
                expired.append(session_id)

        for sid in expired:
            session = self._sessions.get(sid)
            if session:
                await self.destroy_session(session)
