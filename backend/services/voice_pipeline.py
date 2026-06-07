"""
Voice Pipeline Orchestrator — Coordinates the full voice processing pipeline.

Audio → VAD → STT → Agent → TTS → Audio

Handles:
- Streaming audio buffering and VAD
- STT transcription management
- Agent routing with context
- TTS streaming generation
- Interrupt handling mid-pipeline
- Metrics collection per stage
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable

from backend.config.settings import settings
from backend.services.piper_tts import piper_service
from backend.services.whisper_stt import (
    PartialTranscript,
    TranscriptionResult,
    VoiceActivityDetector,
    whisper_service,
)

logger = logging.getLogger("jarvis.voice_pipeline")


class PipelineState(str, Enum):
    """States for the voice pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_STT = "processing_stt"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class PipelineConfig:
    """Configuration for a voice pipeline session."""

    language: str = "en"
    wake_word_enabled: bool = True
    voice_speed: float = 1.0
    voice_pitch: float = 1.0
    voice_name: str | None = None
    silence_timeout_ms: int = 800
    max_utterance_ms: int = 30000
    interrupt_enabled: bool = True
    streaming_tts: bool = True
    offline_mode: bool = False


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""

    stt_latency_ms: float = 0.0
    agent_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    audio_duration_ms: float = 0.0
    transcript_length: int = 0
    response_length: int = 0
    interrupted: bool = False
    stt_confidence: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stt_latency_ms": round(self.stt_latency_ms, 1),
            "agent_latency_ms": round(self.agent_latency_ms, 1),
            "tts_latency_ms": round(self.tts_latency_ms, 1),
            "total_pipeline_ms": round(self.total_pipeline_ms, 1),
            "audio_duration_ms": round(self.audio_duration_ms, 1),
            "transcript_length": self.transcript_length,
            "response_length": self.response_length,
            "interrupted": self.interrupted,
            "stt_confidence": round(self.stt_confidence, 3),
        }


class PipelineEvent:
    """Events emitted by the pipeline for the WebSocket to forward."""

    def __init__(self, event_type: str, data: Any = None) -> None:
        self.type = event_type
        self.data = data


@dataclass
class PipelineResult:
    """Result of a single voice pipeline run."""

    transcript: str
    response: str
    audio_chunks: list[bytes] = field(default_factory=list)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    conversation_id: str | None = None
    agent: str = "router"
    state: PipelineState = PipelineState.IDLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "response": self.response,
            "audio_size": sum(len(c) for c in self.audio_chunks),
            "metrics": self.metrics.to_dict(),
            "conversation_id": self.conversation_id,
            "agent": self.agent,
        }


class VoicePipeline:
    """
    Orchestrates the voice processing pipeline.

    Usage:
        pipeline = VoicePipeline(user_id, conversation_id)
        async for event in pipeline.process_audio_stream(audio_generator):
            # event.type: "partial", "transcript", "thinking",
            #             "tts_chunk", "tts_start", "tts_end", "result", "error"
            ...
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: str | None = None,
        config: PipelineConfig | None = None,
        event_callback: Callable[[PipelineEvent], Any] | None = None,
    ) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.config = config or PipelineConfig()
        self.event_callback = event_callback

        self.state: PipelineState = PipelineState.IDLE
        self._interrupted = asyncio.Event()
        self._interrupted.clear()
        self._metrics = PipelineMetrics()
        self._pipeline_start: float = 0.0
        self._vad: VoiceActivityDetector | None = None
        self._audio_buffer = bytearray()

    async def process_audio_stream(
        self,
        audio_generator: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Process a streaming audio input through the full voice pipeline.

        Args:
            audio_generator: Async generator yielding PCM16 audio chunks

        Yields:
            PipelineEvent objects:
                - "partial": Interim transcript
                - "transcript": Final transcript
                - "thinking": Agent is processing
                - "tts_start": TTS generation started
                - "tts_chunk": PCM16 audio chunk
                - "tts_end": TTS generation ended
                - "result": Final PipelineResult
                - "error": Error occurred
                - "state": State change
        """
        self._pipeline_start = time.monotonic()
        self.state = PipelineState.LISTENING
        yield PipelineEvent("state", self.state.value)

        # Initialize services if needed
        if not whisper_service.available:
            await whisper_service.initialize()
        if not piper_service.available:
            await piper_service.initialize()

        # Phase 1: Collect audio with VAD
        self._vad = whisper_service.create_vad()
        utterance_audio: bytes = b""
        self._audio_buffer = bytearray()

        yield PipelineEvent("state", PipelineState.LISTENING)

        try:
            async for chunk in audio_generator:
                if self._interrupted.is_set():
                    yield PipelineEvent(
                        "error", {"message": "Pipeline interrupted during listening"}
                    )
                    return

                self._audio_buffer.extend(chunk)
                vad_result = self._vad.process_chunk(chunk)

                if vad_result["speech_ended"]:
                    utterance_audio = vad_result["utterance_buffer"]
                    break

            # If no utterance detected via VAD, use the full buffer
            if not utterance_audio and len(self._audio_buffer) > 0:
                utterance_audio = bytes(self._audio_buffer)

        except asyncio.CancelledError:
            yield PipelineEvent("error", {"message": "Pipeline cancelled"})
            return

        if not utterance_audio or len(utterance_audio) < 64:
            yield PipelineEvent(
                "error", {"message": "No speech detected"}
            )
            return

        # Phase 2: STT
        self.state = PipelineState.PROCESSING_STT
        yield PipelineEvent("state", self.state.value)

        stt_start = time.monotonic()
        transcript: TranscriptionResult | None = None

        try:
            transcript = await whisper_service.transcribe(
                audio_data=utterance_audio,
                language=self.config.language if self.config.language != "auto" else None,
            )
        except Exception as e:
            logger.error("STT failed: %s", e)
            yield PipelineEvent("error", {"message": f"STT failed: {e}"})
            self._metrics.error = str(e)
            return

        self._metrics.stt_latency_ms = (time.monotonic() - stt_start) * 1000
        self._metrics.stt_confidence = transcript.confidence
        self._metrics.transcript_length = len(transcript.text)
        self._metrics.audio_duration_ms = (
            len(utterance_audio) / 2 / 16000 * 1000
        )

        if not transcript.text.strip():
            yield PipelineEvent("error", {"message": "No speech recognized"})
            return

        yield PipelineEvent("transcript", transcript.to_dict())

        # Check interrupt
        if self._interrupted.is_set():
            self.state = PipelineState.INTERRUPTED
            self._metrics.interrupted = True
            yield PipelineEvent("state", self.state.value)
            return

        # Phase 3: Agent / Thinking
        self.state = PipelineState.THINKING
        self._metrics.agent = "router"
        yield PipelineEvent("state", self.state.value)
        yield PipelineEvent("thinking", {"agent": "router"})

        agent_start = time.monotonic()
        response_text = ""
        agent_name = "router"

        try:
            # Import router_agent locally to avoid circular imports
            from backend.agents.router_agent import RouterAgent

            router_agent = RouterAgent()
            agent_result = await router_agent.process(
                user_id=self.user_id,
                message=transcript.text,
                conversation_id=self.conversation_id,
                language=self.config.language,
            )

            response_text = agent_result.get("content", "")
            agent_name = agent_result.get("agent", "router")
            self.conversation_id = agent_result.get(
                "conversation_id", self.conversation_id
            )

        except Exception as e:
            logger.error("Agent processing failed: %s", e)
            response_text = f"I encountered an error: {e}"
            self._metrics.error = str(e)

        self._metrics.agent_latency_ms = (time.monotonic() - agent_start) * 1000
        self._metrics.response_length = len(response_text)

        # Check interrupt
        if self._interrupted.is_set():
            self.state = PipelineState.INTERRUPTED
            self._metrics.interrupted = True
            yield PipelineEvent("state", self.state.value)
            return

        # Phase 4: TTS
        self.state = PipelineState.SPEAKING
        yield PipelineEvent("state", self.state.value)
        yield PipelineEvent("tts_start", {})

        tts_start = time.monotonic()
        audio_chunks: list[bytes] = []

        try:
            async for audio_chunk in piper_service.stream_synthesize(
                text=response_text,
                language=self.config.language,
                voice=self.config.voice_name,
                speed=self.config.voice_speed,
                pitch=self.config.voice_pitch,
            ):
                if self._interrupted.is_set():
                    self.state = PipelineState.INTERRUPTED
                    self._metrics.interrupted = True
                    yield PipelineEvent("state", self.state.value)
                    break

                audio_chunks.append(audio_chunk)
                yield PipelineEvent("tts_chunk", audio_chunk)

        except Exception as e:
            logger.error("TTS failed: %s", e)
            yield PipelineEvent("error", {"message": f"TTS failed: {e}"})

        self._metrics.tts_latency_ms = (time.monotonic() - tts_start) * 1000

        # Finalize
        self._metrics.total_pipeline_ms = (
            time.monotonic() - self._pipeline_start
        ) * 1000

        result = PipelineResult(
            transcript=transcript.text,
            response=response_text,
            audio_chunks=audio_chunks,
            metrics=self._metrics,
            conversation_id=self.conversation_id,
            agent=agent_name,
            state=self.state,
        )

        yield PipelineEvent("tts_end", {"chunks": len(audio_chunks)})
        yield PipelineEvent("result", result.to_dict())

        # Reset state
        self.state = PipelineState.IDLE
        yield PipelineEvent("state", self.state.value)

    def interrupt(self) -> None:
        """Interrupt the current pipeline execution."""
        self._interrupted.set()
        self.state = PipelineState.INTERRUPTED
        logger.info("Voice pipeline interrupted for user %s", self.user_id)

    def reset_interrupt(self) -> None:
        """Clear the interrupt flag."""
        self._interrupted.clear()

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted.is_set()


class PipelineManager:
    """
    Manages multiple voice pipelines (one per user/session).

    Provides a central registry for active pipelines and handles
    lifecycle management.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, VoicePipeline] = {}

    def create_pipeline(
        self,
        user_id: str,
        conversation_id: str | None = None,
        config: PipelineConfig | None = None,
        event_callback: Callable[[PipelineEvent], Any] | None = None,
    ) -> VoicePipeline:
        """Create a new pipeline for a user session."""
        pipeline = VoicePipeline(
            user_id=user_id,
            conversation_id=conversation_id,
            config=config,
            event_callback=event_callback,
        )
        self._pipelines[pipeline.conversation_id or user_id] = pipeline
        return pipeline

    def get_pipeline(self, session_key: str) -> VoicePipeline | None:
        """Get an active pipeline by session key."""
        return self._pipelines.get(session_key)

    def remove_pipeline(self, session_key: str) -> None:
        """Remove a pipeline from the manager."""
        self._pipelines.pop(session_key, None)

    def interrupt_pipeline(self, session_key: str) -> bool:
        """Interrupt a pipeline by session key."""
        pipeline = self._pipelines.get(session_key)
        if pipeline:
            pipeline.interrupt()
            return True
        return False

    @property
    def active_count(self) -> int:
        return len(self._pipelines)


# Global pipeline manager
pipeline_manager = PipelineManager()
