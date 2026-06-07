"""Pydantic models and message types for the voice protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ControlMessage(BaseModel):
    """Control message received from the client over WebSocket text frames."""

    type: Literal["audio_start", "audio_end", "interrupt", "config"]
    format: str | None = None
    sample_rate: int | None = None
    language: str | None = None
    voice_speed: float | None = None


class VoiceEvent(BaseModel):
    """Event sent to the client over WebSocket text frames."""

    type: Literal[
        "state_change", "partial", "transcript",
        "tts_start", "tts_end", "thinking", "error",
    ]
    state: str | None = None
    text: str | None = None
    confidence: float | None = None
    message: str | None = None


class DetectionResult(BaseModel):
    """Result of wake word or voice activity detection."""

    detected: bool
    score: float = 0.0
    source: str = "energy"


class STTResult(BaseModel):
    """Result of speech-to-text transcription."""

    text: str
    confidence: float
    language: str
    segments: list[dict] = []


class VoiceCommandLog(BaseModel):
    """Log entry for a single voice command."""

    user_id: str
    text: str
    language: str
    duration_ms: int
    success: bool
    command_type: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VoiceSessionLog(BaseModel):
    """Log entry for a voice session."""

    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    commands: int = 0
    interrupts: int = 0
    language: str = "en"
