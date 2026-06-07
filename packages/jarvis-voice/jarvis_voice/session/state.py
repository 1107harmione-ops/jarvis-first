"""Voice state enum and session dataclass."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from fastapi import WebSocket


class VoiceState(str, Enum):
    """All possible states in the voice session state machine."""

    IDLE = "idle"
    WAKE_PENDING = "wake_pending"
    LISTENING = "listening"
    STT_PROCESSING = "stt_processing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class VoiceSession:
    """Represents a single voice conversation session over a WebSocket."""

    session_id: str
    user_id: str
    websocket: WebSocket
    state: VoiceState = VoiceState.IDLE
    language: str = "en"
    audio_buffer: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)
    partial_text: str = ""
    final_text: str = ""
    tts_task: asyncio.Task | None = None
    stt_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    speaking_done: asyncio.Event = field(default_factory=asyncio.Event)
    metadata: dict = field(default_factory=dict)
    total_commands: int = 0
    total_interrupts: int = 0
