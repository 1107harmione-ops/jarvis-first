"""Shared test fixtures for jarvis-voice tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket

from jarvis_voice.config import VoiceConfig
from jarvis_voice.models import DetectionResult, STTResult
from jarvis_voice.session.manager import VoiceSessionManager
from jarvis_voice.stt.base import BaseSTTProvider
from jarvis_voice.tts.base import BaseTTSProvider
from jarvis_voice.wakeword.base import BaseWakeWordDetector


@pytest.fixture
def voice_config():
    """Return a default VoiceConfig for testing."""
    return VoiceConfig(
        sample_rate=16000,
        frame_ms=30,
        silence_timeout_sec=1.5,
        max_command_duration=30.0,
        min_command_duration=0.3,
        interrupt_energy_threshold=0.03,
        partial_results_interval=0.3,
        default_language="en",
        supported_languages=["en", "hi"],
    )


class MockSTTProvider(BaseSTTProvider):
    """Mock STT provider for testing."""

    def __init__(self, response_text: str = "hello world", confidence: float = 0.95):
        self.response_text = response_text
        self.response_confidence = confidence
        self.call_count = 0

    async def transcribe_stream(
        self,
        audio_queue: asyncio.Queue[bytes],
        sample_rate: int,
        partial_callback=None,
        language: str = "auto",
    ):
        self.call_count += 1
        # Drain the queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if partial_callback:
            await partial_callback(self.response_text, self.response_confidence)
        return self.response_text, self.response_confidence, language

    async def transcribe_file(self, audio_path: str, language: str = "auto") -> STTResult:
        self.call_count += 1
        return STTResult(
            text=self.response_text,
            confidence=self.response_confidence,
            language=language,
        )

    def _audio_bytes_to_float(self, audio_bytes: bytes):
        import numpy as np
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class MockTTSProvider(BaseTTSProvider):
    """Mock TTS provider for testing."""

    def __init__(self):
        self.chunks = []
        self.call_count = 0

    async def speak_stream(
        self,
        text: str,
        language: str,
        chunk_callback,
        interrupt_event: asyncio.Event,
    ):
        self.call_count += 1
        # Simulate streaming PCM16 chunks
        sample_rate = 22050
        duration = 0.5  # 500ms of audio
        n_samples = int(sample_rate * duration)
        # Generate a simple sine wave as mock audio
        import math
        samples = bytearray()
        for i in range(n_samples):
            val = int(math.sin(2 * math.pi * 440 * i / sample_rate) * 8000)
            samples.extend(val.to_bytes(2, "little", signed=True))
            if interrupt_event.is_set():
                return

        # Send in chunks
        chunk_size = 4096
        for offset in range(0, len(samples), chunk_size):
            if interrupt_event.is_set():
                return
            chunk = bytes(samples[offset:offset + chunk_size])
            self.chunks.append(chunk)
            await chunk_callback(chunk)


class MockWakeWordDetector(BaseWakeWordDetector):
    """Mock wake word detector for testing."""

    def __init__(self, detect: bool = False, score: float = 0.0):
        self.detect = detect
        self.score = score
        self.call_count = 0

    async def process_chunk(self, audio_bytes: bytes, sample_rate: int) -> DetectionResult:
        self.call_count += 1
        return DetectionResult(
            detected=self.detect,
            score=self.score,
            source="energy" if not self.detect else "test",
        )

    async def reset(self):
        pass


@pytest.fixture
def mock_stt():
    return MockSTTProvider()


@pytest.fixture
def mock_tts():
    return MockTTSProvider()


@pytest.fixture
def mock_wakeword():
    return MockWakeWordDetector(detect=False)


@pytest.fixture
def mock_detecting_wakeword():
    return MockWakeWordDetector(detect=True, score=0.8)


class MockWebSocket:
    """Mock FastAPI WebSocket for testing sessions."""

    def __init__(self):
        self.sent_texts: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.client = ("127.0.0.1", 54321)
        self._closed = False

    async def accept(self):
        pass

    async def send_text(self, text: str):
        self.sent_texts.append(text)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)

    async def close(self, code=1000, reason=""):
        self._closed = True

    @property
    def closed(self):
        return self._closed


@pytest.fixture
def mock_websocket():
    return MockWebSocket()


@pytest.fixture
async def session_manager(voice_config, mock_stt, mock_tts, mock_wakeword):
    """Create a VoiceSessionManager with mock providers."""
    manager = VoiceSessionManager(
        config=voice_config,
        stt=mock_stt,
        tts=mock_tts,
        wakeword=mock_wakeword,
    )
    return manager


@pytest.fixture
async def active_session(session_manager, mock_websocket):
    """Create a fully initialised VoiceSession."""
    session = await session_manager.create_session(mock_websocket)
    yield session
    await session_manager.destroy_session(session)


def make_audio_chunk(
    sample_rate: int = 16000,
    duration_ms: int = 30,
    amplitude: float = 0.0,
) -> bytes:
    """Generate a PCM16 audio chunk for testing.

    Args:
        sample_rate: Sample rate in Hz.
        duration_ms: Duration in milliseconds.
        amplitude: Amplitude [0, 1]; 0 = silence.

    Returns:
        PCM16 bytes.
    """
    import math
    import struct

    n_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        # Simple tone or silence
        if amplitude > 0:
            val = int(math.sin(2 * math.pi * 440 * i / sample_rate) * amplitude * 32767)
        else:
            val = 0
        samples.append(val)
    return struct.pack(f"<{n_samples}h", *samples)
