"""Abstract base class for STT providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Callable

from jarvis_voice.models import STTResult


class BaseSTTProvider(ABC):
    """Abstract base for speech-to-text providers."""

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_queue: asyncio.Queue[bytes],
        sample_rate: int,
        partial_callback: Callable[[str, float], None],
        language: str = "auto",
    ) -> tuple[str, float, str]:
        """Streaming transcription from an audio chunk queue.

        Args:
            audio_queue: Queue of raw PCM16 audio bytes.
            sample_rate: Sample rate of the audio.
            partial_callback: Called with (partial_text, confidence) periodically.
            language: Language hint ("auto", "en", "hi").

        Returns:
            Tuple of (final_text, confidence, detected_language).
        """
        ...

    @abstractmethod
    async def transcribe_file(self, audio_path: str, language: str = "auto") -> STTResult:
        """Full file transcription (non-streaming fallback)."""
        ...

    @abstractmethod
    def _audio_bytes_to_float(self, audio_bytes: bytes) -> "np.ndarray":
        """Convert PCM16 bytes to float32 array normalized to [-1, 1]."""
        ...
