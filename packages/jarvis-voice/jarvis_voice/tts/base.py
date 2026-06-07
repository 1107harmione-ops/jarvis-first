"""Abstract base class for TTS providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Coroutine


class BaseTTSProvider(ABC):
    """Abstract base for text-to-speech providers."""

    @abstractmethod
    async def speak_stream(
        self,
        text: str,
        language: str,
        chunk_callback: Callable[[bytes], Coroutine],
        interrupt_event: asyncio.Event,
    ) -> None:
        """Stream TTS audio in chunks, yielding PCM16 bytes via callback.

        Args:
            text: Text to synthesize.
            language: Language code ("en", "hi").
            chunk_callback: Async callback called with each PCM16 audio chunk.
            interrupt_event: If set, stop synthesis immediately.
        """
        ...
