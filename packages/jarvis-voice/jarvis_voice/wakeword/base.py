"""Abstract base class for wake word detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jarvis_voice.models import DetectionResult


class BaseWakeWordDetector(ABC):
    """Abstract base for wake word / voice activity detection."""

    @abstractmethod
    async def process_chunk(self, audio_bytes: bytes, sample_rate: int) -> DetectionResult:
        """Process an audio chunk and return detection result."""
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Reset the detector state (e.g., after a detection)."""
        ...
