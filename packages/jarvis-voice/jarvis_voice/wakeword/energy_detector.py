"""Energy-based voice activity detection as a simple wake word fallback."""

from __future__ import annotations

import logging
import time

import numpy as np

from jarvis_voice.models import DetectionResult
from jarvis_voice.pipeline.audio_processor import AudioProcessor
from jarvis_voice.wakeword.base import BaseWakeWordDetector

logger = logging.getLogger("jarvis_voice.wakeword")


class EnergyWakeWordDetector(BaseWakeWordDetector):
    """Simple energy-based voice activity detector.

    Acts as a fallback when OpenWakeWord is unavailable.
    Detects any voice activity above a configurable RMS threshold.
    """

    def __init__(self, threshold: float = 0.03, cooldown: float = 2.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_detection = 0.0
        self._consecutive_active = 0
        self._min_active_frames = 3  # Require N consecutive frames above threshold

    async def process_chunk(self, audio_bytes: bytes, sample_rate: int) -> DetectionResult:
        """Detect voice activity using RMS energy.

        Args:
            audio_bytes: Raw PCM16 audio bytes.
            sample_rate: Sample rate (unused in energy calculation).

        Returns:
            DetectionResult with detection based on energy threshold.
        """
        if len(audio_bytes) < 2:
            return DetectionResult(detected=False, score=0.0, source="energy")

        rms = AudioProcessor.compute_rms(audio_bytes)

        now = time.time()
        if rms > self.threshold:
            self._consecutive_active += 1
        else:
            self._consecutive_active = max(0, self._consecutive_active - 1)

        if (
            self._consecutive_active >= self._min_active_frames
            and (now - self.last_detection) > self.cooldown
        ):
            self.last_detection = now
            self._consecutive_active = 0
            return DetectionResult(
                detected=True,
                score=float(rms),
                source="energy",
            )

        return DetectionResult(
            detected=False,
            score=float(rms),
            source="energy",
        )

    async def reset(self) -> None:
        """Reset internal state."""
        self.last_detection = 0.0
        self._consecutive_active = 0
