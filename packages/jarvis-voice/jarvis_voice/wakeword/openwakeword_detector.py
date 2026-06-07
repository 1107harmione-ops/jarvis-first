"""OpenWakeWord neural wake word detection."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from jarvis_voice.models import DetectionResult
from jarvis_voice.wakeword.base import BaseWakeWordDetector

logger = logging.getLogger("jarvis_voice.wakeword")


class OpenWakeWordDetector(BaseWakeWordDetector):
    """Neural wake word detection using OpenWakeWord.

    Uses the pre-trained OpenWakeWord model with configurable sensitivity.
    Falls back gracefully if the model is not available.
    """

    def __init__(
        self,
        wake_word: str = "hey jarvis",
        sensitivity: float = 0.5,
    ):
        self.wake_word = wake_word.lower().strip()
        self.sensitivity = sensitivity
        # Threshold: lower sensitivity → higher threshold (fewer false positives)
        # Maps sensitivity [0, 1] to threshold [0.8, 0.2]
        self.threshold = 0.8 - sensitivity * 0.6
        self.cooldown = 2.0
        self.last_detection = 0.0
        self._model = None
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to initialise the OpenWakeWord model."""
        try:
            import openwakeword  # noqa: F401 — ensure package is importable
            from openwakeword import Model as OWWModel  # noqa: N813

            # Determine wake word model name from the phrase
            # OpenWakeWord ships with "hey_jarvis" as a built-in
            words = self.wake_word.split()
            if len(words) >= 2:
                model_name = f"hey_{words[-1]}"
            else:
                model_name = f"hey_{words[0]}"

            self._model = OWWModel(wakeword_models=[model_name])
            self._available = True
            logger.info(
                "OpenWakeWord loaded: wake_word=%s model=%s threshold=%.3f",
                self.wake_word, model_name, self.threshold,
            )
        except ImportError:
            logger.warning(
                "openwakeword not installed; wake word detection unavailable"
            )
            self._available = False
        except Exception as exc:
            logger.warning(
                "Failed to load OpenWakeWord model: %s", exc
            )
            self._available = False

    async def process_chunk(self, audio_bytes: bytes, sample_rate: int) -> DetectionResult:
        """Process an audio chunk for wake word presence.

        Args:
            audio_bytes: Raw PCM16 audio bytes.
            sample_rate: Sample rate of the audio (must be 16kHz for OWW).

        Returns:
            DetectionResult with detection status and score.
        """
        if not self._available or self._model is None:
            return DetectionResult(detected=False, score=0.0, source="openwakeword")

        if len(audio_bytes) < 2:
            return DetectionResult(detected=False, score=0.0, source="openwakeword")

        try:
            # Convert PCM16 int16 → float32 [-1, 1]
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            pcm /= 32768.0
            np.clip(pcm, -1.0, 1.0, out=pcm)

            # OpenWakeWord expects predictions per chunk
            prediction = await asyncio_to_thread(self._model.predict, pcm)

            # Get the max score across all models
            if hasattr(prediction, "values"):
                scores = list(prediction.values())
                score = max(scores) if scores else 0.0
            elif isinstance(prediction, dict):
                score = max(prediction.values()) if prediction else 0.0
            else:
                score = float(prediction) if prediction else 0.0

            now = time.time()
            if (
                score > self.threshold
                and (now - self.last_detection) > self.cooldown
            ):
                self.last_detection = now
                logger.debug("Wake word detected! score=%.3f", score)
                return DetectionResult(
                    detected=True,
                    score=float(score),
                    source="openwakeword",
                )

            return DetectionResult(
                detected=False,
                score=float(score),
                source="openwakeword",
            )

        except Exception as exc:
            logger.debug("OpenWakeWord process error: %s", exc)
            return DetectionResult(detected=False, score=0.0, source="openwakeword")

    async def reset(self) -> None:
        """Reset detector state."""
        if self._model is not None and hasattr(self._model, "reset"):
            await asyncio_to_thread(self._model.reset)
        self.last_detection = 0.0


def asyncio_to_thread(func, *args, **kwargs):
    """Run a synchronous function in a thread to avoid blocking."""
    import asyncio as _asyncio
    return _asyncio.to_thread(func, *args, **kwargs)
