"""Wake word detection — Porcupine with simple energy trigger fallback."""
from __future__ import annotations

from typing import Callable, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

WAKE_WORDS = ["jarvis", "hey jarvis", "ok jarvis", "computer"]
SAMPLE_RATE = 16000
FRAME_LENGTH = 512  # Porcupine frame length


class WakeWordDetector:
    """Detect wake words in audio streams.

    Primary: Picovoice Porcupine (offline, efficient).
    Fallback: Energy-based trigger that detects onset of speech after silence.
    """

    def __init__(self, keywords: Optional[list[str]] = None, sensitivity: float = 0.5):
        self.keywords = keywords or WAKE_WORDS
        self.sensitivity = sensitivity
        self._porcupine = None
        self._init_porcupine()

        # Energy trigger state
        self._silent_frames = 0
        self._last_activity = False

    def _init_porcupine(self) -> None:
        """Initialize Porcupine wake word engine if available."""
        try:
            import pvporcupine
            built_in = [kw for kw in self.keywords if kw.replace("hey ", "").replace("ok ", "") in dir(pvporcupine.Keywords)]
            if built_in:
                self._porcupine = pvporcupine.create(
                    keywords=built_in,
                    sensitivities=[self.sensitivity] * len(built_in),
                )
                logger.info("porcupine_initialized", keywords=built_in)
            else:
                logger.info("porcupine_no_builtin_keywords_falling_back")
        except ImportError:
            logger.info("porcupine_unavailable_falling_back_to_energy_trigger")
        except Exception as e:
            logger.warning("porcupine_init_failed", error=str(e))

    @property
    def available(self) -> bool:
        return self._porcupine is not None

    def process_chunk(self, pcm_frame: bytes) -> bool:
        """Process a PCM audio chunk, return True if wake word detected."""
        if self._porcupine:
            import struct
            fmt = "<" + "h" * (len(pcm_frame) // 2)
            samples = struct.unpack(fmt, pcm_frame)
            result = self._porcupine.process(samples)
            if result >= 0:
                logger.info("wake_word_detected_porcupine", keyword_idx=result)
                return True
        return False

    def detect_energy_trigger(
        self,
        pcm_frame: bytes,
        *,
        energy_threshold: float = 500.0,
        silence_frames_required: int = 20,
    ) -> bool:
        """Simple energy-based trigger: detects speech onset after silence.

        Returns True when speech starts after a silence period.
        """
        if len(pcm_frame) < 2:
            return False

        import struct
        fmt = "<" + "h" * (len(pcm_frame) // 2)
        samples = struct.unpack(fmt, pcm_frame)
        if not samples:
            return False

        rms = sum(s * s for s in samples) / len(samples)
        is_active = rms >= energy_threshold

        triggered = False
        if is_active and not self._last_activity and self._silent_frames >= silence_frames_required:
            triggered = True
            logger.info("wake_word_detected_energy_trigger")

        self._last_activity = is_active
        if not is_active:
            self._silent_frames += 1
        else:
            self._silent_frames = 0

        return triggered

    def close(self) -> None:
        """Release Porcupine resources."""
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None
            logger.info("porcupine_resources_released")


wake_word_detector = WakeWordDetector()
