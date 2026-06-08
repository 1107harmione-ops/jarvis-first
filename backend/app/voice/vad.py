"""Voice Activity Detection — WebRTC VAD with energy-based fallback."""
from __future__ import annotations

from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

FRAME_DURATION_MS = 30  # standard VAD frame size
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples


class VoiceActivityDetector:
    """Detect speech in audio using WebRTC VAD or energy-based fallback."""

    def __init__(self, mode: int = 1):
        self.mode = mode  # 0-3, higher = more aggressive
        self._vad = None
        self._init_vad()

    def _init_vad(self) -> None:
        """Initialize WebRTC VAD if available."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.mode)
            logger.info("vad_webrtc_initialized", mode=self.mode)
        except ImportError:
            logger.info("vad_webrtc_unavailable_fallback_to_energy")

    @property
    def has_webrtc(self) -> bool:
        return self._vad is not None

    def is_speech(self, frame: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """Check if a 16-bit PCM frame contains speech."""
        if self._vad:
            try:
                return self._vad.is_speech(frame, sample_rate)
            except Exception:
                pass
        return self._energy_is_speech(frame)

    def _energy_is_speech(self, frame: bytes, threshold: float = 300.0) -> bool:
        """Energy-based speech detection fallback."""
        if len(frame) < 2:
            return False
        import struct
        fmt = "<" + "h" * (len(frame) // 2)
        samples = struct.unpack(fmt, frame)
        if not samples:
            return False
        rms = sum(s * s for s in samples) / len(samples)
        return rms >= threshold

    def detect_activity(
        self,
        audio_bytes: bytes,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[tuple[int, int]]:
        """Detect speech segments in PCM audio.

        Returns list of (start_frame, end_frame) of speech regions.
        """
        frame_size = int(sample_rate * FRAME_DURATION_MS / 1000) * 2  # 2 bytes per sample
        if frame_size < 1:
            return []

        frames = []
        for i in range(0, len(audio_bytes), frame_size):
            chunk = audio_bytes[i:i + frame_size]
            if len(chunk) >= frame_size // 2:  # at least half a frame
                frames.append(self.is_speech(chunk, sample_rate))

        # Merge consecutive speech frames into segments
        segments: list[tuple[int, int]] = []
        in_speech = False
        start = 0
        for idx, is_speech in enumerate(frames):
            if is_speech and not in_speech:
                start = idx
                in_speech = True
            elif not is_speech and in_speech:
                if idx - start >= 2:  # at least 2 frames = 60ms
                    segments.append((start, idx))
                in_speech = False
        if in_speech and len(frames) - start >= 2:
            segments.append((start, len(frames)))

        return segments

    def trim_silence(
        self,
        audio_bytes: bytes,
        sample_rate: int = SAMPLE_RATE,
        padding_frames: int = 5,
    ) -> bytes:
        """Trim silence from beginning and end of PCM audio."""
        segments = self.detect_activity(audio_bytes, sample_rate)
        if not segments:
            return b""

        frame_size = int(sample_rate * FRAME_DURATION_MS / 1000) * 2
        start = max(0, segments[0][0] - padding_frames) * frame_size
        end = min(len(audio_bytes), (segments[-1][1] + padding_frames) * frame_size)
        return audio_bytes[start:end]


vad = VoiceActivityDetector()
