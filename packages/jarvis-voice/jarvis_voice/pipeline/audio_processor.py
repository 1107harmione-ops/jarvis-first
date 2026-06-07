"""Audio processing utilities: VAD, noise gate, resampling, RMS."""

from __future__ import annotations

import math
import struct

import numpy as np


class AudioProcessor:
    """Static methods for common audio processing operations."""

    @staticmethod
    def compute_rms(audio_bytes: bytes) -> float:
        """Compute RMS energy from PCM16 bytes, normalised to [0, 1].

        Args:
            audio_bytes: Raw PCM16 mono audio bytes.

        Returns:
            RMS energy value in [0, 1].
        """
        if len(audio_bytes) < 2:
            return 0.0

        count = len(audio_bytes) // 2
        # Use struct for small buffers, numpy for large ones
        if count > 256:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
        else:
            fmt = f"<{count}h"
            try:
                samples = struct.unpack(fmt, audio_bytes[: count * 2])
            except struct.error:
                return 0.0

        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / count) / 32768.0
        return min(rms, 1.0)

    @staticmethod
    def resample_pcm(audio_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
        """Simple linear resampling of PCM16 audio.

        Args:
            audio_bytes: Input PCM16 bytes.
            from_rate: Source sample rate.
            to_rate: Target sample rate.

        Returns:
            Resampled PCM16 bytes.
        """
        if from_rate == to_rate or len(audio_bytes) < 2:
            return audio_bytes

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        n_samples = len(samples)
        n_output = int(n_samples * to_rate / from_rate)

        if n_output == 0:
            return b""

        # Linear interpolation
        indices = np.arange(n_output) * (n_samples - 1) / max(n_output - 1, 1)
        low = np.floor(indices).astype(int)
        high = np.ceil(indices).astype(int)
        frac = indices - low

        # Clip bounds
        low = np.clip(low, 0, n_samples - 1)
        high = np.clip(high, 0, n_samples - 1)

        resampled = samples[low] * (1.0 - frac) + samples[high] * frac
        resampled = np.clip(resampled, -32768.0, 32767.0).astype(np.int16)
        return resampled.tobytes()

    @staticmethod
    def apply_noise_gate(audio_bytes: bytes, threshold: float) -> bytes:
        """Zero out samples below the RMS threshold (noise gate).

        Args:
            audio_bytes: Raw PCM16 audio bytes.
            threshold: RMS threshold in [0, 1].

        Returns:
            Gated PCM16 audio bytes (samples below threshold zeroed).
        """
        if len(audio_bytes) < 2 or threshold <= 0.0:
            return audio_bytes

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        abs_samples = np.abs(samples) / 32768.0
        mask = abs_samples >= threshold
        samples[~mask] = 0.0
        return samples.astype(np.int16).tobytes()

    @staticmethod
    def compute_speech_ratio(audio_bytes: bytes, energy_threshold: float) -> float:
        """Ratio of audio frames (individual samples) above energy threshold.

        Args:
            audio_bytes: Raw PCM16 audio bytes.
            energy_threshold: RMS threshold in [0, 1].

        Returns:
            Fraction of samples that exceed the threshold.
        """
        if len(audio_bytes) < 2:
            return 0.0

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        abs_samples = np.abs(samples) / 32768.0
        above = np.sum(abs_samples >= energy_threshold)
        return float(above) / len(samples) if len(samples) > 0 else 0.0

    @staticmethod
    def trim_silence(audio_bytes: bytes, sample_rate: int, threshold: float) -> bytes:
        """Trim leading and trailing silence from PCM16 audio.

        Args:
            audio_bytes: Raw PCM16 audio bytes.
            sample_rate: Sample rate in Hz.
            threshold: RMS threshold below which is considered silence.

        Returns:
            Trimmed PCM16 audio bytes.
        """
        if len(audio_bytes) < 2:
            return audio_bytes

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        abs_samples = np.abs(samples) / 32768.0

        # Find first and last non-silent indices
        above = abs_samples >= threshold
        if not np.any(above):
            return b""

        first = int(np.argmax(above))
        last = int(len(samples) - 1 - np.argmax(above[::-1]))

        # Add small padding (50ms)
        pad_samples = int(0.05 * sample_rate)
        first = max(0, first - pad_samples)
        last = min(len(samples) - 1, last + pad_samples)

        trimmed = samples[first : last + 1].astype(np.int16)
        return trimmed.tobytes()
