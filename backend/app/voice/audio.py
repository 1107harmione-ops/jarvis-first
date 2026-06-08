"""Audio processing utilities — conversion, level detection, slicing."""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import wave
from math import sqrt
from pathlib import Path
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

SAMPLE_WIDTH_16BIT = 2
MONO = 1
TARGET_SAMPLE_RATE = 16000


def bytes_to_samples(data: bytes, sample_width: int = SAMPLE_WIDTH_16BIT) -> list[int]:
    """Convert raw PCM bytes to integer samples."""
    fmt = "<" + "h" * (len(data) // sample_width)
    return list(struct.unpack(fmt, data))


def samples_to_bytes(samples: list[int], sample_width: int = SAMPLE_WIDTH_16BIT) -> bytes:
    """Convert integer samples to raw PCM bytes."""
    fmt = "<" + "h" * len(samples)
    return struct.pack(fmt, *samples)


def compute_rms(data: bytes, sample_width: int = SAMPLE_WIDTH_16BIT) -> float:
    """Compute RMS energy level from PCM audio data."""
    if len(data) < sample_width:
        return 0.0
    samples = bytes_to_samples(data, sample_width)
    if not samples:
        return 0.0
    sum_squares = sum(s * s for s in samples)
    return sqrt(sum_squares / len(samples))


def compute_db(rms: float) -> float:
    """Convert RMS to decibel level."""
    if rms == 0:
        return -100.0
    return 20.0 * sqrt(rms / 32768.0)


class AudioProcessor:
    """Audio file and stream processing."""

    async def convert_to_wav(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        target_sr: int = TARGET_SAMPLE_RATE,
    ) -> Path:
        """Convert audio file to WAV mono 16-bit using ffmpeg."""
        input_path = Path(input_path)
        if output_path is None:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output_path = Path(path)
        else:
            output_path = Path(output_path)

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", str(MONO),
            "-ar", str(target_sr),
            "-sample_fmt", "s16",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("audio_convert_failed", input=str(input_path), error=str(e))
            raise RuntimeError(f"Audio conversion failed: {e}") from e

        logger.info("audio_converted", input=str(input_path), output=str(output_path))
        return output_path

    async def get_level(self, audio_path: str | Path) -> float:
        """Get RMS audio level from a WAV file."""
        with wave.open(str(audio_path), "rb") as wf:
            data = wf.readframes(wf.getnframes())
        return compute_rms(data)

    async def slice_audio(
        self,
        audio_path: str | Path,
        start_ms: int,
        end_ms: int,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Slice a segment from a WAV file."""
        if output_path is None:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output_path = Path(path)
        else:
            output_path = Path(output_path)

        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start_ms / 1000.0),
            "-to", str(end_ms / 1000.0),
            "-c", "copy",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Audio slicing failed: {e}") from e

        return output_path

    async def trim_silence(
        self,
        audio_path: str | Path,
        output_path: Optional[str | Path] = None,
        threshold: float = 500.0,
        padding_ms: int = 200,
    ) -> Path:
        """Trim silence from beginning and end of a WAV file."""
        with wave.open(str(audio_path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        samples = bytes_to_samples(frames)
        chunk_size = int(sr * 0.03)  # 30ms chunks
        if chunk_size < 1:
            chunk_size = 1

        # Find first speech chunk
        start_idx = 0
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            rms = sqrt(sum(s * s for s in chunk) / len(chunk)) if chunk else 0
            if rms >= threshold:
                start_idx = max(0, i - int(sr * padding_ms / 1000))
                break

        # Find last speech chunk
        end_idx = len(samples)
        for i in range(len(samples) - chunk_size, 0, -chunk_size):
            chunk = samples[i:i + chunk_size]
            rms = sqrt(sum(s * s for s in chunk) / len(chunk)) if chunk else 0
            if rms >= threshold:
                end_idx = min(len(samples), i + chunk_size + int(sr * padding_ms / 1000))
                break

        if start_idx >= end_idx:
            return Path(audio_path)

        trimmed = samples[start_idx:end_idx]
        trimmed_bytes = samples_to_bytes(trimmed)

        if output_path is None:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output_path = Path(path)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(MONO)
            wf.setsampwidth(SAMPLE_WIDTH_16BIT)
            wf.setframerate(sr)
            wf.writeframes(trimmed_bytes)

        return output_path


audio_processor = AudioProcessor()
