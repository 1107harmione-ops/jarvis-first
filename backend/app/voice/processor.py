from __future__ import annotations

import io
import struct
import wave
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


def convert_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[bytes]:
    try:
        import soundfile as sf
        import numpy as np
        data, orig_rate = sf.read(io.BytesIO(audio_bytes))
        if orig_rate != sample_rate:
            import numpy as np
            ratio = sample_rate / orig_rate
            new_len = int(len(data) * ratio)
            data = np.interp(
                np.arange(new_len) / ratio,
                np.arange(len(data)),
                data,
            )
        buf = io.BytesIO()
        sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except ImportError:
        return _basic_wav_convert(audio_bytes, sample_rate)
    except Exception as e:
        logger.warning("WAV conversion failed", error=str(e))
        return _basic_wav_convert(audio_bytes, sample_rate)


def _basic_wav_convert(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[bytes]:
    try:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        return buf.getvalue()
    except Exception as e:
        logger.error("Basic WAV conversion failed", error=str(e))
        return None


def get_audio_duration(wav_bytes: bytes, sample_rate: int = 16000) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate > 0 else 0.0
    except Exception:
        return len(wav_bytes) / (sample_rate * 2)


def slice_audio(wav_bytes: bytes, start_sec: float, end_sec: float, sample_rate: int = 16000) -> Optional[bytes]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())

        sample_width = params.sampwidth
        channels = params.nchannels
        frame_size = sample_width * channels

        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)

        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size

        sliced = frames[start_byte:end_byte]

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(sliced)
        return buf.getvalue()
    except Exception as e:
        logger.error("Audio slicing failed", error=str(e))
        return None


def get_audio_level(wav_bytes: bytes) -> float:
    try:
        import numpy as np
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        max_val = 32768.0
        return min(rms / max_val, 1.0)
    except Exception:
        return 0.0
