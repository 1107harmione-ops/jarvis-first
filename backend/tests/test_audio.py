"""Audio processing tests."""
from __future__ import annotations

import struct
import wave

import pytest

from app.voice.audio import (
    AudioProcessor,
    bytes_to_samples,
    compute_db,
    compute_rms,
    samples_to_bytes,
)


class TestAudioUtils:
    def test_bytes_to_samples_empty(self):
        assert bytes_to_samples(b"") == []

    def test_bytes_to_samples_silence(self):
        data = b"\x00\x00" * 100
        samples = bytes_to_samples(data)
        assert len(samples) == 100
        assert all(s == 0 for s in samples)

    def test_bytes_to_samples_values(self):
        data = struct.pack("<hh", 100, -200)
        samples = bytes_to_samples(data)
        assert samples == [100, -200]

    def test_samples_to_bytes_roundtrip(self):
        original = [100, -200, 0, 32767, -32768]
        data = samples_to_bytes(original)
        restored = bytes_to_samples(data)
        assert restored == original

    def test_compute_rms_silence(self):
        data = b"\x00\x00" * 100
        assert compute_rms(data) == 0.0

    def test_compute_rms_nonzero(self):
        data = struct.pack("<" + "h" * 100, *([100] * 100))
        rms = compute_rms(data)
        assert rms > 0

    def test_compute_rms_empty(self):
        assert compute_rms(b"") == 0.0

    def test_compute_db_silence(self):
        assert compute_db(0) == -100.0

    def test_compute_db_nonzero(self):
        db = compute_db(100)
        assert isinstance(db, float)


class TestAudioProcessor:
    async def test_get_level_silence(self, tmp_path):
        wav_path = tmp_path / "silence.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

        proc = AudioProcessor()
        level = await proc.get_level(wav_path)
        assert level == 0.0

    async def test_get_level_nonzero(self, tmp_path):
        wav_path = tmp_path / "tone.wav"
        samples = struct.pack("<" + "h" * 16000, *([5000] * 16000))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        proc = AudioProcessor()
        level = await proc.get_level(wav_path)
        assert level > 0

    async def test_trim_silence_no_speech(self, tmp_path):
        wav_path = tmp_path / "nospeech.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

        proc = AudioProcessor()
        result = await proc.trim_silence(wav_path, output_path=wav_path, threshold=500.0)
        assert result == wav_path

    async def test_trim_silence_with_speech(self, tmp_path):
        wav_path = tmp_path / "speech.wav"
        silence = b"\x00\x00" * 8000
        speech = struct.pack("<" + "h" * 16000, *([8000] * 16000))
        data = silence + speech + silence
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(data)

        proc = AudioProcessor()
        result = await proc.trim_silence(wav_path, threshold=500.0)
        assert result != wav_path

        with wave.open(str(result), "rb") as wf:
            trimmed_frames = wf.readframes(wf.getnframes())
        # Should be smaller than original (silence trimmed)
        assert len(trimmed_frames) < len(data)
        # Should still contain audio (not empty)
        assert len(trimmed_frames) > 1000

    async def test_trim_silence_output_path(self, tmp_path):
        wav_path = tmp_path / "input.wav"
        out_path = tmp_path / "trimmed.wav"
        speech = struct.pack("<" + "h" * 32000, *([8000] * 32000))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(speech)

        proc = AudioProcessor()
        result = await proc.trim_silence(wav_path, output_path=out_path, threshold=500.0)
        assert result == out_path
        assert out_path.exists()
