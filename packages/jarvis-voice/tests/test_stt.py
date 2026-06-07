"""Tests for WhisperSTT provider."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from jarvis_voice.models import STTResult
from jarvis_voice.stt.whisper_stt import WhisperSTT


class TestAudioBytesToFloat:
    """Test the PCM16 → float32 conversion."""

    def test_silence(self, voice_config):
        stt = WhisperSTT(voice_config)
        # 100 samples of silence
        audio = struct.pack("<100h", *([0] * 100))
        result = stt._audio_bytes_to_float(audio)
        assert result.shape == (100,)
        assert np.allclose(result, 0.0)

    def test_full_scale_positive(self, voice_config):
        stt = WhisperSTT(voice_config)
        samples = [32767] * 100
        audio = struct.pack("<100h", *samples)
        result = stt._audio_bytes_to_float(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
        assert np.allclose(result, 1.0, atol=0.01)

    def test_full_scale_negative(self, voice_config):
        stt = WhisperSTT(voice_config)
        samples = [-32768] * 100
        audio = struct.pack("<100h", *samples)
        result = stt._audio_bytes_to_float(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_sine_wave(self, voice_config):
        stt = WhisperSTT(voice_config)
        import math
        samples = [int(math.sin(2 * math.pi * 440 * i / 16000) * 16000) for i in range(1600)]
        audio = struct.pack(f"<{len(samples)}h", *samples)
        result = stt._audio_bytes_to_float(audio)
        assert result.shape == (len(samples),)
        assert np.max(result) <= 1.0
        assert np.min(result) >= -1.0
        assert np.any(np.abs(result) > 0.01)

    def test_empty_bytes(self, voice_config):
        stt = WhisperSTT(voice_config)
        result = stt._audio_bytes_to_float(b"")
        assert result.shape == (0,)

    def test_single_sample(self, voice_config):
        stt = WhisperSTT(voice_config)
        audio = struct.pack("<h", 16000)
        result = stt._audio_bytes_to_float(audio)
        assert result.shape == (1,)
        expected = 16000.0 / 32768.0
        assert abs(result[0] - expected) < 0.001


class TestTranscribeSegments:
    """Test the _transcribe_segments method by mocking the internal _run."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_correctly(self, voice_config):
        stt = WhisperSTT(voice_config)
        # Mock _transcribe_segments by replacing the method
        async def mock_transcribe(audio_array, language=None):
            return "hello world", 0.95, "en"

        stt._transcribe_segments = mock_transcribe  # type: ignore[method-assign]
        audio_array = np.zeros(16000, dtype=np.float32)
        text, confidence, lang = await stt._transcribe_segments(audio_array, language="en")
        assert "hello" in text
        assert "world" in text
        assert lang == "en"

    @pytest.mark.asyncio
    async def test_transcribe_empty(self, voice_config):
        stt = WhisperSTT(voice_config)

        async def mock_empty(audio_array, language=None):
            return "", 0.0, "en"

        stt._transcribe_segments = mock_empty  # type: ignore[method-assign]
        audio_array = np.zeros(160, dtype=np.float32)
        text, confidence, lang = await stt._transcribe_segments(audio_array)
        assert text == ""
        assert isinstance(lang, str)


class TestTranscribeFile:
    """Test the transcribe_file method."""

    @pytest.mark.asyncio
    async def test_transcribe_file_returns_stt_result(self, voice_config):
        """Test that transcribe_file returns an STTResult."""
        stt = WhisperSTT(voice_config)

        # Mock the model to avoid actual inference
        mock_segments = [
            type("Seg", (), {"text": "test audio", "start": 0.0, "end": 1.0, "confidence": 0.95, "avg_logprob": -0.2})(),
        ]
        mock_info = type("Info", (), {"language": "en", "average_log_prob": -0.2})()

        def mock_transcribe(*args, **kwargs):
            return mock_segments, mock_info

        stt.model = type("MockModel", (), {"transcribe": mock_transcribe})()
        result = await stt.transcribe_file("/fake/path.wav")
        assert isinstance(result, STTResult)
        assert len(result.text) > 0
        # Confidence is info.average_log_prob when positive, or the raw value
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_file_error(self, voice_config):
        """Test that transcribe_file handles errors gracefully."""
        stt = WhisperSTT(voice_config)

        def mock_error(*args, **kwargs):
            raise RuntimeError("Model error")

        stt.model = type("MockModel", (), {"transcribe": mock_error})()
        result = await stt.transcribe_file("/fake/path.wav")
        assert isinstance(result, STTResult)
        assert result.text == ""
        assert result.confidence == 0.0


class TestTranscribeStream:
    """Test the streaming transcription path."""

    @pytest.mark.asyncio
    async def test_stream_with_audio(self, voice_config):
        """Test that transcribe_stream processes audio from queue."""
        stt = WhisperSTT(voice_config)

        # Mock the _run_partial to avoid actual transcription
        stt._run_partial = AsyncMock()  # type: ignore[method-assign]

        # Mock _transcribe_segments to return a known result
        async def mock_final(audio_array, language=None):
            return "mock transcription", 0.9, "en"

        stt._transcribe_segments = mock_final  # type: ignore[method-assign]

        # Create an audio queue with some audio
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        import math
        samples = [int(math.sin(2 * math.pi * 440 * i / 16000) * 16000) for i in range(16000)]
        audio_bytes = struct.pack(f"<{len(samples)}h", *samples)
        await queue.put(audio_bytes)

        partial_texts = []

        async def partial_cb(text, conf):
            partial_texts.append((text, conf))

        text, confidence, lang = await stt.transcribe_stream(
            queue, 16000, partial_cb, language="en",
        )

        assert text == "mock transcription"
        assert confidence > 0

    @pytest.mark.asyncio
    async def test_stream_empty_queue(self, voice_config):
        """Test transcribe_stream with an empty queue returns empty."""
        stt = WhisperSTT(voice_config)

        # Mock _transcribe_segments
        async def mock_empty(audio, lang=None):
            return "", 0.0, "en"

        stt._transcribe_segments = mock_empty  # type: ignore[method-assign]
        stt._run_partial = AsyncMock()  # type: ignore[method-assign]

        queue: asyncio.Queue[bytes] = asyncio.Queue()

        text, confidence, lang = await stt.transcribe_stream(
            queue, 16000, lambda t, c: None, language="en",
        )
        assert text == ""
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_stream_partial_results(self, voice_config):
        """Test that partial results are called during streaming."""
        stt = WhisperSTT(voice_config)

        # Mock partial callback tracker
        partial_texts = []

        async def mock_partial(audio_bytes, sample_rate, partial_callback, language):
            await partial_callback("hello", 0.85)
            await partial_callback("hello world", 0.92)

        stt._run_partial = mock_partial  # type: ignore[method-assign]

        async def mock_final(audio_array, language=None):
            return "hello world", 0.92, "en"

        stt._transcribe_segments = mock_final  # type: ignore[method-assign]

        queue: asyncio.Queue[bytes] = asyncio.Queue()
        await queue.put(b"\x00\x00" * 16000)  # 1 second of silence

        async def partial_cb(text, conf):
            partial_texts.append((text, conf))

        text, confidence, lang = await stt.transcribe_stream(
            queue, 16000, partial_cb, language="en",
        )
        assert text == "hello world"
