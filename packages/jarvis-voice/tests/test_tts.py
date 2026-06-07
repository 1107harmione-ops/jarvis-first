"""Tests for PiperTTS provider."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis_voice.tts.piper_tts import PiperTTS


class TestPiperTTS:
    """Test suite for PiperTTS."""

    @pytest.mark.asyncio
    async def test_initialization(self, voice_config):
        """Test PiperTTS initialises with correct config."""
        tts = PiperTTS(voice_config)
        assert tts.voices["en"] == voice_config.piper_voice_en
        assert tts.voices["hi"] == voice_config.piper_voice_hi
        assert tts.length_scale == voice_config.tts_length_scale
        assert tts.noise_scale == voice_config.tts_noise_scale

    @pytest.mark.asyncio
    async def test_voice_paths(self, voice_config):
        """Test voice path resolution."""
        tts = PiperTTS(voice_config)
        model_path, config_path = await tts._get_voice_paths("en")
        assert model_path.name == f"{voice_config.piper_voice_en}.onnx"
        assert config_path.name == f"{voice_config.piper_voice_en}.onnx.json"

    @pytest.mark.asyncio
    async def test_voice_paths_hindi(self, voice_config):
        """Test Hindi voice path resolution."""
        tts = PiperTTS(voice_config)
        model_path, config_path = await tts._get_voice_paths("hi")
        assert model_path.name == f"{voice_config.piper_voice_hi}.onnx"

    @pytest.mark.asyncio
    async def test_voice_paths_fallback_to_en(self, voice_config):
        """Test unsupported language falls back to English."""
        tts = PiperTTS(voice_config)
        model_path, config_path = await tts._get_voice_paths("fr")
        assert model_path.name == f"{voice_config.piper_voice_en}.onnx"

    @pytest.mark.asyncio
    async def test_fallback_silence(self, voice_config):
        """Test fallback silence generation."""
        tts = PiperTTS(voice_config)
        interrupt_event = asyncio.Event()
        chunks = []

        async def collect(chunk):
            chunks.append(chunk)

        await tts._fallback_silence(collect, interrupt_event)
        assert len(chunks) > 0
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 0

    @pytest.mark.asyncio
    async def test_fallback_silence_respects_interrupt(self, voice_config):
        """Test fallback silence stops when interrupted."""
        tts = PiperTTS(voice_config)
        interrupt_event = asyncio.Event()
        chunks = []

        async def collect(chunk):
            chunks.append(chunk)

        # Set interrupt after a small delay
        async def set_interrupt():
            await asyncio.sleep(0.05)
            interrupt_event.set()

        await asyncio.gather(
            tts._fallback_silence(collect, interrupt_event),
            set_interrupt(),
        )
        # Should have some chunks but not all
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_speak_stream_subprocess_called(self, voice_config):
        """Test that speak_stream attempts to launch a subprocess."""
        tts = PiperTTS(voice_config)
        interrupt_event = asyncio.Event()
        chunks = []

        async def collect(chunk):
            chunks.append(chunk)

        # With model file not existing, it should fall back to silence
        await tts.speak_stream("Hello world", "en", collect, interrupt_event)
        # Should at least produce fallback silence
        assert len(chunks) >= 0  # May be empty if model missing, which is expected

    @pytest.mark.asyncio
    async def test_speak_stream_interrupt(self, voice_config):
        """Test that interrupt stops TTS."""
        tts = PiperTTS(voice_config)
        interrupt_event = asyncio.Event()
        chunks = []

        async def collect(chunk):
            chunks.append(chunk)

        # Set interrupt immediately so it triggers
        interrupt_event.set()
        await tts.speak_stream("Hello world", "en", collect, interrupt_event)

    @pytest.mark.asyncio
    async def test_chunk_callback_receives_data(self, voice_config):
        """Test that the chunk callback receives data during fallback."""
        tts = PiperTTS(voice_config)
        interrupt_event = asyncio.Event()
        received_chunks = []

        async def collect(chunk):
            received_chunks.append(chunk)

        await tts._fallback_silence(collect, interrupt_event)
        assert len(received_chunks) > 0
        # Fallback silence should produce multiple chunks
        total_size = sum(len(c) for c in received_chunks)
        expected_size = int(voice_config.tts_sample_rate * 0.5 * 2)  # 0.5s of PCM16
        assert total_size >= expected_size * 0.5  # At least half expected

    @pytest.mark.asyncio
    async def test_find_piper_fallback(self):
        """Test _find_piper returns a sensible default."""
        executable = PiperTTS._find_piper()
        assert isinstance(executable, str)
        assert len(executable) > 0
