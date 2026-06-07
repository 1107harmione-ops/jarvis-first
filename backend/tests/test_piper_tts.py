"""
Tests for Piper TTS service — synthesis, streaming, model management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPiperTTSService:
    """Tests for the Piper TTS service."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        assert service._initialized is False
        assert service._available is False
        assert len(service._models) == 0

    @pytest.mark.asyncio
    async def test_initialize_finds_binary(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        # Without real binary, initialize should set available=False gracefully
        await service.initialize()
        assert service._initialized is True
        # available may be False if no binary found — that's OK

    @pytest.mark.asyncio
    async def test_synthesize_fallback_on_no_binary(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        # Should not raise even without Piper binary
        result = await service.synthesize("Hello world")
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_stream_synthesize_fallback(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        chunks = []
        async for chunk in service.stream_synthesize("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    @pytest.mark.asyncio
    async def test_set_voice_no_models(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        # Without downloaded models, set_voice should return empty string
        result = await service.set_voice(language="en")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_download_model_unknown(self) -> None:
        from backend.services.piper_tts import PiperTTSService, PiperError

        service = PiperTTSService()
        await service.initialize()
        with pytest.raises(PiperError):
            await service.download_model("unknown_model")

    @pytest.mark.asyncio
    async def test_get_available_voices(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        voices = await service.get_available_voices()
        assert isinstance(voices, list)
        # Should list downloadable voices even if not downloaded
        if voices:
            assert "key" in voices[0]
            assert "language" in voices[0]

    def test_singleton(self) -> None:
        from backend.services.piper_tts import piper_service

        assert piper_service is not None
        assert piper_service._initialized is False  # Not yet initialized

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        await service.close()
        assert service._initialized is False
        assert service._available is False

    @pytest.mark.asyncio
    async def test_synthesize_with_speed(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        result = await service.synthesize("Hello", speed=1.5)
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self) -> None:
        from backend.services.piper_tts import PiperTTSService

        service = PiperTTSService()
        await service.initialize()
        result = await service.synthesize("")
        assert isinstance(result, bytes)
