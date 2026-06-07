"""
Tests for Whisper STT service — transcription, streaming, VAD, language support.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVoiceActivityDetector:
    """Tests for the VAD component."""

    def setup_method(self) -> None:
        from backend.services.whisper_stt import VoiceActivityDetector

        self.vad = VoiceActivityDetector()

    def test_initial_state(self) -> None:
        assert self.vad.is_speaking is False
        assert self.vad.noise_floor == 0.01

    def test_silence_is_not_speech(self) -> None:
        """Silent audio (all zeros) should not trigger speech."""
        silent_chunk = b"\x00\x00" * 240  # 240 samples of silence
        result = self.vad.process_chunk(silent_chunk)
        assert result["is_speech"] is False
        assert result["speech_started"] is False

    def test_loud_audio_triggers_speech(self) -> None:
        """Loud audio should eventually trigger speech detection."""
        # Generate a loud 500Hz tone
        import struct
        import math

        samples = []
        for i in range(480):  # 30ms at 16kHz
            val = int(math.sin(2 * math.pi * 500 * i / 16000) * 16000)
            samples.append(struct.pack("<h", val))
        loud_chunk = b"".join(samples)

        # First chunk may not trigger (needs 2 frames debounce)
        result1 = self.vad.process_chunk(loud_chunk)
        # Second chunk should trigger
        result2 = self.vad.process_chunk(loud_chunk)
        assert result2["is_speech"] is True

    def test_reset(self) -> None:
        self.vad._is_speaking = True
        self.vad._speech_frames = 100
        self.vad.reset()
        assert self.vad.is_speaking is False
        assert self.vad._speech_frames == 0

    def test_noise_floor_update(self) -> None:
        import struct
        import math

        # Generate consistent background noise
        samples = []
        for i in range(480):
            val = int(math.sin(2 * math.pi * 200 * i / 16000) * 1000)
            samples.append(struct.pack("<h", val))
        chunk = b"".join(samples)

        self.vad.process_chunk(chunk)
        initial_floor = self.vad.noise_floor
        # Noise floor should have updated
        assert initial_floor > 0.01


class TestWhisperSTTService:
    """Tests for the Whisper STT service."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        assert service._initialized is False
        await service.initialize()
        assert service._initialized is True

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        await service.initialize()
        result = await service.transcribe(b"")
        assert result.text == ""
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_very_short_audio(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        await service.initialize()
        result = await service.transcribe(b"\x00\x00" * 10)
        assert result.text == ""

    @pytest.mark.asyncio
    async def test_pcm16_to_wav(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService
        import wave
        import io

        service = WhisperSTTService()
        pcm_data = b"\x00\x00" * 16000  # 1 second of silence at 16kHz
        wav_bytes = service._pcm16_to_wav(pcm_data, 16000)

        # Verify WAV format
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_resample_audio_same_rate(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        data = b"\x00\x01\x02\x03" * 100
        result = service.resample_audio(data, 16000, 16000)
        assert result == data

    def test_resample_audio_downsample(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        data = b"\x00\x01\x02\x03" * 200
        result = service.resample_audio(data, 44100, 16000)
        assert len(result) < len(data)
        assert len(result) > 0

    def test_resample_empty(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        result = service.resample_audio(b"", 16000, 16000)
        assert result == b""

    @pytest.mark.asyncio
    async def test_create_vad(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService, VoiceActivityDetector

        service = WhisperSTTService()
        vad = service.create_vad()
        assert isinstance(vad, VoiceActivityDetector)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        from backend.services.whisper_stt import WhisperSTTService

        service = WhisperSTTService()
        await service.initialize()
        await service.close()
        assert service._initialized is False
        assert service._available is False


class TestPartialTranscript:
    """Tests for the PartialTranscript model."""

    def test_defaults(self) -> None:
        from backend.services.whisper_stt import PartialTranscript

        pt = PartialTranscript(text="hello", confidence=0.9)
        assert pt.text == "hello"
        assert pt.confidence == 0.9
        assert pt.is_final is False
        assert pt.stability == 0.0

    def test_final_transcript(self) -> None:
        from backend.services.whisper_stt import PartialTranscript

        pt = PartialTranscript(text="hello world", confidence=0.95, is_final=True, stability=0.9)
        assert pt.is_final is True
        assert pt.stability == 0.9


class TestTranscriptionResult:
    """Tests for the TranscriptionResult model."""

    def test_to_dict(self) -> None:
        from backend.services.whisper_stt import TranscriptionResult

        result = TranscriptionResult(
            text="hello",
            confidence=0.95,
            language="en",
            duration_ms=1500.0,
        )
        d = result.to_dict()
        assert d["text"] == "hello"
        assert d["confidence"] == 0.95
        assert d["language"] == "en"
        assert d["is_final"] is True
