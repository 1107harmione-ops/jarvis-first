"""Vosk speech-to-text wrapper."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Optional

import vosk

from app.core.config import settings
from app.core.exceptions import VoiceProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class VoskSTT:
    """Offline speech-to-text using Vosk."""

    def __init__(self):
        self.model: Optional[vosk.Model] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load the Vosk model."""
        model_path = Path(settings.VOSK_MODEL_PATH)
        if not model_path.exists():
            logger.warning("vosk_model_not_found", path=str(model_path))
            raise VoiceProcessingError(f"Vosk model not found at {model_path}")

        self.model = vosk.Model(str(model_path))
        self._initialized = True
        logger.info("vosk_model_loaded", path=str(model_path))

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe audio file to text."""
        if not self._initialized:
            raise VoiceProcessingError("Vosk model not initialized")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise VoiceProcessingError(f"Audio file not found: {audio_path}")

        wf = wave.open(str(audio_path), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise VoiceProcessingError("Audio must be WAV: mono, 16-bit")

        recognizer = vosk.KaldiRecognizer(self.model, wf.getframerate())
        recognizer.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        logger.info("stt_transcribed", text=text, audio=str(audio_path))
        return text

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM audio bytes."""
        if not self._initialized:
            raise VoiceProcessingError("Vosk model not initialized")

        recognizer = vosk.KaldiRecognizer(self.model, sample_rate)

        if recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(recognizer.Result())
        else:
            result = json.loads(recognizer.FinalResult())

        text = result.get("text", "").strip()
        logger.info("stt_transcribed_bytes", text=text, length=len(audio_bytes))
        return text

    async def close(self) -> None:
        """Cleanup resources."""
        self.model = None
        self._initialized = False
        logger.info("vosk_model_unloaded")


vosk_stt = VoskSTT()
