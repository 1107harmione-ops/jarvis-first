"""Text-to-speech using Edge TTS with Piper TTS fallback."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EdgeTTS:
    """Text-to-speech using Edge TTS (online, high quality)."""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> None:
        """Check Edge TTS availability."""
        try:
            import edge_tts

            voices = await edge_tts.list_voices()
            logger.info("edge_tts_available", voice_count=len(voices))
            self._initialized = True
        except Exception as e:
            logger.warning("edge_tts_unavailable", error=str(e))
            self._initialized = False

    async def synthesize(self, text: str, output_path: Optional[str | Path] = None) -> Path:
        """Synthesize text to speech audio file."""
        import edge_tts

        if output_path is None:
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            output_path = Path(path)
        else:
            output_path = Path(output_path)

        communicate = edge_tts.Communicate(text, settings.TTS_VOICE)
        await communicate.save(str(output_path))

        logger.info("tts_synthesized", text_len=len(text), output=str(output_path))
        return output_path

    @property
    def available(self) -> bool:
        """Whether the TTS engine is available."""
        return self._initialized

    async def close(self) -> None:
        """Cleanup."""
        self._initialized = False


edge_tts = EdgeTTS()
