"""STT providers."""
from jarvis_voice.stt.base import BaseSTTProvider

try:
    from jarvis_voice.stt.whisper_stt import WhisperSTT
except ImportError:
    from jarvis_voice.stt.base import BaseSTTProvider as WhisperSTT  # type: ignore[assignment]
    WhisperSTT = None  # type: ignore[assignment]


__all__ = ["BaseSTTProvider", "WhisperSTT"]
