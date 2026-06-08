from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class VoiceCommandResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    success: bool
    response_text: str
    audio_file: Optional[str] = None
    data: Any = None
