"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    APP_NAME: str = "Jarvis"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./jarvis.db"

    # Vosk STT
    VOSK_MODEL_PATH: str = "./models/vosk-model-small-en-us-0.15"
    VOSK_SAMPLE_RATE: int = 16000

    # TTS
    TTS_VOICE: str = "en-US-AriaNeural"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Paths
    DATA_DIR: Path = Path("./data")


settings = Settings()
