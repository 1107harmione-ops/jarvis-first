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

    # MongoDB (optional — for scalable storage)
    MONGODB_URL: str = ""
    MONGODB_DB_NAME: str = "jarvis"

    # Groq API (optional — LLM inference)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "mixtral-8x7b-32768"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TEMPERATURE: float = 0.7

    # OpenCode Zen API (optional — AI code assistance)
    OPENCODE_ZEN_API_KEY: str = ""
    OPENCODE_ZEN_MODEL: str = "gpt-4o-mini"
    OPENCODE_ZEN_BASE_URL: str = ""
    OPENCODE_ZEN_MAX_TOKENS: int = 2048

    # Vosk STT
    VOSK_MODEL_PATH: str = "./models/vosk-model-small-en-us-0.15"
    VOSK_SAMPLE_RATE: int = 16000

    # TTS
    TTS_VOICE: str = "en-US-AriaNeural"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_URL: str = ""

    @property
    def redis_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # RQ
    RQ_QUEUE_NAME: str = "jarvis"
    RQ_DEFAULT_TIMEOUT: int = 300

    # Reminder checker
    REMINDER_CHECK_INTERVAL: int = 60  # seconds

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # VAD
    VAD_MODE: int = 1  # 0-3, higher = more aggressive silence removal
    VAD_FRAME_MS: int = 30
    VAD_ENERGY_THRESHOLD: float = 300.0

    # Wake Word
    WAKE_WORD_ENABLED: bool = False
    WAKE_WORD_SENSITIVITY: float = 0.5
    WAKE_WORD_KEYWORDS: str = "jarvis,hey jarvis"
    WAKE_WORD_ENERGY_THRESHOLD: float = 500.0

    # Paths
    DATA_DIR: Path = Path("./data")


settings = Settings()
