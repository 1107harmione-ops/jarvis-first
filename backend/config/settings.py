"""
JARVIS Backend Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enterprise-grade settings management with environment variable support.
All secrets loaded via environment, sensible defaults for development.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator, MongoDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(str, Enum):
    """Deployment environment enum."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level enum."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Central application settings.

    All values loaded from environment variables first, then .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App Core ──────────────────────────────────────────────
    APP_NAME: str = "JARVIS"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Enterprise AI Assistant Backend"
    ENVIRONMENT: AppEnvironment = AppEnvironment.DEVELOPMENT
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    SECRET_KEY: str = Field(
        default="change-me-in-production-please",
        description="Django/FastAPI secret key for crypto signing",
    )

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = Field(default=4, ge=1, le=32, description="Number of Uvicorn workers")
    CORS_ORIGINS: list[str] = ["*"]
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, ge=1, description="Max API requests per minute")

    # ── MongoDB ───────────────────────────────────────────────
    MONGODB_URI: MongoDsn = Field(
        default=MongoDsn("mongodb://localhost:27017/jarvis"),
        description="MongoDB connection string",
    )
    MONGODB_DATABASE: str = "jarvis"
    MONGODB_MAX_POOL_SIZE: int = 100
    MONGODB_MIN_POOL_SIZE: int = 10
    MONGODB_MAX_IDLE_TIME_MS: int = 10000

    # ── JWT Authentication ────────────────────────────────────
    JWT_SECRET_KEY: str = Field(
        default="jwt-secret-change-me",
        description="JWT signing key",
    )
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, ge=5, le=1440, description="Access token TTL"
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, ge=1, le=90, description="Refresh token TTL"
    )

    # ── AI Models ─────────────────────────────────────────────
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    CODEX_API_KEY: str = ""
    CODEX_BASE_URL: str = "https://api.openai.com/v1"
    CODEX_MODEL: str = "gpt-4o"

    MINIMAX_API_KEY: str = ""
    MINIMAX_BASE_URL: str = "https://api.minimax.chat/v1"
    MINIMAX_MODEL: str = "minimax-m2.1"

    MIMO_API_KEY: str = ""
    MIMO_BASE_URL: str = "https://api.mimo.com/v1"
    MIMO_MODEL: str = "mimo-v2-omni"

    # ── Memory ────────────────────────────────────────────────
    MEMORY_STM_TTL_HOURS: int = Field(default=24, description="Short-term memory expiry")
    MEMORY_LTM_IMPORTANCE_THRESHOLD: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum importance for LTM storage"
    )
    MEMORY_VECTOR_DIMENSION: int = 384
    MEMORY_MAX_CONTEXT_TOKENS: int = 4096
    MEMORY_CONSULIDATION_INTERVAL_MINUTES: int = 60

    # ── Voice ─────────────────────────────────────────────────
    STT_MODEL: str = "whisper-1"
    TTS_MODEL: str = "tts-1"
    TTS_VOICE: str = "alloy"
    VOICE_SESSION_TIMEOUT_SECONDS: int = Field(
        default=300, ge=30, description="Voice session idle timeout"
    )
    VOICE_STREAMING_ENABLED: bool = Field(
        default=True, description="Enable streaming voice processing"
    )
    VOICE_WEBSOCKET_TIMEOUT: int = Field(
        default=600, ge=60, description="Voice WebSocket idle timeout in seconds"
    )

    # ── Piper TTS (local neural TTS) ──────────────────────────
    PIPER_ENABLED: bool = Field(
        default=True, description="Enable local Piper TTS"
    )
    PIPER_MODEL_PATH: str = Field(
        default="./models/piper/", description="Path to Piper voice models directory"
    )
    PIPER_VOICE_EN: str = Field(
        default="en_US-lessac-medium", description="Default English Piper voice"
    )
    PIPER_VOICE_HI: str = Field(
        default="hi_IN-medium", description="Default Hindi Piper voice"
    )
    PIPER_EXECUTABLE_PATH: str = Field(
        default="", description="Path to Piper binary (auto-detect if empty)"
    )

    # ── Whisper STT ───────────────────────────────────────────
    WHISPER_API_KEY: str = Field(
        default="", description="OpenAI API key for Whisper STT"
    )
    WHISPER_MODEL: str = Field(
        default="whisper-1", description="Whisper model name"
    )
    WHISPER_API_BASE_URL: str = Field(
        default="", description="OpenAI API base URL (default: api.openai.com)"
    )

    # ── Task System ───────────────────────────────────────────
    TASK_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    TASK_BACKGROUND_POLL_SECONDS: int = Field(
        default=30, ge=5, description="Background worker poll interval"
    )

    # ── Research ───────────────────────────────────────────
    RESEARCH_CACHE_TTL_HOURS: int = Field(default=24, ge=1, description="Research cache TTL in hours")
    RESEARCH_MAX_SOURCES: int = Field(default=50, ge=5, le=200, description="Max sources per research query")
    RESEARCH_DEFAULT_DEPTH: str = "moderate"

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    LOG_FILE: str | None = None

    # ── Redis (optional caching) ──────────────────────────────
    REDIS_URL: str | None = None

    # ── Validators ────────────────────────────────────────────

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == AppEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == AppEnvironment.DEVELOPMENT


# Global singleton
settings = Settings()
