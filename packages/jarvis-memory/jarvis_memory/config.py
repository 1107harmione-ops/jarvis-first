"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "jarvis"

    # Embedding
    EMBEDDING_PROVIDER: str = "sentence_transformers"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_API_KEY: str | None = None

    # Memory lifecycle
    MEMORY_STM_TTL_HOURS: int = 24
    MEMORY_CONSOLIDATION_INTERVAL_MINUTES: int = 30
    MEMORY_DECAY_RATE: float = 0.1
    MEMORY_DEFAULT_TOP_K: int = 10
    MEMORY_MAX_CONTEXT_TOKENS: int = 3000

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
