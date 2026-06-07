"""
Logging configuration for JARVIS backend.
Structured JSON logging for production, pretty console for development.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.settings import settings


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging (production)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_entry.update(record.extra)  # type: ignore[arg-type]
        return json.dumps(log_entry)


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Adapter that adds agent context to log records."""

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra", {})
        extra["agent"] = self.extra.get("agent", "unknown")
        extra["session_id"] = self.extra.get("session_id", "unknown")
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging() -> None:
    """Configure root logger based on environment."""
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.value)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if settings.is_production:
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(
            logging.Formatter(settings.LOG_FORMAT)
        )
    root.addHandler(console)

    # File handler (optional)
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Set third-party log levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)


def get_logger(name: str, agent: str | None = None, session_id: str | None = None) -> logging.Logger | AgentLoggerAdapter:
    """Get a logger, optionally with agent context.

    Args:
        name: Logger name (typically __name__).
        agent: Optional agent name for context.
        session_id: Optional session ID for context.

    Returns:
        Logger or AgentLoggerAdapter if agent context provided.
    """
    logger = logging.getLogger(name)
    if agent:
        return AgentLoggerAdapter(logger, {"agent": agent, "session_id": session_id or "unknown"})
    return logger
