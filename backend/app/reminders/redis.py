"""Redis connection manager for reminder scheduling."""

from __future__ import annotations

from typing import Optional

import redis as sync_redis

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RedisManager:
    """Manages a synchronous Redis connection used by RQ."""

    def __init__(self) -> None:
        self._client: Optional[sync_redis.Redis] = None

    @property
    def client(self) -> sync_redis.Redis:
        if self._client is None:
            self._client = sync_redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            logger.info("redis_connected", url=settings.redis_url)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("redis_close_error", error=str(e))
            self._client = None
            logger.info("redis_disconnected")

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self.client.ping()
        except Exception:
            return False


redis_manager = RedisManager()
