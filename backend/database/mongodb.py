"""
MongoDB connection management with Motor async driver.
Singleton pattern for connection pooling, lifecycle hooks for startup/shutdown.
"""

from __future__ import annotations

import asyncio
from typing import Any

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo.errors import ConnectionFailure, OperationFailure

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MongoDBManager:
    """Manages MongoDB connection lifecycle and provides collection accessors."""

    def __init__(self) -> None:
        self._client: AsyncIOMotorClient[Any] | None = None
        self._db: AsyncIOMotorDatabase | None = None
        self._connected = False
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize the MongoDB connection pool."""
        async with self._lock:
            if self._connected:
                return
            try:
                self._client = AsyncIOMotorClient(
                    str(settings.MONGODB_URI),
                    maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
                    minPoolSize=settings.MONGODB_MIN_POOL_SIZE,
                    maxIdleTimeMS=settings.MONGODB_MAX_IDLE_TIME_MS,
                    serverSelectionTimeoutMS=5000,
                )
                # Ping to verify connection
                await self._client.admin.command("ping")
                self._db = self._client[settings.MONGODB_DATABASE]
                self._connected = True
                logger.info(
                    "Connected to MongoDB",
                    extra={"database": settings.MONGODB_DATABASE},
                )
            except (ConnectionFailure, OperationFailure) as exc:
                logger.error("MongoDB connection failed", extra={"error": str(exc)})
                raise

    async def disconnect(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._client:
                self._client.close()
                self._client = None
                self._db = None
                self._connected = False
                logger.info("Disconnected from MongoDB")

    # ── Collection Accessors ───────────────────────────────────

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get the database instance. Raises RuntimeError if not connected."""
        if not self._db:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db

    def get_collection(self, name: str) -> AsyncIOMotorCollection:
        """Get a collection by name."""
        return self.db[name]

    # ── Collection Definitions ─────────────────────────────────

    @property
    def users(self) -> AsyncIOMotorCollection:
        return self.get_collection("users")

    @property
    def conversations(self) -> AsyncIOMotorCollection:
        return self.get_collection("conversations")

    @property
    def messages(self) -> AsyncIOMotorCollection:
        return self.get_collection("messages")

    @property
    def tasks(self) -> AsyncIOMotorCollection:
        return self.get_collection("tasks")

    @property
    def memories(self) -> AsyncIOMotorCollection:
        return self.get_collection("memories")

    @property
    def settings(self) -> AsyncIOMotorCollection:
        return self.get_collection("settings")

    @property
    def agent_logs(self) -> AsyncIOMotorCollection:
        return self.get_collection("agent_logs")

    @property
    def analytics(self) -> AsyncIOMotorCollection:
        return self.get_collection("analytics")

    @property
    def knowledge(self) -> AsyncIOMotorCollection:
        return self.get_collection("knowledge")

    @property
    def voice_history(self) -> AsyncIOMotorCollection:
        return self.get_collection("voice_history")

    @property
    def voice_commands(self) -> AsyncIOMotorCollection:
        return self.get_collection("voice_commands")

    @property
    def voice_preferences(self) -> AsyncIOMotorCollection:
        return self.get_collection("voice_preferences")

    @property
    def voice_sessions(self) -> AsyncIOMotorCollection:
        return self.get_collection("voice_sessions")

    @property
    def offline_queue(self) -> AsyncIOMotorCollection:
        return self.get_collection("offline_queue")

    @property
    def research_reports(self) -> AsyncIOMotorCollection:
        return self.get_collection("research_reports")

    @property
    def research_sources(self) -> AsyncIOMotorCollection:
        return self.get_collection("research_sources")

    @property
    def research_cache(self) -> AsyncIOMotorCollection:
        return self.get_collection("research_cache")

    # ── Index Management ───────────────────────────────────────

    async def ensure_indexes(self) -> None:
        """Create all required indexes on startup."""
        indexes: dict[str, list[tuple[str, int] | tuple[str, int, dict[str, Any]]]] = {
            "users": [
                ("email", 1),
                ("api_key_hashed", 1),
                ("created_at", -1),
            ],
            "conversations": [
                ("user_id", 1),
                ("created_at", -1),
                [("user_id", 1), ("updated_at", -1)],
            ],
            "messages": [
                ("conversation_id", 1),
                ("created_at", 1),
                [("conversation_id", 1), ("created_at", 1)],
            ],
            "tasks": [
                ("user_id", 1),
                ("status", 1),
                ("due_at", 1),
                [("user_id", 1), ("status", 1)],
            ],
            "memories": [
                ("user_id", 1),
                ("memory_type", 1),
                ("created_at", -1),
                [("user_id", 1), ("memory_type", 1)],
                [("user_id", 1), ("importance_score", -1)],
            ],
            "agent_logs": [
                ("agent_name", 1),
                ("session_id", 1),
                ("created_at", -1),
                [("agent_name", 1), ("created_at", -1)],
            ],
            "analytics": [
                ("event_type", 1),
                ("created_at", -1),
                [("user_id", 1), ("event_type", 1)],
            ],
            "knowledge": [
                ("source", 1),
                ("created_at", -1),
                [("tags", 1)],
            ],
            "voice_history": [
                ("user_id", 1),
                ("created_at", -1),
                [("user_id", 1), ("created_at", -1)],
                [("user_id", 1), ("language", 1)],
            ],
            "voice_commands": [
                ("user_id", 1),
                ("command", 1),
                ("count", -1),
                [("user_id", 1), ("command", 1)],
            ],
            "voice_preferences": [
                [("user_id", 1)],
            ],
            "voice_sessions": [
                ("user_id", 1),
                ("created_at", -1),
                [("user_id", 1), ("created_at", -1)],
            ],
            "offline_queue": [
                ("user_id", 1),
                ("status", 1),
                ("created_at", 1),
                [("user_id", 1), ("status", 1)],
            ],
            "research_reports": [
                ("user_id", 1),
                ("created_at", -1),
                ("research_type", 1),
                [("user_id", 1), ("created_at", -1)],
                [("tags", 1)],
            ],
            "research_sources": [
                ("url", 1),
                ("domain", 1),
                ("overall_score", -1),
                ("created_at", -1),
            ],
            "research_cache": [
                ("cache_key", 1),
                ("ttl", 1),
            ],
        }

        for collection_name, index_specs in indexes.items():
            collection = self.get_collection(collection_name)
            existing = await collection.index_information()
            for idx in index_specs:
                if isinstance(idx, list):
                    keys = [(k, v) for k, v in idx]  # type: ignore[misc]
                elif isinstance(idx, tuple):
                    keys = [idx]  # type: ignore[list-item]
                else:
                    continue
                # Skip if similar index exists
                index_name = "_".join(f"{k}_{v}" for k, v in keys)
                if index_name not in existing:
                    kwargs: dict[str, Any] = {}
                    if len(keys) == 1 and isinstance(keys[0], tuple) and len(keys[0]) == 3:
                        keys, kwargs = [(keys[0][0], keys[0][1])], keys[0][2]  # type: ignore[assignment]
                    try:
                        await collection.create_index(keys, **kwargs)
                        logger.debug("Created index", extra={"collection": collection_name, "index": keys})
                    except OperationFailure as exc:
                        logger.warning("Index creation failed", extra={"collection": collection_name, "error": str(exc)})

        logger.info("All indexes ensured")


# Global singleton
mongodb = MongoDBManager()
