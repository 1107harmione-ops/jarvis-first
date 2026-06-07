"""
Short-Term Memory (STM) implementation.
STM stores recent interactions with TTL-based expiry and sliding-window access.
Managed in MongoDB with automatic cleanup via TTL indexes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.database.schemas import new_memory_doc, serialize_doc
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ShortTermMemory:
    """Short-term memory manager.

    STM properties:
    - TTL-based expiry (default 24 hours)
    - Sliding window — oldest evicted when limit reached
    - Low importance threshold, no consolidation needed
    - Fast retrieval by recency
    """

    MAX_STM_PER_USER: int = 200  # Max STM items per user

    async def store(
        self,
        user_id: str,
        content: str,
        tags: list[str] | None = None,
        importance_score: float = 0.3,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new short-term memory."""
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.MEMORY_STM_TTL_HOURS
        )
        doc = new_memory_doc(
            user_id=user_id,
            content=content,
            memory_type="short_term",
            importance_score=importance_score,
            tags=tags,
            expires_at=expires_at,
            metadata=metadata,
        )
        result = await mongodb.memories.insert_one(doc)

        # Enforce per-user STM limit
        await self._enforce_limit(user_id)

        logger.debug(
            "STM stored",
            extra={"user_id": user_id, "memory_id": str(result.inserted_id)},
        )
        return serialize_doc(doc)

    async def get_recent(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """Get most recent short-term memories for a user."""
        cursor = mongodb.memories.find(
            {"user_id": user_id, "memory_type": "short_term"},
            sort=[("created_at", -1)],
            limit=limit,
            skip=skip,
        )
        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def get_context_window(
        self,
        user_id: str,
        max_items: int = 15,
    ) -> list[dict[str, Any]]:
        """Get recent memories for context building (most relevant first)."""
        cursor = mongodb.memories.find(
            {"user_id": user_id, "memory_type": "short_term", "expires_at": {"$gt": datetime.now(timezone.utc)}},
            sort=[("created_at", -1)],
            limit=max_items,
        )
        docs = await cursor.to_list(length=max_items)

        # Update access count
        ids = [d["_id"] for d in docs]
        if ids:
            await mongodb.memories.update_many(
                {"_id": {"$in": ids}},
                {"$inc": {"access_count": 1}, "$set": {"last_accessed": datetime.now(timezone.utc)}},
            )

        return [serialize_doc(d) for d in docs]

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search STM by content text (basic keyword, no embedding needed)."""
        cursor = mongodb.memories.find(
            {
                "user_id": user_id,
                "memory_type": "short_term",
                "content": {"$regex": query, "$options": "i"},
                "expires_at": {"$gt": datetime.now(timezone.utc)},
            },
            sort=[("importance_score", -1), ("created_at", -1)],
            limit=limit,
        )
        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def update_importance(
        self, memory_id: str, score: float
    ) -> bool:
        """Update importance score (e.g., after memory is accessed)."""
        result = await mongodb.memories.update_one(
            {"_id": memory_id},
            {"$set": {"importance_score": score, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    async def delete_expired(self) -> int:
        """Manually purge expired STM (backstop for TTL index)."""
        result = await mongodb.memories.delete_many({
            "memory_type": "short_term",
            "expires_at": {"$lte": datetime.now(timezone.utc)},
        })
        if result.deleted_count:
            logger.info("Purged expired STM", extra={"count": result.deleted_count})
        return result.deleted_count

    async def count_active(self, user_id: str) -> int:
        """Count active STM items for a user."""
        return await mongodb.memories.count_documents({
            "user_id": user_id,
            "memory_type": "short_term",
            "expires_at": {"$gt": datetime.now(timezone.utc)},
        })

    async def clear(self, user_id: str) -> int:
        """Clear all STM for a user (e.g., on logout)."""
        result = await mongodb.memories.delete_many({
            "user_id": user_id,
            "memory_type": "short_term",
        })
        return result.deleted_count

    # ── Internal ───────────────────────────────────────────────

    async def _enforce_limit(self, user_id: str) -> None:
        """Evict oldest STM items when limit exceeded."""
        count = await mongodb.memories.count_documents({
            "user_id": user_id,
            "memory_type": "short_term",
        })
        if count > self.MAX_STM_PER_USER:
            excess = count - self.MAX_STM_PER_USER
            cursor = mongodb.memories.find(
                {"user_id": user_id, "memory_type": "short_term"},
                sort=[("created_at", 1)],
                limit=excess,
            )
            old_ids = [doc["_id"] async for doc in cursor]
            if old_ids:
                result = await mongodb.memories.delete_many(
                    {"_id": {"$in": old_ids}}
                )
                logger.debug(
                    "Evicted excess STM",
                    extra={"user_id": user_id, "count": result.deleted_count},
                )


stm = ShortTermMemory()
