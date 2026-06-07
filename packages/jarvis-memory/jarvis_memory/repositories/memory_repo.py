"""Memory repository — domain-specific queries."""

from datetime import datetime, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from jarvis_memory.models.memory import MemoryDocument, MemoryType
from jarvis_memory.repositories.base import BaseRepository


class MemoryRepository(BaseRepository[MemoryDocument]):
    """Repository for ``memories`` collection."""

    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        super().__init__(collection, MemoryDocument)

    async def search_by_user_type(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[MemoryDocument]:
        """Return memories for a user, optionally filtered by type.

        Args:
            user_id: The user's external ID.
            memory_type: Optional memory type filter.
            limit: Maximum number of results.
            skip: Number to skip for pagination.

        Returns:
            List of memory documents.
        """
        filter: dict[str, Any] = {"user_id": user_id}
        if memory_type:
            filter["memory_type"] = memory_type
        return await self.find(
            filter=filter,
            sort=[("importance_score", -1)],
            limit=limit,
            skip=skip,
        )

    async def find_by_embedding(
        self,
        embedding: list[float],
        user_id: str,
        memory_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[MemoryDocument]:
        """Vector-similarity search placeholder.

        This uses ``$vectorSearch`` when available on Atlas M10+.
        Falls back to returning recent memories sorted by importance.

        Args:
            embedding: 384-dim query embedding.
            user_id: The user's external ID.
            memory_types: Optional list of memory types to filter.
            top_k: Maximum results.

        Returns:
            List of memory documents sorted by vector similarity (or
            importance as a fallback).
        """
        pipeline: list[dict[str, Any]] = []

        # Atlas $vectorSearch stage
        pre_filter: dict[str, Any] = {"user_id": user_id}
        if memory_types:
            pre_filter["memory_type"] = {"$in": memory_types}

        pipeline.append({
            "$vectorSearch": {
                "index": "memory_vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": top_k * 4,
                "limit": top_k,
                "filter": pre_filter,
            }
        })

        # Optional: include score
        pipeline.append({
            "$addFields": {
                "vector_score": {"$meta": "vectorSearchScore"},
            }
        })

        pipeline.append({"$limit": top_k})

        try:
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            return [self._model_class.model_validate(doc) for doc in results]
        except Exception:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Vector search failed (M10+ cluster required); falling back to importance-based sort")
            # Fallback: return memories sorted by importance
            filter: dict[str, Any] = {"user_id": user_id}
            if memory_types:
                filter["memory_type"] = {"$in": memory_types}
            return await self.find(
                filter=filter,
                sort=[("importance_score", -1)],
                limit=top_k,
            )

    async def get_consolidation_candidates(
        self,
        user_id: str,
        min_importance: float = 0.6,
        age_hours: float = 1.0,
    ) -> list[MemoryDocument]:
        """Return short-term memories eligible for LTM consolidation.

        Candidates are STM entries created more than *age_hours* ago
        with an importance score ≥ *min_importance* and not yet consolidated.

        Args:
            user_id: The user's external ID.
            min_importance: Minimum importance score threshold.
            age_hours: Minimum age in hours to consider for consolidation.

        Returns:
            List of candidate memory documents.
        """
        cutoff = datetime.utcnow() - timedelta(hours=age_hours)
        return await self.find(
            filter={
                "user_id": user_id,
                "memory_type": "short_term",
                "consolidated": False,
                "importance_score": {"$gte": min_importance},
                "created_at": {"$lte": cutoff},
            },
            sort=[("importance_score", -1)],
            limit=100,
        )

    async def increment_access(
        self,
        memory_id: str,
    ) -> MemoryDocument | None:
        """Atomically increment ``access_count`` and update ``last_accessed``.

        Args:
            memory_id: The memory ``_id`` as a string.

        Returns:
            The updated memory document or ``None``.
        """
        from bson import ObjectId

        result = await self.collection.find_one_and_update(
            {"_id": ObjectId(memory_id)},
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed": datetime.utcnow()},
            },
            return_document=True,
        )
        if result is None:
            return None
        return self._model_class.model_validate(result)
