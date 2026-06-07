"""
Long-Term Memory (LTM) implementation.
LTM stores consolidated, high-importance memories with embedding support.
Manages consolidation from STM, importance scoring, and semantic retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.database.schemas import new_memory_doc, serialize_doc
from backend.memory.short_term import stm
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """Long-term memory manager.

    LTM properties:
    - Persistent storage (no TTL)
    - Importance threshold for admission
    - Embedding-based semantic retrieval
    - Consolidation from STM via background job
    - Deduplication by cosine similarity
    """

    DEDUP_COSINE_THRESHOLD: float = 0.92
    MAX_LTM_PER_USER: int = 5000

    async def store(
        self,
        user_id: str,
        content: str,
        embedding: list[float] | None = None,
        tags: list[str] | None = None,
        importance_score: float = 0.7,
        summary: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new long-term memory. Checks dedup before insert."""
        if importance_score < settings.MEMORY_LTM_IMPORTANCE_THRESHOLD:
            logger.debug(
                "LTM skipped — below importance threshold",
                extra={"score": importance_score, "threshold": settings.MEMORY_LTM_IMPORTANCE_THRESHOLD},
            )
            raise ValueError(
                f"Importance score {importance_score} below threshold "
                f"{settings.MEMORY_LTM_IMPORTANCE_THRESHOLD}"
            )

        # Deduplication check
        if embedding:
            duplicate = await self._find_duplicate(user_id, embedding)
            if duplicate:
                # Update existing instead of creating new
                await mongodb.memories.update_one(
                    {"_id": duplicate["_id"]},
                    {
                        "$inc": {"access_count": 1},
                        "$set": {
                            "content": content,
                            "importance_score": importance_score,
                            "summary": summary,
                            "updated_at": datetime.now(timezone.utc),
                        },
                    },
                )
                logger.debug("LTM duplicate updated", extra={"memory_id": str(duplicate["_id"])})
                return serialize_doc(duplicate)

        doc = new_memory_doc(
            user_id=user_id,
            content=content,
            memory_type="long_term",
            importance_score=importance_score,
            tags=tags,
            embedding=embedding,
            summary=summary,
            source=source,
        )
        # LTM doesn't expire
        doc["expires_at"] = None
        result = await mongodb.memories.insert_one(doc)
        await self._enforce_limit(user_id)

        logger.debug(
            "LTM stored",
            extra={"user_id": user_id, "memory_id": str(result.inserted_id), "score": importance_score},
        )
        return serialize_doc(doc)

    async def semantic_search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Search LTM by embedding similarity using cosine distance.

        Uses MongoDB $vectorSearch when available, falls back to
        in-memory cosine similarity scan.
        """
        try:
            return await self._vector_search(user_id, query_embedding, limit)
        except Exception:
            logger.warning("Vector search failed, falling back to scan")
            return await self._cosine_scan(user_id, query_embedding, limit, threshold)

    async def get_by_importance(
        self,
        user_id: str,
        limit: int = 20,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Get LTM items ranked by importance score."""
        cursor = mongodb.memories.find(
            {
                "user_id": user_id,
                "memory_type": "long_term",
                "importance_score": {"$gte": min_score},
            },
            sort=[("importance_score", -1), ("last_accessed", -1)],
            limit=limit,
        )
        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def consolidate_from_stm(
        self, user_id: str, embedding_func: callable | None = None
    ) -> int:
        """Consolidate high-importance STM items into LTM.

        Args:
            user_id: Target user.
            embedding_func: Optional async callable to generate embeddings.

        Returns:
            Number of items consolidated.
        """
        # Find STM items with high importance or access count
        cursor = mongodb.memories.find({
            "user_id": user_id,
            "memory_type": "short_term",
            "consolidated": False,
            "$or": [
                {"importance_score": {"$gte": settings.MEMORY_LTM_IMPORTANCE_THRESHOLD}},
                {"access_count": {"$gte": 3}},
            ],
        })
        consolidated = 0
        async for doc in cursor:
            try:
                embedding = None
                if embedding_func and doc.get("content"):
                    embedding = await embedding_func(doc["content"])

                await self.store(
                    user_id=user_id,
                    content=doc["content"],
                    embedding=embedding,
                    tags=doc.get("tags"),
                    importance_score=doc.get("importance_score", 0.5),
                    summary=doc.get("summary"),
                    source=doc.get("source", "stm_consolidation"),
                    metadata=doc.get("metadata"),
                )
                # Mark original as consolidated
                await mongodb.memories.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"consolidated": True, "updated_at": datetime.now(timezone.utc)}},
                )
                consolidated += 1
            except ValueError:
                # Below importance threshold after all — skip
                continue
            except Exception as exc:
                logger.error("Consolidation failed", extra={"memory_id": str(doc["_id"]), "error": str(exc)})

        if consolidated:
            logger.info("STM→LTM consolidation complete", extra={"user_id": user_id, "count": consolidated})
        return consolidated

    async def get_stats(self, user_id: str) -> dict[str, int]:
        """Get LTM statistics for a user."""
        total = await mongodb.memories.count_documents({
            "user_id": user_id, "memory_type": "long_term",
        })
        by_importance = await mongodb.memories.count_documents({
            "user_id": user_id,
            "memory_type": "long_term",
            "importance_score": {"$gte": 0.8},
        })
        return {"total": total, "high_importance": by_importance}

    # ── Internal ───────────────────────────────────────────────

    async def _find_duplicate(
        self, user_id: str, embedding: list[float]
    ) -> dict[str, Any] | None:
        """Check if a similar memory already exists."""
        try:
            results = await self._vector_search(user_id, embedding, limit=1)
            if results:
                from numpy.linalg import norm
                emb = np.array(embedding)
                existing = np.array(results[0].get("embedding", []))
                if len(emb) == len(existing) and len(emb) > 0:
                    similarity = float(np.dot(emb, existing) / (norm(emb) * norm(existing)))
                    if similarity > self.DEDUP_COSINE_THRESHOLD:
                        return results[0]
        except Exception:
            pass
        return None

    async def _vector_search(
        self, user_id: str, embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """MongoDB Atlas $vectorSearch aggregation."""
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": "memory_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"user_id": user_id, "memory_type": "long_term"},
                }
            },
            {
                "$project": {
                    "score": {"$meta": "vectorSearchScore"},
                    "content": 1,
                    "memory_type": 1,
                    "importance_score": 1,
                    "tags": 1,
                    "summary": 1,
                    "source": 1,
                    "created_at": 1,
                }
            },
        ]
        cursor = mongodb.memories.aggregate(pipeline)
        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def _cosine_scan(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Fallback: in-memory cosine similarity scan."""
        from numpy.linalg import norm

        query_vec = np.array(query_embedding, dtype=np.float32)
        cursor = mongodb.memories.find({
            "user_id": user_id,
            "memory_type": "long_term",
            "embedding": {"$exists": True, "$ne": None},
        })
        scored: list[tuple[float, dict[str, Any]]] = []
        async for doc in cursor:
            emb = np.array(doc.get("embedding", []), dtype=np.float32)
            if len(emb) == 0 or len(emb) != len(query_vec):
                continue
            sim = float(np.dot(query_vec, emb) / (norm(query_vec) * norm(emb) + 1e-10))
            if sim >= threshold:
                scored.append((sim, serialize_doc(doc)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    async def _enforce_limit(self, user_id: str) -> None:
        """Evict oldest LTM items when limit exceeded."""
        count = await mongodb.memories.count_documents({
            "user_id": user_id, "memory_type": "long_term",
        })
        if count > self.MAX_LTM_PER_USER:
            excess = count - self.MAX_LTM_PER_USER
            cursor = mongodb.memories.find(
                {"user_id": user_id, "memory_type": "long_term"},
                sort=[("importance_score", 1), ("last_accessed", 1)],
                limit=excess,
            )
            ids = [doc["_id"] async for doc in cursor]
            if ids:
                result = await mongodb.memories.delete_many({"_id": {"$in": ids}})
                logger.debug("Evicted excess LTM", extra={"user_id": user_id, "count": result.deleted_count})


ltm = LongTermMemory()
