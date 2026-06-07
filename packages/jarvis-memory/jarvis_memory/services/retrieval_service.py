"""Hybrid retrieval service — vector + keyword search with fusion."""

import logging
from dataclasses import dataclass, field
from typing import Any

from jarvis_memory.models.memory import MemoryType
from jarvis_memory.repositories.memory_repo import MemoryRepository
from jarvis_memory.repositories.knowledge_repo import KnowledgeRepository
from jarvis_memory.services.embedding_service import EmbeddingService
from jarvis_memory.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """A memory result with a fused relevance score."""

    memory_id: str
    content: str
    memory_type: str
    importance_score: float
    embedding: list[float] | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_accessed: Any = None
    access_count: int = 0
    source: str | None = None
    source_id: str | None = None
    score: float = 0.0
    vector_score: float = 0.0
    text_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance_score": self.importance_score,
            "tags": self.tags,
            "source": self.source,
            "source_id": self.source_id,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "text_score": round(self.text_score, 4),
        }


class RetrievalService:
    """Multi-strategy retrieval with recency-fused ranking.

    Combines:
    1. Vector search (Atlas ``$vectorSearch``)
    2. Keyword text search
    3. Recency & importance boost
    """

    VECTOR_WEIGHT = 0.6
    IMPORTANCE_WEIGHT = 0.2
    RECENCY_WEIGHT = 0.2

    def __init__(
        self,
        memory_repo: MemoryRepository,
        knowledge_repo: KnowledgeRepository,
        embedding_service: EmbeddingService,
        scoring_service: ScoringService,
    ) -> None:
        self._memory_repo = memory_repo
        self._knowledge_repo = knowledge_repo
        self._embedder = embedding_service
        self._scoring = scoring_service

    async def hybrid_search(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        memory_types: list[MemoryType] | None = None,
        top_k: int = 10,
    ) -> list[ScoredMemory]:
        """Hybrid search: vector search + keyword search with fusion.

        Args:
            user_id: The user's external ID.
            query: Raw text query.
            query_embedding: Pre-computed query embedding. If ``None``,
                it is generated on the fly.
            memory_types: Optional list of memory types to restrict to.
            top_k: Maximum number of results.

        Returns:
            List of ``ScoredMemory`` objects sorted by fused score.
        """
        if query_embedding is None:
            query_embedding = await self._embedder.embed(query)

        # 1. Vector search
        vector_results = await self._vector_search(
            query_embedding, user_id, memory_types, k=top_k * 2
        )

        # 2. Text search (keyword fallback)
        text_results = await self._text_search(
            query, user_id, memory_types, k=top_k
        )

        # 3. Fuse & deduplicate
        fused = self._fuse_results(vector_results + text_results)

        # 4. Apply recency & importance boost
        for mem in fused:
            recency = self._scoring.compute_recency(mem.last_accessed)
            mem.score = (
                self.VECTOR_WEIGHT * mem.vector_score
                + self.IMPORTANCE_WEIGHT * mem.importance_score
                + self.RECENCY_WEIGHT * recency
            )

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    async def search_knowledge(
        self,
        user_id: str | None,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search knowledge documents using vector similarity.

        Args:
            user_id: Optional user ID for scoping. ``None`` returns
                global knowledge.
            query: Raw text query.
            query_embedding: Pre-computed query embedding.
            top_k: Maximum results.

        Returns:
            List of scored knowledge dicts.
        """
        if query_embedding is None:
            query_embedding = await self._embedder.embed(query)

        docs = await self._knowledge_repo.find_by_embedding(
            embedding=query_embedding,
            user_id=user_id,
            top_k=top_k,
        )

        results: list[dict[str, Any]] = []
        for doc in docs:
            try:
                score = await self._embedder.cosine_similarity(
                    doc.embedding or [], query_embedding
                )
            except Exception:
                score = 0.0

            results.append({
                "knowledge_id": doc.knowledge_id,
                "title": doc.title,
                "summary": doc.summary,
                "source_type": doc.source_type,
                "url": doc.url,
                "relevance": round(max(score, 0.0), 4),
            })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    async def _vector_search(
        self,
        embedding: list[float],
        user_id: str,
        memory_types: list[MemoryType] | None = None,
        k: int = 20,
    ) -> list[ScoredMemory]:
        """Execute vector search via Atlas ``$vectorSearch``.

        Args:
            embedding: Query embedding.
            user_id: User ID filter.
            memory_types: Optional memory type filter.
            k: Number of candidates.

        Returns:
            List of ``ScoredMemory`` from vector search.
        """
        try:
            memories = await self._memory_repo.find_by_embedding(
                embedding=embedding,
                user_id=user_id,
                memory_types=memory_types,
                top_k=k,
            )
        except Exception as exc:
            logger.warning("Vector search failed (M10+ cluster required): %s", exc)
            # Fallback: get recent memories by importance
            memories = await self._memory_repo.search_by_user_type(
                user_id, limit=k
            )

        results: list[ScoredMemory] = []
        for mem in memories:
            vector_score = getattr(mem, "vector_score", 0.0) or 0.0
            results.append(
                ScoredMemory(
                    memory_id=mem.memory_id,
                    content=mem.content,
                    memory_type=mem.memory_type or "",
                    importance_score=mem.importance_score,
                    embedding=mem.embedding,
                    tags=mem.tags,
                    metadata=mem.metadata,
                    last_accessed=mem.last_accessed,
                    access_count=mem.access_count,
                    source=mem.source,
                    source_id=mem.source_id,
                    vector_score=vector_score,
                )
            )
        return results

    async def _text_search(
        self,
        query: str,
        user_id: str,
        memory_types: list[MemoryType] | None = None,
        k: int = 10,
    ) -> list[ScoredMemory]:
        """Execute keyword text search via MongoDB ``$text``.

        Args:
            query: Raw text query.
            user_id: User ID filter.
            memory_types: Optional memory type filter.
            k: Maximum results.

        Returns:
            List of ``ScoredMemory`` from text search.
        """
        pipeline: list[dict[str, Any]] = []

        match: dict[str, Any] = {"user_id": user_id}
        if memory_types:
            match["memory_type"] = {"$in": memory_types}

        pipeline.append({"$match": match})
        pipeline.append({
            "$match": {
                "$or": [
                    {"content": {"$regex": query, "$options": "i"}},
                    {"tags": {"$regex": query, "$options": "i"}},
                ]
            }
        })
        pipeline.append({"$addFields": {"text_score": {"$literal": 1.0}}})
        pipeline.append({"$sort": {"importance_score": -1}})
        pipeline.append({"$limit": k})

        try:
            cursor = self._memory_repo.collection.aggregate(pipeline)
            docs = await cursor.to_list(length=k)
        except Exception as exc:
            logger.warning("Text search failed: %s", exc)
            return []

        results: list[ScoredMemory] = []
        for doc in docs:
            mem = MemoryType  # noqa: F841 — just importing for hint
            results.append(
                ScoredMemory(
                    memory_id=doc.get("memory_id", ""),
                    content=doc.get("content", ""),
                    memory_type=doc.get("memory_type", ""),
                    importance_score=doc.get("importance_score", 0.0),
                    embedding=doc.get("embedding"),
                    tags=doc.get("tags", []),
                    metadata=doc.get("metadata", {}),
                    last_accessed=doc.get("last_accessed"),
                    access_count=doc.get("access_count", 0),
                    source=doc.get("source"),
                    source_id=doc.get("source_id"),
                    text_score=1.0,
                )
            )
        return results

    def _fuse_results(
        self,
        results: list[ScoredMemory],
    ) -> list[ScoredMemory]:
        """Fuse and deduplicate results from multiple search strategies.

        When two results share the same ``memory_id``, the one with the
        higher ``vector_score`` is kept.

        Args:
            results: Combined list of scored memories.

        Returns:
            Deduplicated list.
        """
        seen: set[str] = set()
        fused: list[ScoredMemory] = []

        # Sort: vector results first (higher priority)
        results.sort(key=lambda x: x.vector_score, reverse=True)

        for mem in results:
            if mem.memory_id in seen:
                continue
            seen.add(mem.memory_id)
            fused.append(mem)

        return fused
