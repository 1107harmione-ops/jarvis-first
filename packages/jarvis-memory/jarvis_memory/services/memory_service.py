"""Memory CRUD service with embedding generation and scoring."""

import logging
from datetime import datetime, timedelta
from typing import Any

from jarvis_memory.config import settings
from jarvis_memory.models.memory import MemoryDocument, MemoryType
from jarvis_memory.repositories.memory_repo import MemoryRepository
from jarvis_memory.services.embedding_service import EmbeddingService
from jarvis_memory.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)


class MemoryService:
    """High-level memory operations with lifecycle management.

    Args:
        memory_repo: Repository for the ``memories`` collection.
        embedding_service: Service for generating vector embeddings.
        scoring_service: Service for computing importance scores.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        embedding_service: EmbeddingService,
        scoring_service: ScoringService,
    ) -> None:
        self._repo = memory_repo
        self._embedder = embedding_service
        self._scoring = scoring_service

    async def create_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
        source_id: str | None = None,
    ) -> MemoryDocument:
        """Create a new memory with auto-generated embedding and score.

        Args:
            user_id: The user's external ID.
            memory_type: Type of memory (short_term, long_term, etc.).
            content: Memory text content.
            tags: Optional tags.
            metadata: Optional metadata dict.
            source: Source context (conversation, user_input, etc.).
            source_id: ID of source conversation/message.

        Returns:
            The created ``MemoryDocument``.
        """
        # Generate embedding
        embedding = await self._embedder.embed(content)

        # Compute importance score
        importance = self._scoring.compute_importance(
            content, tags=tags, metadata=metadata
        )

        # Compute expires_at for STM
        expires_at: datetime | None = None
        if memory_type == "short_term":
            expires_at = datetime.utcnow() + timedelta(
                hours=settings.MEMORY_STM_TTL_HOURS
            )

        memory = MemoryDocument(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance_score=importance,
            tags=tags or [],
            metadata=metadata or {},
            source=source,
            source_id=source_id,
            expires_at=expires_at,
        )

        created = await self._repo.create(memory)
        logger.info(
            "Created memory %s (type=%s, importance=%.3f) for user %s",
            created.memory_id,
            memory_type,
            importance,
            user_id,
        )
        return created

    async def get_memory(self, memory_id: str) -> MemoryDocument | None:
        """Retrieve a memory by its ID.

        Args:
            memory_id: The memory ``_id`` as a string.

        Returns:
            The ``MemoryDocument`` or ``None``.
        """
        memory = await self._repo.get(memory_id)
        if memory is not None:
            # Increment access count and update last_accessed
            await self._repo.increment_access(memory_id)
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
        return memory

    async def update_memory(
        self,
        memory_id: str,
        updates: dict[str, Any],
    ) -> MemoryDocument | None:
        """Update a memory and recompute embedding/score if content changed.

        Args:
            memory_id: The memory ``_id`` as a string.
            updates: Dict of fields to update.

        Returns:
            The updated ``MemoryDocument`` or ``None``.
        """
        # If content changed, re-embed and re-score
        if "content" in updates:
            content = updates["content"]
            updates["embedding"] = await self._embedder.embed(content)
            updates["importance_score"] = self._scoring.compute_importance(
                content,
                tags=updates.get("tags"),
                metadata=updates.get("metadata"),
            )

        # If type changes to short_term, set expires_at
        if updates.get("memory_type") == "short_term":
            updates["expires_at"] = datetime.utcnow() + timedelta(
                hours=settings.MEMORY_STM_TTL_HOURS
            )

        updated = await self._repo.update(memory_id, updates)
        if updated:
            logger.info(
                "Updated memory %s (importance=%.3f)",
                memory_id,
                updated.importance_score,
            )
        return updated

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its ID.

        Args:
            memory_id: The memory ``_id`` as a string.

        Returns:
            ``True`` if deleted, ``False`` otherwise.
        """
        deleted = await self._repo.delete(memory_id)
        if deleted:
            logger.info("Deleted memory %s", memory_id)
        return deleted

    async def get_user_memories(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 50,
    ) -> list[MemoryDocument]:
        """Return memories for a user, optionally filtered by type.

        Args:
            user_id: The user's external ID.
            memory_type: Optional memory type filter.
            limit: Maximum number of results.

        Returns:
            List of ``MemoryDocument`` instances.
        """
        return await self._repo.search_by_user_type(
            user_id, memory_type=memory_type, limit=limit
        )

    async def search_memories(
        self,
        user_id: str,
        query: str,
        memory_types: list[MemoryType] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories by hybrid vector + keyword matching.

        Args:
            user_id: The user's external ID.
            query: Text query to search for.
            memory_types: Optional list of memory types to restrict to.
            top_k: Maximum results.

        Returns:
            List of scored memory dicts with ``memory`` and ``score`` keys.
        """
        query_embedding = await self._embedder.embed(query)

        # Vector search
        memories = await self._repo.find_by_embedding(
            embedding=query_embedding,
            user_id=user_id,
            memory_types=memory_types,
            top_k=top_k * 2,
        )

        # Score each result
        results: list[dict[str, Any]] = []
        for mem in memories:
            mem_dict = mem.model_dump()
            score = await self._scoring.compute_score(
                mem_dict, query_embedding=query_embedding
            )
            results.append({
                "memory": mem_dict,
                "score": round(score, 4),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
