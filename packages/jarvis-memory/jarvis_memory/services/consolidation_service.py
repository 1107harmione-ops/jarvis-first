"""STM → LTM consolidation service.

Promotes high-importance short-term memories to long-term storage with
deduplication and optional summarization.
"""

import logging
from datetime import datetime
from typing import Any

from jarvis_memory.models.memory import MemoryDocument
from jarvis_memory.repositories.memory_repo import MemoryRepository
from jarvis_memory.services.embedding_service import EmbeddingService
from jarvis_memory.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)


class ConsolidationService:
    """Manages the lifecycle transition of STM memories to LTM.

    Args:
        memory_repo: Repository for the ``memories`` collection.
        embedding_service: Service for generating embeddings.
        scoring_service: Service for computing importance scores.
        min_importance: Minimum importance threshold for promotion.
        age_hours: Minimum age in hours before a STM is considered
            for consolidation.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        embedding_service: EmbeddingService,
        scoring_service: ScoringService,
        min_importance: float = 0.6,
        age_hours: float = 1.0,
    ) -> None:
        self._repo = memory_repo
        self._embedder = embedding_service
        self._scoring = scoring_service
        self._min_importance = min_importance
        self._age_hours = age_hours

    async def consolidate(self, user_id: str) -> int:
        """Run consolidation for a given user.

        Finds STM candidates, deduplicates against existing LTM, optionally
        summarizes related entries, and promotes qualifying memories.

        Args:
            user_id: The user's external ID.

        Returns:
            Number of memories successfully promoted to LTM.
        """
        candidates = await self._find_consolidation_candidates(user_id)
        if not candidates:
            logger.info("No consolidation candidates for user %s", user_id)
            return 0

        promoted_count = 0
        for stm in candidates:
            # Skip if duplicate content already exists in LTM
            is_duplicate = await self._deduplicate(stm.content, user_id)
            if is_duplicate:
                # Update access on the existing LTM instead
                logger.debug(
                    "STM %s is duplicate of existing LTM; skipping",
                    stm.memory_id,
                )
                continue

            # Promote to LTM
            try:
                await self._promote_to_ltm(stm)
                promoted_count += 1
            except Exception as exc:
                logger.error(
                    "Failed to promote STM %s: %s",
                    stm.memory_id,
                    exc,
                )

        logger.info(
            "Consolidation complete for user %s: %d promoted out of %d candidates",
            user_id,
            promoted_count,
            len(candidates),
        )
        return promoted_count

    async def _find_consolidation_candidates(
        self,
        user_id: str,
    ) -> list[MemoryDocument]:
        """Find STM entries eligible for promotion.

        Returns STM entries that are:
        - Older than ``age_hours``
        - Above ``min_importance`` threshold
        - Not yet consolidated

        Args:
            user_id: The user's external ID.

        Returns:
            List of candidate memory documents.
        """
        return await self._repo.get_consolidation_candidates(
            user_id,
            min_importance=self._min_importance,
            age_hours=self._age_hours,
        )

    async def _promote_to_ltm(self, stm_memory: MemoryDocument) -> MemoryDocument:
        """Promote a single STM memory to LTM.

        Creates a new LTM entry referencing the original STM via
        ``source_id``, then marks the STM as consolidated.

        Args:
            stm_memory: The STM memory to promote.

        Returns:
            The newly created LTM ``MemoryDocument``.
        """
        # Generate embedding if not present
        embedding = stm_memory.embedding
        if embedding is None:
            embedding = await self._embedder.embed(stm_memory.content)

        # Compute a fresh importance score
        importance = self._scoring.compute_importance(
            stm_memory.content,
            tags=stm_memory.tags,
            metadata=stm_memory.metadata,
        )

        ltm = MemoryDocument(
            user_id=stm_memory.user_id,
            memory_type="long_term",
            content=stm_memory.content,
            embedding=embedding,
            importance_score=importance,
            summary=stm_memory.summary,
            tags=stm_memory.tags,
            source=stm_memory.source,
            source_id=stm_memory.memory_id,
            context=stm_memory.context,
            metadata={
                **(stm_memory.metadata or {}),
                "consolidated_from": stm_memory.memory_id,
                "consolidated_at": datetime.utcnow().isoformat(),
            },
            consolidated=False,
            expires_at=None,  # LTM does not expire
        )

        created = await self._repo.create(ltm)

        # Mark the STM as consolidated
        await self._repo.update(
            stm_memory.id,
            {
                "consolidated": True,
                "memory_type": "long_term",
                "updated_at": datetime.utcnow(),
            },
        )

        logger.info(
            "Promoted STM %s to LTM %s (importance=%.3f)",
            stm_memory.memory_id,
            created.memory_id,
            importance,
        )
        return created

    async def _summarize_group(
        self,
        memories: list[MemoryDocument],
    ) -> str:
        """Summarize a group of related STM entries into a single string.

        A basic heuristic: concatenate unique content lines.
        In production this would call an LLM.

        Args:
            memories: List of related STM memory documents.

        Returns:
            A summary string.
        """
        seen: set[str] = set()
        lines: list[str] = []
        for mem in memories:
            text = mem.content.strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(text)
        return " | ".join(lines)

    async def _deduplicate(
        self,
        content: str,
        user_id: str,
    ) -> bool:
        """Check if *content* already exists as LTM for *user_id*.

        Uses simple substring matching and embedding similarity.

        Args:
            content: Content text to check.
            user_id: The user's external ID.

        Returns:
            ``True`` if a duplicate already exists.
        """
        # Quick check: exact or near-exact match in LTM
        existing = await self._repo.find(
            filter={
                "user_id": user_id,
                "memory_type": "long_term",
                "content": content,
            },
            limit=1,
        )
        if existing:
            return True

        # Check via embedding similarity
        embedding = await self._embedder.embed(content)
        similar = await self._repo.find_by_embedding(
            embedding=embedding,
            user_id=user_id,
            memory_types=["long_term"],
            top_k=3,
        )

        for candidate in similar:
            if candidate.embedding:
                sim = await self._embedder.cosine_similarity(
                    embedding, candidate.embedding
                )
                if sim > 0.92:  # High similarity threshold
                    logger.debug(
                        "Deduplicate: content similar (%.4f) to LTM %s",
                        sim,
                        candidate.memory_id,
                    )
                    return True

        return False
