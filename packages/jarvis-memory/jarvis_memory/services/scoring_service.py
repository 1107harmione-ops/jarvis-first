"""Importance scoring and ranking service.

Computes the composite score ``S`` for a memory:

    S = w₁·R + w₂·I + w₃·F + w₄·P + w₅·C

where R = recency, I = importance, F = frequency, P = preference,
C = relevance (semantic similarity to query).
"""

import math
from datetime import datetime, timedelta
from typing import Any

from jarvis_memory.services.embedding_service import EmbeddingService


class ScoringService:
    """Service that computes and ranks memory importance scores.

    Args:
        embedding_service: Service used for cosine similarity
            computations (relevance factor).
        decay_rate: Exponential decay rate λ for the recency factor.
            Default matches the spec (0.1).
    """

    WEIGHTS: dict[str, float] = {
        "recency": 0.25,
        "importance": 0.30,
        "frequency": 0.15,
        "preference": 0.20,
        "relevance": 0.10,
    }

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        decay_rate: float = 0.1,
    ) -> None:
        self._embedding_service = embedding_service
        self._decay_rate = decay_rate

    def compute_recency(self, last_accessed: datetime | None) -> float:
        """Compute recency score ``R`` using exponential decay.

        ``R = exp(-λ · Δt_hours / 24)``

        A memory decays to ~37 % of its recency after 10 days without access.

        Args:
            last_accessed: The last access datetime. ``None`` returns 0.

        Returns:
            Recency score in [0, 1].
        """
        if last_accessed is None:
            return 0.0
        delta = datetime.utcnow() - last_accessed
        hours = delta.total_seconds() / 3600.0
        return math.exp(-self._decay_rate * hours / 24.0)

    def compute_importance(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Compute explicit importance score ``I`` from content analysis.

        Uses heuristics:
        - Base score of 0.5 for any content.
        - Bonus for longer content (signals substance).
        - Bonus based on tag keywords.
        - Bonus from metadata hints.

        Args:
            content: The memory content text.
            tags: Optional list of tags.
            metadata: Optional metadata dict.

        Returns:
            Importance score in [0, 1].
        """
        score = 0.5

        # Content length bonus (up to +0.2)
        length_bonus = min(len(content) / 1000.0, 0.2)
        score += length_bonus

        # Tag keyword bonus (up to +0.15)
        high_importance_keywords = {
            "important", "critical", "urgent", "key", "vital",
            "essential", "priority", "remember", "never", "always",
        }
        if tags:
            tag_set = {t.lower() for t in tags}
            overlap = tag_set & high_importance_keywords
            score += min(len(overlap) * 0.05, 0.15)

        # Metadata bonus (up to +0.1)
        if metadata:
            explicit_score = metadata.get("importance")
            if isinstance(explicit_score, (int, float)):
                score = max(score, float(explicit_score))
            if metadata.get("user_rated"):
                score += 0.1

        # Content keyword bonus (up to +0.15)
        content_lower = content.lower()
        keywords_found = sum(
            1 for kw in high_importance_keywords if kw in content_lower
        )
        score += min(keywords_found * 0.03, 0.15)

        return min(score, 1.0)

    def compute_frequency(
        self,
        access_count: int,
        max_count: int = 100,
    ) -> float:
        """Compute frequency factor ``F``.

        ``F = min(access_count / max_count, 1.0)``

        Args:
            access_count: Number of times the memory was accessed.
            max_count: Normalization ceiling.

        Returns:
            Frequency score in [0, 1].
        """
        if max_count <= 0:
            return 0.0
        return min(access_count / max_count, 1.0)

    def compute_preference(
        self,
        memory_type: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Compute preference boost ``P``.

        ``user_preference`` type memories and preference-tagged entries
        get a high boost.

        Args:
            memory_type: The memory type string.
            tags: Optional tags.
            metadata: Optional metadata.

        Returns:
            Preference score in [0, 1].
        """
        if memory_type == "user_preference":
            return 1.0
        if tags:
            pref_tags = {"preference", "preferred", "favorite", "likes", "dislikes"}
            if pref_tags & {t.lower() for t in tags}:
                return 0.8
        if metadata and metadata.get("is_preference"):
            return 0.9
        return 0.3

    async def compute_relevance(
        self,
        memory_embedding: list[float] | None,
        query_embedding: list[float] | None,
    ) -> float:
        """Compute relevance factor ``C`` via cosine similarity.

        Args:
            memory_embedding: The memory's embedding vector.
            query_embedding: The query embedding vector.

        Returns:
            Relevance score in [0, 1] (negative similarities clamped to 0).
        """
        if not memory_embedding or not query_embedding:
            return 0.0
        if self._embedding_service is None:
            return 0.5  # Neutral when no embedding service
        sim = await self._embedding_service.cosine_similarity(
            memory_embedding, query_embedding
        )
        return max(sim, 0.0)

    async def compute_score(
        self,
        memory: dict[str, Any],
        query_embedding: list[float] | None = None,
    ) -> float:
        """Compute the fused importance score ``S`` for a memory dict.

        Args:
            memory: A dict with keys like ``last_accessed``, ``content``,
                ``tags``, ``metadata``, ``access_count``, ``memory_type``,
                ``embedding``.
            query_embedding: Optional query embedding for the relevance
                factor.

        Returns:
            Composite score ``S`` in [0, 1].
        """
        # R — Recency
        recency = self.compute_recency(memory.get("last_accessed"))

        # I — Importance
        importance = self.compute_importance(
            content=memory.get("content", ""),
            tags=memory.get("tags"),
            metadata=memory.get("metadata"),
        )

        # F — Frequency
        frequency = self.compute_frequency(
            access_count=memory.get("access_count", 0)
        )

        # P — Preference
        preference = self.compute_preference(
            memory_type=memory.get("memory_type", ""),
            tags=memory.get("tags"),
            metadata=memory.get("metadata"),
        )

        # C — Relevance
        relevance = await self.compute_relevance(
            memory_embedding=memory.get("embedding"),
            query_embedding=query_embedding,
        )

        score = (
            self.WEIGHTS["recency"] * recency
            + self.WEIGHTS["importance"] * importance
            + self.WEIGHTS["frequency"] * frequency
            + self.WEIGHTS["preference"] * preference
            + self.WEIGHTS["relevance"] * relevance
        )

        return min(score, 1.0)

    def normalize_factor(self, value: float, max_value: float) -> float:
        """Normalize *value* to [0, 1] given a known *max_value*.

        Args:
            value: Raw factor value.
            max_value: Maximum possible value.

        Returns:
            Normalized value in [0, 1].
        """
        if max_value <= 0:
            return 0.0
        return min(value / max_value, 1.0)
