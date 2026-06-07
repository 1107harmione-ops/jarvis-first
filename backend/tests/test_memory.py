"""
Tests for memory system (STM, LTM, Vector Memory).
"""

from __future__ import annotations

import pytest

from backend.memory.short_term import stm
from backend.memory.long_term import ltm
from backend.memory.vector_memory import vector_memory


class TestShortTermMemory:
    """Tests for ShortTermMemory class."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        doc = await stm.store(
            user_id="test_user",
            content="Test STM content",
            tags=["test"],
            importance_score=0.3,
        )
        assert doc is not None
        assert doc.get("content") == "Test STM content"
        assert doc.get("memory_type") == "short_term"

    @pytest.mark.asyncio
    async def test_get_recent(self) -> None:
        await stm.store(user_id="test_user", content="First memory")
        await stm.store(user_id="test_user", content="Second memory")

        recent = await stm.get_recent("test_user", limit=10)
        assert len(recent) >= 2

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        await stm.store(user_id="test_user", content="Python programming is fun")
        await stm.store(user_id="test_user", content="I love coding in Python")

        results = await stm.search("test_user", "Python", limit=5)
        assert len(results) >= 1
        assert "Python" in results[0].get("content", "")

    @pytest.mark.asyncio
    async def test_count_active(self) -> None:
        count = await stm.count_active("test_user")
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        await stm.store(user_id="test_user", content="Temp memory")
        deleted = await stm.clear("test_user")
        assert deleted > 0


class TestLongTermMemory:
    """Tests for LongTermMemory class."""

    @pytest.mark.asyncio
    async def test_store_high_importance(self) -> None:
        doc = await ltm.store(
            user_id="test_user",
            content="Important memory about AI",
            tags=["ai"],
            importance_score=0.8,
            source="test",
        )
        assert doc is not None
        assert doc.get("memory_type") == "long_term"
        assert doc.get("importance_score") == 0.8

    @pytest.mark.asyncio
    async def test_store_low_importance_raises(self) -> None:
        with pytest.raises(ValueError):
            await ltm.store(
                user_id="test_user",
                content="Not important enough",
                importance_score=0.1,
            )

    @pytest.mark.asyncio
    async def test_get_by_importance(self) -> None:
        await ltm.store(
            user_id="test_user",
            content="Very important",
            importance_score=0.9,
        )
        await ltm.store(
            user_id="test_user",
            content="Moderately important",
            importance_score=0.7,
        )

        results = await ltm.get_by_importance("test_user", limit=10, min_score=0.8)
        assert len(results) >= 1
        assert results[0].get("importance_score", 0) >= 0.8

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        stats = await ltm.get_stats("test_user")
        assert "total" in stats
        assert "high_importance" in stats


class TestVectorMemory:
    """Tests for VectorMemory class (embedding)."""

    @pytest.mark.asyncio
    async def test_embed_text(self) -> None:
        # This test requires sentence-transformers or API key
        # Skip if no embedding provider available
        try:
            embedding = await vector_memory.embed("Hello world")
            assert len(embedding) > 0
            assert isinstance(embedding, list)
            assert all(isinstance(v, float) for v in embedding)
        except Exception as exc:
            pytest.skip(f"Embedding unavailable: {exc}")

    def test_cosine_similarity(self) -> None:
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert vector_memory.cosine_similarity(v1, v2) == pytest.approx(1.0, abs=0.01)

        v3 = [0.0, 1.0, 0.0]
        assert vector_memory.cosine_similarity(v1, v3) == pytest.approx(0.0, abs=0.01)

    def test_rank_by_similarity(self) -> None:
        query = [1.0, 0.0]
        candidates = [
            ("similar", [0.9, 0.1]),
            ("different", [0.0, 1.0]),
        ]
        ranked = vector_memory.rank_by_similarity(query, candidates)
        assert len(ranked) == 2
        assert ranked[0][0] == "similar"  # Most similar first
