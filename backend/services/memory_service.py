"""
Memory Service — high-level memory operations for the API layer.
Coordinates STM, LTM, and vector memory into a unified interface.
"""

from __future__ import annotations

import time
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_memory_doc, serialize_doc
from backend.memory.long_term import ltm
from backend.memory.short_term import stm
from backend.memory.vector_memory import vector_memory
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryService:
    """Coordinated memory service combining STM, LTM, and vector search."""

    async def store(
        self,
        user_id: str,
        content: str,
        memory_type: str = "short_term",
        tags: list[str] | None = None,
        importance_score: float = 0.5,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a memory, routing to STM or LTM based on importance."""
        if memory_type == "long_term" or importance_score >= 0.6:
            # Generate embedding for LTM
            try:
                embedding = await vector_memory.embed(content)
            except Exception:
                embedding = None

            try:
                return await ltm.store(
                    user_id=user_id,
                    content=content,
                    embedding=embedding,
                    tags=tags,
                    importance_score=importance_score,
                    source=source,
                    metadata=metadata,
                )
            except ValueError:
                # Below threshold, store as STM
                pass

        return await stm.store(
            user_id=user_id,
            content=content,
            tags=tags,
            importance_score=importance_score,
            metadata=metadata,
        )

    async def search(
        self,
        user_id: str,
        query: str,
        memory_type: str | None = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Search across all memory types using semantic + text search.

        Performs vector search first, falls back to text search.
        """
        start = time.monotonic()
        results: list[dict[str, Any]] = []

        # Try vector search
        try:
            query_embedding = await vector_memory.embed(query)
            ltm_results = await ltm.semantic_search(
                user_id, query_embedding, limit=limit, threshold=threshold
            )
            results.extend(ltm_results)
        except Exception as exc:
            logger.debug("Vector search unavailable, using text search", extra={"error": str(exc)})

        # Supplement with STM text search
        try:
            stm_results = await stm.search(user_id, query, limit=limit)
            # Deduplicate by content
            existing_contents = {r.get("content", "") for r in results}
            for r in stm_results:
                if r.get("content", "") not in existing_contents:
                    results.append(r)
        except Exception as exc:
            logger.debug("STM search failed", extra={"error": str(exc)})

        # Apply memory_type filter if specified
        if memory_type:
            results = [r for r in results if r.get("memory_type") == memory_type]

        # Sort by importance score
        results.sort(key=lambda x: x.get("importance_score", 0), reverse=True)

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Memory search completed",
            extra={"user_id": user_id, "results": len(results), "duration_ms": f"{elapsed:.1f}"},
        )

        return results[:limit]

    async def get_context(
        self, user_id: str, query: str | None = None, max_tokens: int = 4096
    ) -> str:
        """Build a context string from relevant memories.

        Used by the router agent to inject memory context into conversations.
        """
        parts: list[str] = []

        # Recent STM
        try:
            recent = await stm.get_context_window(user_id, max_items=10)
            if recent:
                parts.append("## Recent Interactions")
                for m in recent:
                    parts.append(f"- {m.get('content', '')[:200]}")
        except Exception:
            pass

        # Relevant LTM (if query provided)
        if query:
            try:
                embedding = await vector_memory.embed(query)
                memories = await ltm.semantic_search(user_id, embedding, limit=5)
                if memories:
                    parts.append("## Relevant Memories")
                    for m in memories:
                        parts.append(f"- [{m.get('importance_score', 0):.1f}] {m.get('content', '')[:200]}")
            except Exception:
                pass

        context = "\n".join(parts)
        # Rough token estimate (4 chars ≈ 1 token)
        if len(context) > max_tokens * 4:
            context = context[: max_tokens * 4] + "\n[context truncated...]"

        return context

    async def consolidate(self, user_id: str) -> int:
        """Trigger STM → LTM consolidation."""
        return await ltm.consolidate_from_stm(
            user_id,
            embedding_func=vector_memory.embed,
        )

    async def get_stats(self, user_id: str) -> dict[str, Any]:
        """Get memory statistics."""
        stm_count = await stm.count_active(user_id)
        ltm_stats = await ltm.get_stats(user_id)
        return {
            "short_term": stm_count,
            "long_term": ltm_stats.get("total", 0),
            "high_importance": ltm_stats.get("high_importance", 0),
        }

    async def clear_stm(self, user_id: str) -> int:
        """Clear all short-term memory for a user."""
        return await stm.clear(user_id)

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific memory."""
        result = await mongodb.memories.delete_one({"_id": memory_id, "user_id": user_id})
        return result.deleted_count > 0


# Global singleton
memory_service = MemoryService()
