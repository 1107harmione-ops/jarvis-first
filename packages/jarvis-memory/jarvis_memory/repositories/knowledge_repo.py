"""Knowledge repository — domain-specific queries."""

from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from jarvis_memory.models.knowledge import KnowledgeDocument
from jarvis_memory.repositories.base import BaseRepository


class KnowledgeRepository(BaseRepository[KnowledgeDocument]):
    """Repository for ``knowledge`` collection."""

    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        super().__init__(collection, KnowledgeDocument)

    async def find_by_source(
        self,
        source_type: str,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[KnowledgeDocument]:
        """Return knowledge entries filtered by source type.

        Args:
            source_type: Type of knowledge source.
            user_id: Optional user filter. ``None`` returns global knowledge.
            limit: Maximum number of results.

        Returns:
            List of knowledge documents.
        """
        filter: dict[str, Any] = {"source_type": source_type}
        if user_id is not None:
            filter["user_id"] = user_id
        return await self.find(filter=filter, sort=[("created_at", -1)], limit=limit)

    async def find_by_embedding(
        self,
        embedding: list[float],
        user_id: str | None = None,
        top_k: int = 10,
    ) -> list[KnowledgeDocument]:
        """Vector-similarity search for knowledge documents.

        Args:
            embedding: 384-dim query embedding.
            user_id: Optional user filter.
            top_k: Maximum results.

        Returns:
            List of knowledge documents.
        """
        pipeline: list[dict[str, Any]] = []

        pre_filter: dict[str, Any] = {}
        if user_id is not None:
            pre_filter["user_id"] = user_id

        pipeline.append({
            "$vectorSearch": {
                "index": "knowledge_vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": top_k * 4,
                "limit": top_k,
                "filter": pre_filter,
            }
        })

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
            logger.warning("Vector search failed (M10+ cluster required); falling back to recent documents")
            # Fallback: return recent documents sorted by creation date
            return await self.find(
                filter={"user_id": user_id} if user_id else {},
                sort=[("created_at", -1)],
                limit=top_k,
            )
