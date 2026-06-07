"""Generic async CRUD base repository."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

DocumentT = TypeVar("DocumentT", bound=BaseModel)


class BaseRepository(Generic[DocumentT]):
    """Generic CRUD repository for Motor-backed Pydantic models.

    Args:
        collection: An ``AsyncIOMotorCollection`` instance.
        model_class: The Pydantic model class for documents in this collection.
    """

    def __init__(
        self,
        collection: AsyncIOMotorCollection,
        model_class: type[DocumentT],
    ) -> None:
        self._collection = collection
        self._model_class = model_class

    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Return the underlying Motor collection."""
        return self._collection

    async def create(self, document: DocumentT) -> DocumentT:
        """Insert *document* and return it with its generated ``_id``.

        Args:
            document: The Pydantic model instance to insert.

        Returns:
            The document with the ``_id`` populated.
        """
        data = document.model_dump(by_alias=True, exclude_none=False)
        if "_id" in data and data["_id"] is None:
            del data["_id"]
        result = await self._collection.insert_one(data)
        data["_id"] = result.inserted_id
        return self._model_class.model_validate(data)

    async def get(self, doc_id: str) -> DocumentT | None:
        """Retrieve a single document by its ``_id``.

        Args:
            doc_id: The document ``_id`` as a string.

        Returns:
            The document or ``None`` if not found.
        """
        from bson import ObjectId

        # Try with ObjectId first, then fall back to plain string _id
        doc = await self._collection.find_one({"_id": ObjectId(doc_id)})
        if doc is None:
            doc = await self._collection.find_one({"_id": doc_id})
        if doc is None:
            return None
        return self._model_class.model_validate(doc)

    async def get_by_field(self, field: str, value: Any) -> DocumentT | None:
        """Retrieve a single document by an arbitrary field match.

        Args:
            field: Field name.
            value: Value to match.

        Returns:
            The document or ``None``.
        """
        doc = await self._collection.find_one({field: value})
        if doc is None:
            return None
        return self._model_class.model_validate(doc)

    async def update(self, doc_id: str, updates: dict[str, Any]) -> DocumentT | None:
        """Update a document by ``_id`` with the given ``$set`` fields.

        Args:
            doc_id: The document ``_id`` as a string.
            updates: A dict of field→value pairs to set.

        Returns:
            The updated document or ``None`` if not found.
        """
        from bson import ObjectId

        updates["updated_at"] = datetime.utcnow()
        # Try with ObjectId first, then fall back to plain string _id
        result = await self._collection.find_one_and_update(
            {"_id": ObjectId(doc_id)},
            {"$set": updates},
            return_document=True,
        )
        if result is None:
            result = await self._collection.find_one_and_update(
                {"_id": doc_id},
                {"$set": updates},
                return_document=True,
            )
        if result is None:
            return None
        return self._model_class.model_validate(result)

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ``_id``.

        Args:
            doc_id: The document ``_id`` as a string.

        Returns:
            ``True`` if a document was deleted, ``False`` otherwise.
        """
        from bson import ObjectId

        result = await self._collection.delete_one({"_id": ObjectId(doc_id)})
        if result.deleted_count == 0:
            result = await self._collection.delete_one({"_id": doc_id})
        return result.deleted_count > 0

    async def find(
        self,
        filter: dict[str, Any],
        sort: list[tuple[str, int]] | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[DocumentT]:
        """Find documents matching *filter* with optional sort, limit, skip.

        Args:
            filter: MongoDB query filter.
            sort: List of ``(field, direction)`` tuples (e.g. ``[("created_at", -1)]``).
            limit: Maximum number of documents to return.
            skip: Number of documents to skip.

        Returns:
            A list of documents.
        """
        cursor = self._collection.find(filter)
        if sort:
            cursor = cursor.sort(sort)
        cursor = cursor.skip(skip).limit(limit)
        results = await cursor.to_list(length=limit)
        return [self._model_class.model_validate(doc) for doc in results]

    async def count(self, filter: dict[str, Any]) -> int:
        """Count documents matching *filter*.

        Args:
            filter: MongoDB query filter.

        Returns:
            Document count.
        """
        return await self._collection.count_documents(filter)

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run an aggregation pipeline.

        Args:
            pipeline: List of aggregation stages.

        Returns:
            Aggregation results as raw dicts.
        """
        cursor = self._collection.aggregate(pipeline)
        return await cursor.to_list(length=None)
