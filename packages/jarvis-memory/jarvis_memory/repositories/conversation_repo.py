"""Conversation repository — domain-specific queries."""

from motor.motor_asyncio import AsyncIOMotorCollection

from jarvis_memory.models.conversation import ConversationDocument
from jarvis_memory.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[ConversationDocument]):
    """Repository for ``conversations`` collection."""

    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        super().__init__(collection, ConversationDocument)

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0,
    ) -> list[ConversationDocument]:
        """Return the most recent conversations for a user.

        Args:
            user_id: The user's external ID.
            limit: Maximum number of conversations.
            skip: Number to skip for pagination.

        Returns:
            List of conversation documents.
        """
        return await self.find(
            filter={"user_id": user_id},
            sort=[("updated_at", -1)],
            limit=limit,
            skip=skip,
        )

    async def add_message_count(
        self,
        conv_id: str,
        tokens: int = 0,
    ) -> ConversationDocument | None:
        """Atomically increment ``message_count`` and ``tokens_used``.

        Args:
            conv_id: The conversation ``_id`` as a string.
            tokens: Number of tokens to add.

        Returns:
            The updated conversation or ``None``.
        """
        from bson import ObjectId

        result = await self.collection.find_one_and_update(
            {"_id": ObjectId(conv_id)},
            {"$inc": {"message_count": 1, "tokens_used": tokens}},
            return_document=True,
        )
        if result is None:
            return None
        return self._model_class.model_validate(result)
