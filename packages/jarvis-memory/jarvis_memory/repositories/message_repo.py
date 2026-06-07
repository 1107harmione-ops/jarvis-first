"""Message repository — domain-specific queries."""

from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorCollection

from jarvis_memory.models.message import MessageDocument
from jarvis_memory.repositories.base import BaseRepository


class MessageRepository(BaseRepository[MessageDocument]):
    """Repository for ``messages`` collection."""

    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        super().__init__(collection, MessageDocument)

    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        skip: int = 0,
    ) -> list[MessageDocument]:
        """Return messages in a conversation ordered by timestamp.

        Args:
            conversation_id: The conversation ID.
            limit: Maximum number of messages.
            skip: Number to skip for pagination.

        Returns:
            List of messages.
        """
        return await self.find(
            filter={"conversation_id": conversation_id},
            sort=[("timestamp", 1)],
            limit=limit,
            skip=skip,
        )

    async def get_user_messages(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0,
    ) -> list[MessageDocument]:
        """Return the most recent messages for a user.

        Args:
            user_id: The user's external ID.
            limit: Maximum number of messages.
            skip: Number to skip for pagination.

        Returns:
            List of messages.
        """
        return await self.find(
            filter={"user_id": user_id},
            sort=[("timestamp", -1)],
            limit=limit,
            skip=skip,
        )

    async def get_important_messages(
        self,
        user_id: str,
        min_score: float = 0.5,
        limit: int = 20,
    ) -> list[MessageDocument]:
        """Return high-importance messages for a user.

        Args:
            user_id: The user's external ID.
            min_score: Minimum importance score threshold.
            limit: Maximum number of messages.

        Returns:
            List of messages.
        """
        return await self.find(
            filter={"user_id": user_id, "importance_score": {"$gte": min_score}},
            sort=[("importance_score", -1)],
            limit=limit,
        )

    async def count_by_intent(
        self,
        intent: str,
        since: datetime | None = None,
    ) -> int:
        """Count messages matching a given intent, optionally since a date.

        Args:
            intent: Intent string to match.
            since: Optional start datetime.

        Returns:
            Message count.
        """
        filter: dict = {"intent": intent}
        if since:
            filter["timestamp"] = {"$gte": since}
        return await self.count(filter)
