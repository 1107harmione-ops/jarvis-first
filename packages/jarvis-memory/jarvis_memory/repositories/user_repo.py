"""User repository — domain-specific queries."""

from motor.motor_asyncio import AsyncIOMotorCollection

from jarvis_memory.models.user import UserDocument
from jarvis_memory.repositories.base import BaseRepository


class UserRepository(BaseRepository[UserDocument]):
    """Repository for ``users`` collection."""

    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        super().__init__(collection, UserDocument)

    async def find_by_email(self, email: str) -> UserDocument | None:
        """Return a user by email address.

        Args:
            email: Email address to look up.

        Returns:
            The user document or ``None``.
        """
        return await self.get_by_field("email", email)

    async def update_preferences(
        self,
        user_id: str,
        preferences: dict,
    ) -> UserDocument | None:
        """Update a user's preferences.

        Args:
            user_id: The ``user_id`` field (external ID).
            preferences: Preference key→value pairs.

        Returns:
            The updated user document or ``None``.
        """
        return await self.update(user_id, {"preferences": preferences})

    async def update_voice_settings(
        self,
        user_id: str,
        voice_settings: dict,
    ) -> UserDocument | None:
        """Update a user's voice settings.

        Args:
            user_id: The ``user_id`` field (external ID).
            voice_settings: Voice setting key→value pairs.

        Returns:
            The updated user document or ``None``.
        """
        return await self.update(user_id, {"voice_settings": voice_settings})
