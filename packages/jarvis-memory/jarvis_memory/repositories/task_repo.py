"""Task repository — domain-specific queries."""

from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection

from jarvis_memory.models.task import TaskDocument
from jarvis_memory.repositories.base import BaseRepository


class TaskRepository(BaseRepository[TaskDocument]):
    """Repository for ``tasks`` collection."""

    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        super().__init__(collection, TaskDocument)

    async def get_user_tasks(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[TaskDocument]:
        """Return tasks for a user, optionally filtered by status.

        Args:
            user_id: The user's external ID.
            status: Optional status filter (e.g. "pending").
            limit: Maximum number of results.
            skip: Number to skip for pagination.

        Returns:
            List of task documents.
        """
        filter: dict[str, Any] = {"user_id": user_id}
        if status:
            filter["status"] = status
        return await self.find(
            filter=filter,
            sort=[("priority", -1), ("due_at", 1)],
            limit=limit,
            skip=skip,
        )

    async def get_pending_tasks_by_priority(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[TaskDocument]:
        """Return pending tasks sorted by priority (highest first).

        Args:
            user_id: The user's external ID.
            limit: Maximum number of results.

        Returns:
            List of pending task documents.
        """
        return await self.find(
            filter={"user_id": user_id, "status": "pending"},
            sort=[("priority", -1), ("due_at", 1)],
            limit=limit,
        )

    async def get_recurring_tasks(
        self,
        user_id: str,
        status: str | None = None,
    ) -> list[TaskDocument]:
        """Return recurring tasks for a user.

        Args:
            user_id: The user's external ID.
            status: Optional status filter.

        Returns:
            List of recurring task documents.
        """
        filter: dict[str, Any] = {
            "user_id": user_id,
            "task_type": {"$in": ["recurring", "recurring_template"]},
        }
        if status:
            filter["status"] = status
        return await self.find(filter=filter, limit=100)

    async def get_completed_tasks(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[TaskDocument]:
        """Return recently completed tasks for a user.

        Args:
            user_id: The user's external ID.
            limit: Maximum number of results.

        Returns:
            List of completed task documents.
        """
        return await self.find(
            filter={"user_id": user_id, "status": "completed"},
            sort=[("completed_at", -1)],
            limit=limit,
        )
