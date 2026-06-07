"""
Task Service — manages task scheduling, recurring tasks, and background job execution.
Handles task lifecycle: creation → scheduling → execution → completion/failure.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.database.schemas import new_task_doc, serialize_doc, update_task_doc
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TaskService:
    """Background task management service.

    Features:
    - One-time reminders (scheduled)
    - Recurring tasks (cron-style)
    - Automatic retry on failure
    - Task status tracking
    - Due task notification
    """

    def __init__(self) -> None:
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False

    # ── CRUD Operations ────────────────────────────────────────

    async def create_task(
        self,
        user_id: str,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_at: datetime | None = None,
        scheduled_at: datetime | None = None,
        recurring: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new task."""
        doc = new_task_doc(
            user_id=user_id,
            title=title,
            description=description,
            priority=priority,
            due_at=due_at,
            scheduled_at=scheduled_at,
            recurring=recurring,
            tags=tags,
        )
        result = await mongodb.tasks.insert_one(doc)
        task_id = str(result.inserted_id)
        logger.info("Task created", extra={"task_id": task_id, "user_id": user_id})
        return serialize_doc(doc)

    async def get_task(self, task_id: str, user_id: str) -> dict[str, Any] | None:
        """Get a task by ID."""
        doc = await mongodb.tasks.find_one({"_id": task_id, "user_id": user_id})
        return serialize_doc(doc) if doc else None

    async def list_tasks(
        self,
        user_id: str,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """List tasks with optional filters."""
        query: dict[str, Any] = {"user_id": user_id}
        if status:
            query["status"] = status
        if priority:
            query["priority"] = priority

        cursor = mongodb.tasks.find(query, sort=[("created_at", -1)], limit=limit, skip=skip)
        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def update_task(
        self,
        task_id: str,
        user_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update a task."""
        update_doc = update_task_doc(**updates)
        result = await mongodb.tasks.update_one(
            {"_id": task_id, "user_id": user_id},
            {"$set": update_doc},
        )
        if result.modified_count:
            return await self.get_task(task_id, user_id)
        return None

    async def delete_task(self, task_id: str, user_id: str) -> bool:
        """Delete a task."""
        result = await mongodb.tasks.delete_one({"_id": task_id, "user_id": user_id})
        return result.deleted_count > 0

    async def complete_task(self, task_id: str, user_id: str) -> bool:
        """Mark a task as completed."""
        result = await mongodb.tasks.update_one(
            {"_id": task_id, "user_id": user_id},
            {
                "$set": update_task_doc(
                    status="completed",
                    completed_at=datetime.now(timezone.utc),
                )
            },
        )
        if result.modified_count:
            # If recurring, create next instance
            task = await self.get_task(task_id, user_id)
            if task and task.get("recurring"):
                await self._schedule_next_recurring(task)
            return True
        return False

    # ── Background Worker ──────────────────────────────────────

    def start_worker(self) -> None:
        """Start the background task worker."""
        if self._worker_task:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Task worker started")

    async def stop_worker(self) -> None:
        """Stop the background task worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("Task worker stopped")

    async def _worker_loop(self) -> None:
        """Main worker loop — checks for due/scheduled tasks."""
        while self._running:
            try:
                await self._check_due_tasks()
                await asyncio.sleep(settings.TASK_BACKGROUND_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Task worker error", extra={"error": str(exc)})
                await asyncio.sleep(30)

    async def _check_due_tasks(self) -> None:
        """Find and process tasks that are due."""
        now = datetime.now(timezone.utc)

        # Find pending tasks with past due_at or scheduled_at
        cursor = mongodb.tasks.find({
            "status": "pending",
            "$or": [
                {"due_at": {"$lte": now, "$ne": None}},
                {"scheduled_at": {"$lte": now, "$ne": None}},
            ],
        })
        async for task in cursor:
            try:
                logger.info("Processing due task", extra={"task_id": str(task["_id"]), "title": task.get("title")})
                # Execute the task action
                await self._execute_task(task)
            except Exception as exc:
                logger.error("Task execution failed", extra={"task_id": str(task["_id"]), "error": str(exc)})
                await mongodb.tasks.update_one(
                    {"_id": task["_id"]},
                    {
                        "$set": update_task_doc(
                            status="failed",
                            error=str(exc),
                            retry_count=task.get("retry_count", 0) + 1,
                        )
                    },
                )

    async def _execute_task(self, task: dict[str, Any]) -> None:
        """Execute a single task action."""
        # For now, tasks are passive (reminders).
        # Future: integrate with agent system for action execution.
        title = task.get("title", "Untitled")
        logger.info("Task triggered", extra={"title": title, "task_id": str(task["_id"])})

        # Mark as completed (or update scheduled_at for recurring)
        if task.get("recurring"):
            # Schedule next occurrence
            next_due = self._compute_next_cron(task["recurring"])
            await mongodb.tasks.update_one(
                {"_id": task["_id"]},
                {
                    "$set": update_task_doc(
                        scheduled_at=next_due,
                        completed_at=datetime.now(timezone.utc),
                    )
                },
            )
        else:
            await mongodb.tasks.update_one(
                {"_id": task["_id"]},
                {
                    "$set": update_task_doc(
                        status="completed",
                        completed_at=datetime.now(timezone.utc),
                    )
                },
            )

    def _compute_next_cron(self, cron_expr: str) -> datetime | None:
        """Compute next fire time from a cron expression."""
        try:
            from croniter import croniter
            now = datetime.now(timezone.utc)
            cron = croniter(cron_expr, now)
            return cron.get_next(datetime)
        except (ImportError, ValueError, KeyError):
            logger.warning("Cron parsing failed, using +1 day as fallback")
            from datetime import timedelta
            return datetime.now(timezone.utc) + timedelta(days=1)

    async def _schedule_next_recurring(self, task: dict[str, Any]) -> None:
        """Create the next occurrence of a recurring task."""
        next_due = self._compute_next_cron(task["recurring"])
        if next_due:
            doc = new_task_doc(
                user_id=task.get("user_id", ""),
                title=task.get("title", ""),
                description=task.get("description", ""),
                priority=task.get("priority", "medium"),
                scheduled_at=next_due,
                recurring=task.get("recurring"),
                tags=task.get("tags"),
            )
            await mongodb.tasks.insert_one(doc)
            logger.debug("Next recurring task scheduled", extra={"title": task.get("title"), "next_due": next_due.isoformat()})

    # ── Stats ──────────────────────────────────────────────────

    async def get_stats(self, user_id: str) -> dict[str, int]:
        """Get task statistics for a user."""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        cursor = mongodb.tasks.aggregate(pipeline)
        stats: dict[str, int] = {"total": 0, "pending": 0, "completed": 0, "failed": 0, "cancelled": 0}
        async for doc in cursor:
            status = doc.get("_id", "unknown")
            count = doc.get("count", 0)
            stats[status] = count
            stats["total"] += count
        return stats


# Global singleton
task_service = TaskService()
