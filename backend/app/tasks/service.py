"""Task business logic layer."""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.logger import get_logger
from app.database.models import Task
from app.tasks.schemas import TaskCreate, TaskUpdate

logger = get_logger(__name__)


class TaskService:
    """CRUD operations for tasks."""

    async def create(self, db: AsyncSession, data: TaskCreate) -> Task:
        """Create a new task."""
        task = Task(
            title=data.title,
            description=data.description,
            priority=data.priority,
            due_date=data.due_date,
            tags=data.tags,
            category=data.category,
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        logger.info("task_created", task_id=task.id, title=task.title)
        return task

    async def get(self, db: AsyncSession, task_id: int) -> Task:
        """Get a task by ID."""
        task = await db.get(Task, task_id)
        if not task:
            raise NotFoundError("Task", task_id)
        return task

    async def list(
        self,
        db: AsyncSession,
        status: Optional[str] = None,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Task], int]:
        """List tasks with optional filters and search."""
        query = select(Task)

        if status:
            query = query.where(Task.status == status)
        if category:
            query = query.where(Task.category == category)
        if priority:
            query = query.where(Task.priority == priority)
        if search:
            search_term = f"%{search}%"
            query = query.where(
                Task.title.ilike(search_term)
                | Task.description.ilike(search_term)
                | Task.tags.ilike(search_term)
            )

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply ordering, pagination
        query = query.order_by(Task.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        tasks = list(result.scalars().all())

        return tasks, total

    async def update(self, db: AsyncSession, task_id: int, data: TaskUpdate) -> Task:
        """Update a task."""
        task = await self.get(db, task_id)
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)
        task.updated_at = datetime.datetime.now()
        await db.commit()
        await db.refresh(task)
        logger.info("task_updated", task_id=task.id, changes=update_data)
        return task

    async def complete(self, db: AsyncSession, task_id: int) -> Task:
        """Mark a task as completed."""
        return await self.update(db, task_id, TaskUpdate(status="completed"))

    async def delete(self, db: AsyncSession, task_id: int) -> None:
        """Delete a task."""
        task = await self.get(db, task_id)
        await db.delete(task)
        await db.commit()
        logger.info("task_deleted", task_id=task_id)

    async def search(
        self, db: AsyncSession, query_str: str, limit: int = 20
    ) -> tuple[list[Task], int]:
        """Search tasks by title, description, and tags."""
        return await self.list(db, search=query_str, limit=limit)


task_service = TaskService()
