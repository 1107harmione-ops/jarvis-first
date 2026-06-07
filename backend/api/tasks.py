"""
Tasks API — CRUD for tasks, reminders, and scheduled items.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.database.models import TaskCreate, TaskUpdate
from backend.database.schemas import serialize_doc
from backend.services.task_service import task_service
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/tasks", tags=["Tasks"])


@router.post("/")
async def create_task(
    body: TaskCreate,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new task/reminder."""
    from datetime import datetime

    due_at = None
    if body.due_at:
        try:
            due_at = datetime.fromisoformat(body.due_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    scheduled_at = None
    if body.scheduled_at:
        try:
            scheduled_at = datetime.fromisoformat(body.scheduled_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    task = await task_service.create_task(
        user_id=user["id"],
        title=body.title,
        description=body.description,
        priority=body.priority.value,
        due_at=due_at,
        scheduled_at=scheduled_at,
        recurring=body.recurring,
        tags=body.tags,
    )
    return {"success": True, "data": task}


@router.get("/")
async def list_tasks(
    status: str | None = Query(None),
    priority: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """List tasks with optional filters."""
    skip = (page - 1) * page_size
    tasks = await task_service.list_tasks(
        user["id"], status=status, priority=priority, limit=page_size, skip=skip
    )
    total = len(tasks)  # Simplified; use count query in production

    return {
        "success": True,
        "data": {
            "items": tasks,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    }


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get a single task."""
    task = await task_service.get_task(task_id, user["id"])
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "data": task}


@router.patch("/{task_id}")
async def update_task(
    task_id: str,
    body: TaskUpdate,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Update a task."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return {"success": True, "message": "No changes provided"}

    task = await task_service.update_task(task_id, user["id"], updates)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "data": task}


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete a task."""
    deleted = await task_service.delete_task(task_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "message": "Task deleted"}


@router.post("/{task_id}/complete")
async def complete_task(
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Mark a task as completed."""
    completed = await task_service.complete_task(task_id, user["id"])
    if not completed:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "message": "Task completed"}


@router.get("/stats/summary")
async def get_task_stats(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get task statistics."""
    stats = await task_service.get_stats(user["id"])
    return {"success": True, "data": stats}
