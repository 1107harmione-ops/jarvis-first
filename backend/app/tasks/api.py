"""Task REST API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.tasks.schemas import TaskCreate, TaskRead, TaskListResponse, TaskUpdate
from app.tasks.service import task_service

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("", response_model=TaskRead, status_code=201)
async def create_task(data: TaskCreate, db: AsyncSession = Depends(get_db)):
    """Create a new task."""
    return await task_service.create(db, data)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, pattern=r"^(pending|completed|cancelled)$"),
    category: Optional[str] = None,
    priority: Optional[str] = Query(None, pattern=r"^(low|medium|high|urgent)$"),
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List tasks with optional filters."""
    tasks, total = await task_service.list(db, status, category, priority, search, limit, offset)
    return TaskListResponse(tasks=tasks, total=total)


@router.get("/search", response_model=TaskListResponse)
async def search_tasks(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Search tasks by query string."""
    tasks, total = await task_service.search(db, q, limit)
    return TaskListResponse(tasks=tasks, total=total)


@router.get("/{task_id}", response_model=TaskRead)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Get a task by ID."""
    try:
        return await task_service.get(db, task_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.patch("/{task_id}", response_model=TaskRead)
async def update_task(task_id: int, data: TaskUpdate, db: AsyncSession = Depends(get_db)):
    """Update a task."""
    try:
        return await task_service.update(db, task_id, data)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.patch("/{task_id}/complete", response_model=TaskRead)
async def complete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Mark a task as completed."""
    try:
        return await task_service.complete(db, task_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a task."""
    try:
        await task_service.delete(db, task_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
