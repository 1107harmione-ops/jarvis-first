from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.exceptions import ValidationError
from app.schemas.task import TaskCreate, TaskResponse, TaskListResponse, TaskUpdate
from app.tasks import service as task_service

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("", response_model=TaskResponse, status_code=201)
async def create_task(body: TaskCreate, db: AsyncSession = Depends(get_db)):
    valid_priorities = {"low", "medium", "high", "urgent"}
    if body.priority not in valid_priorities:
        raise ValidationError(f"Priority must be one of: {', '.join(sorted(valid_priorities))}")
    task = await task_service.create_task(
        db=db,
        title=body.title,
        description=body.description,
        priority=body.priority,
        due_date=body.due_date,
        tags=body.tags,
        category=body.category,
    )
    return task


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    tasks, total = await task_service.list_tasks(
        db=db, status=status, priority=priority, category=category,
        limit=limit, offset=offset,
    )
    return TaskListResponse(tasks=tasks, total=total)


@router.get("/search", response_model=TaskListResponse)
async def search_tasks(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    tasks, total = await task_service.search_tasks(db=db, query=q, limit=limit)
    return TaskListResponse(tasks=tasks, total=total)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    task = await task_service.get_task(db=db, task_id=task_id)
    return task


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, body: TaskUpdate, db: AsyncSession = Depends(get_db)):
    kwargs = body.model_dump(exclude_unset=True)
    task = await task_service.update_task(db=db, task_id=task_id, **kwargs)
    return task


@router.patch("/{task_id}/complete", response_model=TaskResponse)
async def complete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    task = await task_service.complete_task(db=db, task_id=task_id)
    return task


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    await task_service.delete_task(db=db, task_id=task_id)
