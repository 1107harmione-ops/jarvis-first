"""Reminder REST API endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.reminders.schemas import ReminderCreate, ReminderListResponse, ReminderRead, ReminderUpdate
from app.reminders.service import reminder_service

router = APIRouter(prefix="/api/reminders", tags=["reminders"])


@router.post("", response_model=ReminderRead, status_code=201)
async def create_reminder(data: ReminderCreate, db: AsyncSession = Depends(get_db)):
    return await reminder_service.create(db, data)


@router.get("", response_model=ReminderListResponse)
async def list_reminders(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    reminders, total = await reminder_service.list(db, status, limit, offset)
    return ReminderListResponse(reminders=reminders, total=total)


@router.get("/{reminder_id}", response_model=ReminderRead)
async def get_reminder(reminder_id: int, db: AsyncSession = Depends(get_db)):
    try:
        return await reminder_service.get(db, reminder_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Reminder {reminder_id} not found")


@router.patch("/{reminder_id}", response_model=ReminderRead)
async def update_reminder(reminder_id: int, data: ReminderUpdate, db: AsyncSession = Depends(get_db)):
    try:
        return await reminder_service.update(db, reminder_id, data)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Reminder {reminder_id} not found")


@router.delete("/{reminder_id}", status_code=204)
async def delete_reminder(reminder_id: int, db: AsyncSession = Depends(get_db)):
    try:
        await reminder_service.delete(db, reminder_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Reminder {reminder_id} not found")


@router.post("/check-due", response_model=ReminderListResponse)
async def check_due_reminders(db: AsyncSession = Depends(get_db)):
    reminders = await reminder_service.get_due(db)
    for r in reminders:
        await reminder_service.mark_triggered(db, r.id)
    return ReminderListResponse(reminders=reminders, total=len(reminders))
