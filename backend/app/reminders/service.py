"""Reminder business logic layer."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.logger import get_logger
from app.database.models import Reminder
from app.reminders.schemas import ReminderCreate, ReminderUpdate

logger = get_logger(__name__)


class ReminderService:
    async def create(self, db: AsyncSession, data: ReminderCreate) -> Reminder:
        reminder = Reminder(
            title=data.title,
            reminder_time=data.reminder_time,
            repeat_type=data.repeat_type,
        )
        db.add(reminder)
        await db.commit()
        await db.refresh(reminder)

        # Schedule RQ job in background (best-effort)
        try:
            from app.reminders.scheduler import schedule_reminder
            schedule_reminder(reminder.id, reminder.reminder_time)
        except Exception as e:
            logger.warning("reminder_rq_schedule_skipped", reminder_id=reminder.id, error=str(e))

        logger.info("reminder_created", reminder_id=reminder.id, title=reminder.title)
        return reminder

    async def get(self, db: AsyncSession, reminder_id: int) -> Reminder:
        reminder = await db.get(Reminder, reminder_id)
        if not reminder:
            raise NotFoundError("Reminder", reminder_id)
        return reminder

    async def list(
        self,
        db: AsyncSession,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Reminder], int]:
        query = select(Reminder)
        if status:
            query = query.where(Reminder.status == status)
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        query = query.order_by(Reminder.reminder_time.asc()).offset(offset).limit(limit)
        result = await db.execute(query)
        reminders = list(result.scalars().all())
        return reminders, total

    async def update(self, db: AsyncSession, reminder_id: int, data: ReminderUpdate) -> Reminder:
        reminder = await self.get(db, reminder_id)
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(reminder, field, value)
        await db.commit()
        await db.refresh(reminder)
        logger.info("reminder_updated", reminder_id=reminder.id)
        return reminder

    async def delete(self, db: AsyncSession, reminder_id: int) -> None:
        reminder = await self.get(db, reminder_id)
        await db.delete(reminder)
        await db.commit()

        # Cancel any pending RQ jobs
        try:
            from app.reminders.scheduler import cancel_reminder_jobs
            cancel_reminder_jobs(reminder_id)
        except Exception as e:
            logger.warning("reminder_rq_cancel_skipped", reminder_id=reminder_id, error=str(e))

        logger.info("reminder_deleted", reminder_id=reminder_id)

    async def get_due(self, db: AsyncSession) -> list[Reminder]:
        now = func.now()
        query = select(Reminder).where(
            and_(
                Reminder.status == "pending",
                Reminder.reminder_time <= now,
                Reminder.triggered == False,
            )
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    async def mark_triggered(self, db: AsyncSession, reminder_id: int) -> Reminder:
        return await self.update(db, reminder_id, ReminderUpdate(triggered=True))


reminder_service = ReminderService()
