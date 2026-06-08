"""Note business logic layer."""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.logger import get_logger
from app.database.models import Note
from app.notes.schemas import NoteCreate, NoteUpdate

logger = get_logger(__name__)


class NoteService:
    """CRUD operations for notes."""

    async def create(self, db: AsyncSession, data: NoteCreate) -> Note:
        """Create a new note."""
        note = Note(
            title=data.title,
            content=data.content,
            category=data.category,
            tags=data.tags,
            priority=data.priority,
        )
        db.add(note)
        await db.commit()
        await db.refresh(note)
        logger.info("note_created", note_id=note.id, title=note.title)
        return note

    async def get(self, db: AsyncSession, note_id: int) -> Note:
        """Get a note by ID."""
        note = await db.get(Note, note_id)
        if not note:
            raise NotFoundError("Note", note_id)
        return note

    async def list(
        self,
        db: AsyncSession,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Note], int]:
        """List notes with optional filters and search."""
        query = select(Note)

        if category:
            query = query.where(Note.category == category)
        if priority:
            query = query.where(Note.priority == priority)
        if search:
            search_term = f"%{search}%"
            query = query.where(
                Note.title.ilike(search_term)
                | Note.content.ilike(search_term)
                | Note.tags.ilike(search_term)
            )

        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        query = query.order_by(Note.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        notes = list(result.scalars().all())

        return notes, total

    async def update(self, db: AsyncSession, note_id: int, data: NoteUpdate) -> Note:
        """Update a note."""
        note = await self.get(db, note_id)
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(note, field, value)
        note.updated_at = datetime.datetime.now()
        await db.commit()
        await db.refresh(note)
        logger.info("note_updated", note_id=note.id, changes=update_data)
        return note

    async def delete(self, db: AsyncSession, note_id: int) -> None:
        """Delete a note."""
        note = await self.get(db, note_id)
        await db.delete(note)
        await db.commit()
        logger.info("note_deleted", note_id=note_id)

    async def search(
        self, db: AsyncSession, query_str: str, limit: int = 20
    ) -> tuple[list[Note], int]:
        """Search notes by title, content, and tags."""
        return await self.list(db, search=query_str, limit=limit)


note_service = NoteService()
