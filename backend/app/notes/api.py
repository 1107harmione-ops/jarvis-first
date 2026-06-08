"""Note REST API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.notes.schemas import NoteCreate, NoteRead, NoteListResponse, NoteUpdate
from app.notes.service import note_service

router = APIRouter(prefix="/api/notes", tags=["notes"])


@router.post("", response_model=NoteRead, status_code=201)
async def create_note(data: NoteCreate, db: AsyncSession = Depends(get_db)):
    """Create a new note."""
    return await note_service.create(db, data)


@router.get("", response_model=NoteListResponse)
async def list_notes(
    category: Optional[str] = Query(None, pattern=r"^(learning|project|personal|ideas|research)$"),
    priority: Optional[str] = Query(None, pattern=r"^(low|medium|high|urgent)$"),
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List notes with optional filters."""
    notes, total = await note_service.list(db, category, priority, search, limit, offset)
    return NoteListResponse(notes=notes, total=total)


@router.get("/search", response_model=NoteListResponse)
async def search_notes(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Search notes by query string."""
    notes, total = await note_service.search(db, q, limit)
    return NoteListResponse(notes=notes, total=total)


@router.get("/{note_id}", response_model=NoteRead)
async def get_note(note_id: int, db: AsyncSession = Depends(get_db)):
    """Get a note by ID."""
    try:
        return await note_service.get(db, note_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")


@router.patch("/{note_id}", response_model=NoteRead)
async def update_note(note_id: int, data: NoteUpdate, db: AsyncSession = Depends(get_db)):
    """Update a note."""
    try:
        return await note_service.update(db, note_id, data)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")


@router.delete("/{note_id}", status_code=204)
async def delete_note(note_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a note."""
    try:
        await note_service.delete(db, note_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
