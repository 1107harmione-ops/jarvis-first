"""Memory REST API endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.memory.schemas import MemoryCreate, MemoryListResponse, MemoryRead
from app.memory.service import memory_service

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.post("", response_model=MemoryRead, status_code=201)
async def store_memory(data: MemoryCreate, db: AsyncSession = Depends(get_db)):
    entry = await memory_service.store(db, data.fact, data.category, data.importance)
    if not entry:
        raise HTTPException(status_code=400, detail="Memory not stored (content filtered)")
    return entry


@router.get("", response_model=MemoryListResponse)
async def list_memory(
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    entries, total = await memory_service.list(db, category, limit)
    return MemoryListResponse(entries=entries, total=total)


@router.get("/search", response_model=MemoryListResponse)
async def search_memory(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    entries, total = await memory_service.search(db, q, limit)
    return MemoryListResponse(entries=entries, total=total)


@router.delete("/{entry_id}", status_code=204)
async def forget_memory(entry_id: int, db: AsyncSession = Depends(get_db)):
    try:
        await memory_service.forget(db, entry_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Memory entry {entry_id} not found")
