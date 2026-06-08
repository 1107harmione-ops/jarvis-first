"""Memory business logic layer."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select, func, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.logger import get_logger
from app.database.models import MemoryEntry
from app.memory.schemas import MemoryCreate

logger = get_logger(__name__)

# Phrases that should never be saved
SKIP_PATTERNS = [
    "hello", "hi", "hey", "good morning", "good evening",
    "how are you", "what's up", "how's it going",
    "okay", "ok", "thanks", "thank you", "bye", "goodbye",
    "stop", "exit", "quiet", "help",
]

# High-importance keywords
HIGH_IMPORTANCE_KEYWORDS = [
    "my name", "i am", "i work", "i like", "i love", "i hate",
    "i prefer", "i need", "i want", "my favorite", "my goal",
    "i'm learning", "i study", "i use",
]


class MemoryService:
    async def store(self, db: AsyncSession, fact: str, category: str = "general", importance: int = 3) -> Optional[MemoryEntry]:
        fact_lower = fact.lower().strip()
        for skip in SKIP_PATTERNS:
            if fact_lower == skip or fact_lower.startswith(skip):
                logger.debug("memory_skipped", fact=fact_lower[:50])
                return None

        for kw in HIGH_IMPORTANCE_KEYWORDS:
            if kw in fact_lower:
                importance = max(importance, 4)
                break

        data = MemoryCreate(fact=fact, category=category, importance=importance)
        entry = MemoryEntry(fact=data.fact, category=data.category, importance=data.importance)
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        logger.info("memory_stored", entry_id=entry.id, fact=fact[:50])
        return entry

    async def list(self, db: AsyncSession, category: Optional[str] = None, limit: int = 50) -> tuple[list[MemoryEntry], int]:
        query = select(MemoryEntry)
        if category:
            query = query.where(MemoryEntry.category == category)
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        query = query.order_by(MemoryEntry.importance.desc(), MemoryEntry.created_at.desc()).limit(limit)
        result = await db.execute(query)
        entries = list(result.scalars().all())
        return entries, total

    async def search(self, db: AsyncSession, query_str: str, limit: int = 20) -> tuple[list[MemoryEntry], int]:
        like = f"%{query_str}%"
        stmt = (
            select(MemoryEntry)
            .where(MemoryEntry.fact.ilike(like))
            .order_by(MemoryEntry.importance.desc(), MemoryEntry.created_at.desc())
            .limit(limit)
        )
        count_stmt = select(func.count()).select_from(stmt.subquery())
        result = await db.execute(stmt)
        entries = list(result.scalars().all())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0
        return entries, total

    async def forget(self, db: AsyncSession, entry_id: int) -> None:
        entry = await db.get(MemoryEntry, entry_id)
        if not entry:
            raise NotFoundError("MemoryEntry", entry_id)
        await db.delete(entry)
        await db.commit()
        logger.info("memory_deleted", entry_id=entry_id)

    async def forget_by_fact(self, db: AsyncSession, fact_query: str) -> int:
        like = f"%{fact_query}%"
        stmt = select(MemoryEntry).where(MemoryEntry.fact.ilike(like))
        result = await db.execute(stmt)
        entries = list(result.scalars().all())
        count = len(entries)
        for entry in entries:
            await db.delete(entry)
        if count > 0:
            await db.commit()
            logger.info("memory_deleted_many", query=fact_query, count=count)
        return count


memory_service = MemoryService()
