"""Database migration / schema creation."""

from __future__ import annotations

from app.core.logger import get_logger
from app.database.connection import Base, engine

logger = get_logger(__name__)


async def create_tables() -> None:
    """Create all tables defined in ORM models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
