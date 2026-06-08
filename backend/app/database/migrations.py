"""Database migration / schema creation."""
from __future__ import annotations

from app.core.logger import get_logger
from app.database.connection import Base, engine

# Import all models so they register with Base.metadata
import app.database.models  # noqa: F401

logger = get_logger(__name__)


async def create_tables() -> None:
    """Create all tables defined in ORM models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

    # Set up FTS5 virtual tables (best-effort)
    try:
        from app.database.fts import setup_fts
        await setup_fts()
    except Exception as e:
        logger.warning("fts_setup_skipped", error=str(e))
