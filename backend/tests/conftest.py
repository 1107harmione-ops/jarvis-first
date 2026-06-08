"""Test fixtures and configuration."""
from __future__ import annotations

import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text

from app.core.config import settings

# Override database URL to use test file BEFORE any app imports
TEST_DB_URL = "sqlite+aiosqlite:///./test_jarvis.db"
settings.DATABASE_URL = TEST_DB_URL
os.environ["DATABASE_URL"] = TEST_DB_URL


def _init_db():
    """Synchronously create tables for testing."""
    import app.database.models  # noqa: F401
    from app.database.connection import Base
    sync_url = TEST_DB_URL.replace("+aiosqlite", "")
    sync_engine = create_engine(sync_url)
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()

    # Create FTS5 tables
    import sqlalchemy as sa
    fts_engine = create_engine(sync_url)
    with fts_engine.begin() as conn:
        from app.database.fts import FTS_SETUP_SQL
        for statement in FTS_SETUP_SQL.strip().split("\n\n"):
            stmt = statement.strip()
            if stmt:
                conn.execute(sa.text(stmt))
    fts_engine.dispose()


_init_db()


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    """Clean all table data before each test."""
    import app.database.models  # noqa: F401 — ensure all models loaded
    from app.database.connection import Base, async_session_factory
    async with async_session_factory() as session:
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(text(f"DELETE FROM {table.name}"))
        await session.commit()
    yield


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Provide an async test client with lifespan support."""
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
