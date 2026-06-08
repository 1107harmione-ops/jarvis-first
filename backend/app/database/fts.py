"""FTS5 virtual table setup and maintenance."""
from __future__ import annotations

from sqlalchemy import text

from app.core.logger import get_logger
from app.database.connection import engine

logger = get_logger(__name__)

FTS_SETUP_SQL = """
-- Tasks FTS
CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
    title, description, tags,
    content='tasks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS tasks_fts_ai AFTER INSERT ON tasks BEGIN
    INSERT INTO tasks_fts(rowid, title, description, tags)
    VALUES (new.id, new.title, new.description, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS tasks_fts_ad AFTER DELETE ON tasks BEGIN
    INSERT INTO tasks_fts(tasks_fts, rowid, title, description, tags)
    VALUES ('delete', old.id, old.title, old.description, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS tasks_fts_au AFTER UPDATE ON tasks BEGIN
    INSERT INTO tasks_fts(tasks_fts, rowid, title, description, tags)
    VALUES ('delete', old.id, old.title, old.description, old.tags);
    INSERT INTO tasks_fts(rowid, title, description, tags)
    VALUES (new.id, new.title, new.description, new.tags);
END;

-- Notes FTS
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content, tags,
    content='notes',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS notes_fts_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_fts_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_fts_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
    INSERT INTO notes_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

-- Memory FTS
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    fact,
    content='memory_entries',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory_entries BEGIN
    INSERT INTO memory_fts(rowid, fact)
    VALUES (new.id, new.fact);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory_entries BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, fact)
    VALUES ('delete', old.id, old.fact);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_au AFTER UPDATE ON memory_entries BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, fact)
    VALUES ('delete', old.id, old.fact);
    INSERT INTO memory_fts(rowid, fact)
    VALUES (new.id, new.fact);
END;
"""


async def setup_fts() -> None:
    """Create FTS5 virtual tables and triggers."""
    async with engine.begin() as conn:
        for statement in FTS_SETUP_SQL.strip().split("\n\n"):
            stmt = statement.strip()
            if stmt:
                await conn.execute(text(stmt))
    logger.info("FTS5 virtual tables created")


async def rebuild_fts() -> None:
    """Rebuild FTS indexes from current data."""
    async with engine.begin() as conn:
        for table in ("tasks_fts", "notes_fts", "memory_fts"):
            await conn.execute(text(f"INSERT INTO {table}({table}) VALUES('rebuild')"))
    logger.info("FTS5 indexes rebuilt")
