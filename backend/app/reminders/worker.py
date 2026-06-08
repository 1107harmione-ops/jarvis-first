"""RQ worker — fires due reminders.

This module contains the synchronous worker functions that RQ invokes.
It uses synchronous SQLAlchemy access to avoid async-in-sync issues.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the backend package is importable
_backend_root = Path(__file__).resolve().parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


def fire_reminder(reminder_id: int) -> dict:
    """Fire a reminder: mark as triggered in the database.

    This function is called by the RQ worker. It uses a synchronous
    SQLAlchemy connection to avoid the async-in-sync problem.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.config import settings

    sync_url = settings.DATABASE_URL.replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    try:
        from app.database.models import Reminder  # noqa: F401

        with Session(engine) as session:
            reminder = session.get(Reminder, reminder_id)
            if reminder is None:
                return {"status": "not_found", "reminder_id": reminder_id}

            if reminder.triggered:
                return {"status": "already_triggered", "reminder_id": reminder_id}

            reminder.triggered = True
            reminder.status = "fired"
            session.commit()

            return {
                "status": "fired",
                "reminder_id": reminder.id,
                "title": reminder.title,
            }
    finally:
        engine.dispose()


# ── CLI entrypoint for the RQ worker ─────────────────────────────

def run_worker() -> None:
    """Run the RQ worker process (blocking)."""
    from rq import Connection, Worker

    from app.reminders.redis import redis_manager

    conn = redis_manager.client
    with Connection(conn):
        queues = ["jarvis"]
        w = Worker(queues)
        w.work()


if __name__ == "__main__":
    run_worker()
