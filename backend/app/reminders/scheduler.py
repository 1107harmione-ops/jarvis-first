"""RQ scheduler for reminder jobs."""

from __future__ import annotations

import datetime
import os
from typing import Optional

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── RQ imports (optional — gracefully degrade if RQ not installed) ──

try:
    import rq
    from rq import Queue, Retry
    from rq.job import Job

    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False

    class Queue:  # type: ignore[no-redef]
        """Stub when RQ is not installed."""

        def __init__(self, *args, **kwargs): ...  # noqa: N807

    class Retry:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs): ...  # noqa: N807


def _get_connection():
    """Get Redis connection for RQ."""
    from app.reminders.redis import redis_manager

    return redis_manager.client


def get_queue() -> Optional[Queue]:
    """Return the RQ Queue or None if RQ/Redis unavailable."""
    if not RQ_AVAILABLE:
        logger.warning("rq_not_installed")
        return None
    try:
        conn = _get_connection()
        return Queue(settings.RQ_QUEUE_NAME, connection=conn)
    except Exception as e:
        logger.warning("rq_queue_unavailable", error=str(e))
        return None


def schedule_reminder(reminder_id: int, reminder_time: datetime.datetime) -> bool:
    """Enqueue a reminder-firing job at the specified time."""
    queue = get_queue()
    if queue is None:
        logger.warning("reminder_not_scheduled_rq", reminder_id=reminder_id)
        return False

    # Calculate delay in seconds
    now = datetime.datetime.now(datetime.timezone.utc)
    if reminder_time.tzinfo is None:
        reminder_time = reminder_time.replace(tzinfo=datetime.timezone.utc)
    delay = max(0, int((reminder_time - now).total_seconds()))

    try:
        queue.enqueue(
            "app.reminders.worker.fire_reminder",
            args=(reminder_id,),
            job_timeout=settings.RQ_DEFAULT_TIMEOUT,
            meta={"reminder_id": reminder_id},
            result_ttl=86400,  # keep result for 1 day
            failure_ttl=86400,
            retry=Retry(max=3, interval=[60, 300, 600]),
        )
        logger.info(
            "reminder_scheduled_rq",
            reminder_id=reminder_id,
            delay_seconds=delay,
        )
        return True
    except Exception as e:
        logger.error("reminder_schedule_error", reminder_id=reminder_id, error=str(e))
        return False


def cancel_reminder_jobs(reminder_id: int) -> None:
    """Cancel all pending RQ jobs for a given reminder (best-effort)."""
    queue = get_queue()
    if queue is None:
        return
    try:
        # RQ doesn't have a native "cancel by meta" — we iterate registered jobs
        # This is best-effort for small queues.
        registry = queue.started_job_registry
        for job_id in registry.get_job_ids():
            job = queue.fetch_job(job_id)
            if job and job.meta.get("reminder_id") == reminder_id:
                job.cancel()
    except Exception as e:
        logger.warning("reminder_cancel_error", reminder_id=reminder_id, error=str(e))
