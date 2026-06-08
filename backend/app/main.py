"""Jarvis Voice Productivity Assistant — Main Application Entry Point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.exceptions import JarvisError, NotFoundError, ValidationError
from app.core.logger import get_logger, setup_logging
from app.database.connection import async_session_factory, engine
from app.database.migrations import create_tables

# Setup logging on import
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    # ── Startup ──
    logger.info(
        "Starting Jarvis Voice Assistant",
        version=settings.APP_VERSION,
    )

    # Ensure data directories exist
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create database tables
    await create_tables()

    # Initialize voice services (optional — graceful if unavailable)
    try:
        from app.voice.tts import edge_tts

        await edge_tts.initialize()
    except Exception as e:
        logger.warning("tts_init_skipped", error=str(e))

    # Start background reminder checker
    reminder_task = asyncio.create_task(_check_reminders_loop())
    logger.info("reminder_checker_started", interval_seconds=settings.REMINDER_CHECK_INTERVAL)

    yield

    # ── Shutdown ──
    logger.info("Shutting down Jarvis Voice Assistant")
    reminder_task.cancel()
    try:
        await reminder_task
    except asyncio.CancelledError:
        pass

    await engine.dispose()

    try:
        from app.voice.tts import edge_tts

        await edge_tts.close()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


# ── Background Reminder Checker ──────────────────────────────────


async def _check_reminders_loop() -> None:
    """Periodically check for due reminders and fire them."""
    while True:
        try:
            from app.reminders.schemas import ReminderUpdate
            from app.reminders.service import reminder_service

            async with async_session_factory() as db:
                due = await reminder_service.get_due(db)
                for reminder in due:
                    await reminder_service.mark_triggered(db, reminder.id)
                    logger.info(
                        "reminder_fired",
                        reminder_id=reminder.id,
                        title=reminder.title,
                    )
                    # For repeat reminders, reschedule
                    if reminder.repeat_type == "daily":
                        new_time = reminder.reminder_time.replace(
                            day=reminder.reminder_time.day + 1
                        )
                        await reminder_service.update(
                            db,
                            reminder.id,
                            ReminderUpdate(
                                reminder_time=new_time,
                                triggered=False,
                                status="pending",
                            ),
                        )
                    elif reminder.repeat_type == "weekly":
                        new_time = reminder.reminder_time.replace(
                            day=reminder.reminder_time.day + 7
                        )
                        await reminder_service.update(
                            db,
                            reminder.id,
                            ReminderUpdate(
                                reminder_time=new_time,
                                triggered=False,
                                status="pending",
                            ),
                        )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("reminder_check_error", error=str(e))
        await asyncio.sleep(settings.REMINDER_CHECK_INTERVAL)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────

@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(JarvisError)
async def jarvis_error_handler(request: Request, exc: JarvisError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Health Check ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME,
    }


@app.get("/api/admin/health")
async def admin_health():
    """Render health check endpoint."""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME,
    }


# ── Mount Routers ─────────────────────────────────────────────

from app.tasks.api import router as tasks_router
from app.notes.api import router as notes_router
from app.api.voice import router as voice_router
from app.api.reminders import router as reminders_router
from app.api.memory import router as memory_router
from app.search.api import router as search_router

# Note: routers already have their own prefix (e.g., /api/tasks, /api/voice)
app.include_router(tasks_router)
app.include_router(notes_router)
app.include_router(voice_router)
app.include_router(reminders_router)
app.include_router(memory_router)
app.include_router(search_router)

# ── Mount Static Files ────────────────────────────────────────────
from pathlib import Path

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("static_files_mounted", path=str(static_dir))
else:
    logger.warning("static_dir_not_found", path=str(static_dir))
