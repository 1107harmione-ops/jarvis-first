"""Jarvis Voice Productivity Assistant — Main Application Entry Point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import JarvisError, NotFoundError, ValidationError
from app.core.logger import get_logger, setup_logging
from app.database.connection import engine
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

    yield

    # ── Shutdown ──
    logger.info("Shutting down Jarvis Voice Assistant")
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


# ── Mount Routers ─────────────────────────────────────────────

from app.tasks.api import router as tasks_router
from app.api.voice import router as voice_router

# Note: routers already have their own prefix (e.g., /api/tasks, /api/voice)
app.include_router(tasks_router)
app.include_router(voice_router)
