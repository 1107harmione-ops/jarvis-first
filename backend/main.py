"""
JARVIS Backend — Main Application Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enterprise-grade AI Assistant backend with:
- Multi-agent system (coding, research, vision, memory, task, planner)
- MongoDB + Vector memory (STM/LTM)
- WebSocket real-time chat with streaming
- Voice processing (STT/TTS)
- JWT authentication
- Production-ready configuration

Usage:
    uvicorn backend.main:app --reload
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import admin, agents, auth, chat, memory, tasks, voice
from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.services.task_service import task_service
from backend.services.voice_service import voice_service
from backend.utils.logger import setup_logging, get_logger
from backend.utils.security import rate_limiter
from backend.websocket.chat_socket import chat_websocket

# Setup logging on import
setup_logging()
logger = get_logger(__name__)


# ── Application Lifecycle ────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    # ── Startup ──
    logger.info(
        "Starting JARVIS backend",
        extra={"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT.value},
    )

    # Connect to MongoDB
    await mongodb.connect()
    await mongodb.ensure_indexes()

    # Start background workers
    task_service.start_worker()
    voice_service.start_cleanup_task()

    yield

    # ── Shutdown ──
    logger.info("Shutting down JARVIS backend")

    # Stop workers
    await task_service.stop_worker()

    # Close connections
    await mongodb.disconnect()
    await voice_service.close()

    # Close LLM clients
    from backend.llm.deepseek import deepseek
    from backend.llm.codex import codex
    from backend.llm.minimax import minimax
    from backend.llm.mimo import mimo
    await deepseek.close()
    await codex.close()
    await minimax.close()
    await mimo.close()

    logger.info("Shutdown complete")


# ── FastAPI App ──────────────────────────────────────────────────


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────────────────


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: callable) -> JSONResponse:
    """Apply rate limiting per IP."""
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_ip}"

    if not rate_limiter.check(key, settings.RATE_LIMIT_PER_MINUTE, window_seconds=60):
        logger.warning("Rate limit exceeded", extra={"ip": client_ip, "path": request.url.path})
        return JSONResponse(
            status_code=429,
            content={"success": False, "error": "Rate limit exceeded. Try again later."},
        )

    response = await call_next(request)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next: callable) -> JSONResponse:
    """Log all API requests."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": f"{elapsed:.1f}",
            "ip": request.client.host if request.client else "unknown",
        },
    )
    return response


# ── WebSocket Route ──────────────────────────────────────────────


@app.websocket("/ws/chat")
async def websocket_chat(websocket: Request) -> None:
    """WebSocket endpoint for real-time chat.

    Connect with JWT token as query parameter or first message:
        ws://host:port/ws/chat?token=<jwt>

    Or send first message: {"token": "<jwt>"}
    """
    token = websocket.query_params.get("token")
    await chat_websocket.handle(websocket, token)


# ── API Routes ───────────────────────────────────────────────────


app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(voice.router)
app.include_router(memory.router)
app.include_router(tasks.router)
app.include_router(agents.router)
app.include_router(admin.router)

# Auth routes are in api/auth.py (included via chat router's dependency)


# ── Root Endpoint ────────────────────────────────────────────────


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "docs": "/docs",
        "health": "/api/admin/health",
    }


@app.get("/robots.txt")
async def robots() -> str:
    """Disallow all crawlers."""
    return "User-agent: *\nDisallow: /"


# ── Exception Handlers ──────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        extra={"path": request.url.path, "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"},
    )


# ── Entry Point ──────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.value.lower(),
        reload=settings.is_development,
    )
