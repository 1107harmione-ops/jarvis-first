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
from backend.services.piper_tts import piper_service
from backend.services.whisper_stt import whisper_service
from backend.services.offline_handler import offline_handler
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

    # Initialize the multi-agent system
    from backend.agents_v2.init import initialize_agent_system
    await initialize_agent_system()

    # Initialize voice services
    await piper_service.initialize()
    await whisper_service.initialize()

    # Start background workers
    task_service.start_worker()
    voice_service.start_cleanup_task()
    await offline_handler.start_monitoring()

    yield

    # ── Shutdown ──
    logger.info("Shutting down JARVIS backend")

    # Stop workers
    await task_service.stop_worker()
    await offline_handler.stop_monitoring()

    # Close voice services
    await voice_service.close()
    await piper_service.close()
    await whisper_service.close()

    # Shutdown agent system
    from backend.agents_v2.init import shutdown_agent_system
    await shutdown_agent_system()

    # Close connections
    await mongodb.disconnect()

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


@app.websocket("/ws/voice")
async def websocket_voice(websocket: Request) -> None:
    """WebSocket endpoint for real-time voice conversations.

    Streaming audio in (PCM16) → STT → Agent → TTS → Streaming audio out.

    Connect with JWT token as query parameter:
        ws://host:port/ws/voice?token=<jwt>

    Protocol:
        Client → Server:
          - Binary: PCM16 audio chunks (16000Hz MONO)
          - JSON: {"type":"audio_start|audio_end|interrupt|config|ping"}

        Server → Client:
          - Binary: PCM16 audio chunks (22050Hz MONO) for TTS
          - JSON: state/transcript/thinking/tts_start/tts_end/error/pong
    """
    from backend.websocket.voice_socket import voice_ws_manager
    from backend.utils.auth import decode_token

    token = websocket.query_params.get("token", "")
    user = decode_token(token)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return

    user_id = user.get("sub", user.get("id", ""))
    if not user_id:
        from bson import ObjectId
        user_id = str(user.get("_id", ""))
        if not user_id:
            await websocket.close(code=4001, reason="Invalid token")
            return

    config_str = websocket.query_params.get("config")
    initial_config = None
    if config_str:
        try:
            import json
            initial_config = json.loads(config_str)
        except (json.JSONDecodeError, Exception):
            pass

    await voice_ws_manager.handle_connection(websocket, user_id, initial_config)


# ── API Routes ───────────────────────────────────────────────────


app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(voice.router)
app.include_router(memory.router)
app.include_router(tasks.router)
app.include_router(agents.router)
app.include_router(admin.router)

# Register v2 agent API (LangGraph-powered multi-agent system)
from backend.api.agents_v2 import router as agents_v2_router
app.include_router(agents_v2_router)

# Register research API
from backend.api.research import router as research_router
app.include_router(research_router)

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
