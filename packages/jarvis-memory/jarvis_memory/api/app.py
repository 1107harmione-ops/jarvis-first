"""FastAPI application factory for jarvis-memory REST API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jarvis_memory.api.routes import router
from jarvis_memory.config import settings
from jarvis_memory.database import db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: connect to MongoDB on startup, disconnect on shutdown."""
    logger.info(
        "Starting jarvis-memory API — connecting to MongoDB at %s",
        settings.MONGODB_URI.split("@")[-1] if "@" in settings.MONGODB_URI else settings.MONGODB_URI,
    )
    await db.connect(settings.MONGODB_URI, settings.MONGODB_DB_NAME)
    yield
    await db.disconnect()
    logger.info("jarvis-memory API shut down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured ``FastAPI`` instance.
    """
    app = FastAPI(
        title="jarvis-memory",
        description="MongoDB-backed memory architecture for AI assistants",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "jarvis-memory",
            "version": "0.1.0",
            "database": "connected" if db.db is not None else "disconnected",
        }

    return app


app = create_app()
