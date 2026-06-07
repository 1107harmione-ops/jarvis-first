"""
Admin API — system management, user management, monitoring, and configuration endpoints.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.database.mongodb import mongodb
from backend.database.schemas import serialize_doc
from backend.utils.auth import get_current_user, require_admin
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ── Health / System ─────────────────────────────────────────────


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """System health check."""
    import os

    # Check MongoDB
    mongo_status = "disconnected"
    try:
        await mongodb.db.command("ping")
        mongo_status = "connected"
    except Exception:
        mongo_status = "error"

    return {
        "status": "ok" if mongo_status == "connected" else "degraded",
        "version": "1.0.0",
        "environment": "production",  # Will be overridden by config
        "mongodb": mongo_status,
        "uptime_seconds": time.time() - _start_time,
    }


@router.get("/health/full")
async def full_health_check(
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Full system health check (admin only)."""
    import os, psutil

    mongo_status = "disconnected"
    mongo_latency = 0.0
    try:
        start = time.monotonic()
        await mongodb.db.command("ping")
        mongo_latency = (time.monotonic() - start) * 1000
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"error: {exc}"

    # Collection stats
    collections = ["users", "conversations", "messages", "memories", "tasks", "agent_logs"]
    collection_counts: dict[str, int] = {}
    for col in collections:
        try:
            collection_counts[col] = await mongodb.get_collection(col).count_documents({})
        except Exception:
            collection_counts[col] = -1

    return {
        "status": "ok" if mongo_status == "connected" else "degraded",
        "version": "1.0.0",
        "environment": "production",
        "mongodb": {
            "status": mongo_status,
            "latency_ms": round(mongo_latency, 1),
            "collections": collection_counts,
        },
        "system": {
            "uptime_seconds": round(time.time() - _start_time, 1),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
        },
        "active_connections": len(mongodb._client.nodes) if mongodb._client else 0,
    }


# ── User Management (Admin) ─────────────────────────────────────


@router.get("/users")
async def list_users(
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """List all users (admin only)."""
    cursor = mongodb.users.find({}, {
        "password_hash": 0,
        "api_key_hashed": 0,
    }).sort("created_at", -1).limit(100)
    users = await cursor.to_list(length=100)
    return {
        "success": True,
        "data": {"users": [serialize_doc(u) for u in users], "total": len(users)},
    }


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get user details (admin only)."""
    user = await mongodb.users.find_one(
        {"_id": user_id},
        {"password_hash": 0, "api_key_hashed": 0},
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "data": serialize_doc(user)}


@router.patch("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Deactivate a user (admin only)."""
    result = await mongodb.users.update_one(
        {"_id": user_id},
        {"$set": {"is_active": False}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "message": "User deactivated"}


# ── Monitoring ──────────────────────────────────────────────────


@router.get("/logs")
async def get_system_logs(
    level: str = "ERROR",
    limit: int = 100,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get system logs (admin only)."""
    cursor = mongodb.agent_logs.find(
        {"status": "error"} if level == "ERROR" else {},
        sort=[("created_at", -1)],
        limit=limit,
    )
    logs = await cursor.to_list(length=limit)
    return {
        "success": True,
        "data": {"items": [serialize_doc(log) for log in logs], "total": len(logs)},
    }


@router.get("/metrics")
async def get_metrics(
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get system metrics (admin only)."""
    # Aggregate metrics from agent_logs
    pipeline = [
        {"$group": {
            "_id": "$agent_name",
            "total_calls": {"$sum": 1},
            "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}},
            "avg_duration_ms": {"$avg": "$duration_ms"},
            "total_tokens": {"$sum": {"$ifNull": ["$tokens_used", 0]}},
        }},
    ]
    cursor = mongodb.agent_logs.aggregate(pipeline)
    agent_metrics = await cursor.to_list(length=50)

    return {
        "success": True,
        "data": {
            "by_agent": agent_metrics,
            "total_calls": sum(m.get("total_calls", 0) for m in agent_metrics),
            "total_errors": sum(m.get("errors", 0) for m in agent_metrics),
        },
    }


# ── Config ──────────────────────────────────────────────────────


@router.get("/config")
async def get_config(
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get non-sensitive configuration (admin only)."""
    from backend.config.settings import settings

    return {
        "success": True,
        "data": {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "mongodb_db": settings.MONGODB_DATABASE,
            "memory_stm_ttl_hours": settings.MEMORY_STM_TTL_HOURS,
            "memory_ltm_threshold": settings.MEMORY_LTM_IMPORTANCE_THRESHOLD,
            "voice_session_timeout": settings.VOICE_SESSION_TIMEOUT_SECONDS,
            "rate_limit": settings.RATE_LIMIT_PER_MINUTE,
            "debug": settings.DEBUG,
        },
    }


# Startup time for uptime tracking
_start_time = time.time()

# Import psutil for health check
try:
    import psutil  # noqa: F401
except ImportError:
    psutil = None  # type: ignore[assignment]
