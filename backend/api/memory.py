"""
Memory API — memory storage, retrieval, search, and management endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.database.models import MemoryCreate, MemorySearchRequest
from backend.database.schemas import serialize_doc
from backend.services.memory_service import memory_service
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/memory", tags=["Memory"])


@router.post("/store")
async def store_memory(
    body: MemoryCreate,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Store a new memory."""
    doc = await memory_service.store(
        user_id=user["id"],
        content=body.content,
        memory_type=body.memory_type.value,
        tags=body.tags,
        importance_score=body.importance_score,
        source=body.source or "api",
        metadata=body.metadata,
    )
    return {"success": True, "data": doc}


@router.post("/search")
async def search_memories(
    body: MemorySearchRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Search memories by query (semantic + text)."""
    import time
    start = time.monotonic()
    results = await memory_service.search(
        user_id=user["id"],
        query=body.query,
        memory_type=body.memory_type.value if body.memory_type else None,
        limit=body.limit,
        threshold=body.threshold,
    )
    elapsed = (time.monotonic() - start) * 1000
    return {
        "success": True,
        "data": {
            "results": results,
            "total": len(results),
            "query_time_ms": round(elapsed, 1),
        },
    }


@router.get("/recent")
async def get_recent_memories(
    limit: int = Query(20, ge=1, le=100),
    memory_type: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get recent memories."""
    from backend.memory.short_term import stm

    results = await stm.get_recent(user["id"], limit=limit)
    if memory_type:
        results = [r for r in results if r.get("memory_type") == memory_type]
    return {"success": True, "data": {"items": results, "total": len(results)}}


@router.get("/important")
async def get_important_memories(
    limit: int = Query(20, ge=1, le=100),
    min_score: float = Query(0.6, ge=0.0, le=1.0),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get high-importance long-term memories."""
    from backend.memory.long_term import ltm

    results = await ltm.get_by_importance(user["id"], limit=limit, min_score=min_score)
    return {"success": True, "data": {"items": results, "total": len(results)}}


@router.post("/consolidate")
async def consolidate_memories(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Trigger STM → LTM consolidation."""
    count = await memory_service.consolidate(user["id"])
    return {
        "success": True,
        "data": {"consolidated": count},
        "message": f"Consolidated {count} memories",
    }


@router.get("/stats")
async def get_memory_stats(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get memory statistics."""
    stats = await memory_service.get_stats(user["id"])
    return {"success": True, "data": stats}


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete a specific memory."""
    deleted = await memory_service.delete_memory(memory_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"success": True, "message": "Memory deleted"}


@router.delete("/clear-stm")
async def clear_short_term(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Clear all short-term memories."""
    count = await memory_service.clear_stm(user["id"])
    return {"success": True, "message": f"Cleared {count} short-term memories"}
