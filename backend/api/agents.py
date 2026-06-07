"""
Agents API — agent routing, status, and direct agent interaction endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.agents.router_agent import router_agent
from backend.database.models import AgentType, ChatRequest
from backend.database.schemas import serialize_doc
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/agents", tags=["Agents"])


@router.post("/route")
async def route_to_agent(
    body: ChatRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Send a message to a specific agent.

    Overrides the automatic routing and forces a specific agent.
    """
    response = await router_agent.process(
        user_id=user["id"],
        message=body.message,
        conversation_id=body.conversation_id,
        stream=False,
        attachments=body.attachments,
        metadata=body.metadata,
    )

    return {
        "success": True,
        "data": {
            "content": response.get("content", ""),
            "agent": response.get("agent", "router"),
            "conversation_id": response.get("conversation_id"),
            "category": response.get("category", "general"),
            "duration_ms": response.get("duration_ms", 0),
        },
    }


@router.get("/logs")
async def get_agent_logs(
    agent_name: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get agent execution logs."""
    from backend.database.mongodb import mongodb

    query: dict[str, Any] = {"user_id": user["id"]}
    if agent_name:
        query["agent_name"] = agent_name

    cursor = mongodb.agent_logs.find(
        query, sort=[("created_at", -1)], limit=limit
    )
    logs = await cursor.to_list(length=limit)
    return {
        "success": True,
        "data": {
            "items": [serialize_doc(log) for log in logs],
            "total": len(logs),
        },
    }


@router.get("/status")
async def get_agent_status(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get status of all available agents."""
    agents = [
        {"name": "router", "model": "DeepSeek V4 Flash", "status": "active", "description": "Request classification and routing"},
        {"name": "coding", "model": "Codex (GPT-4o)", "status": "active", "description": "Code generation, review, and debugging"},
        {"name": "research", "model": "Minimax M2.1", "status": "active", "description": "Web search, knowledge extraction, reports"},
        {"name": "vision", "model": "Mimo V2 Omni", "status": "active", "description": "Image analysis, OCR, visual QA"},
        {"name": "memory", "model": "DeepSeek V4 Flash", "status": "active", "description": "Memory storage, retrieval, consolidation"},
        {"name": "task", "model": "DeepSeek V4 Flash", "status": "active", "description": "Task management, reminders, scheduling"},
        {"name": "planner", "model": "DeepSeek V4 Flash", "status": "active", "description": "Architecture, roadmaps, task decomposition"},
    ]
    return {"success": True, "data": {"agents": agents}}
