"""
Agent API Routes
================
FastAPI endpoints for the multi-agent system.

Provides:
- POST /api/v2/agents/route     — Route a message through the agent graph
- POST /api/v2/agents/process   — Direct agent call (bypass routing)
- GET  /api/v2/agents           — List available agents
- GET  /api/v2/agents/metrics   — Get agent system metrics
- GET  /api/v2/agents/logs      — Get agent execution logs
- GET  /api/v2/agents/sessions/{session_id} — Get session details
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from agents_v2.state import AgentState, create_initial_state
from agents_v2.graph import create_agent_graph
from agents_v2.registry import get_agent_registry
from agents_v2.monitor import get_agent_monitor
from api.auth import get_current_user
from database.models import UserResponse

router = APIRouter(prefix="/api/v2/agents", tags=["agents-v2"])


# ── Request/Response models ───────────────────


class AgentRouteRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    max_retries: int = 3


class AgentRouteResponse(BaseModel):
    session_id: str
    response: str
    category: Optional[str] = None
    agents_used: List[str] = []
    execution_path: List[str] = []
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    errors: List[Dict[str, Any]] = []


class AgentInfo(BaseModel):
    name: str
    model: str
    description: str
    available: bool


# ── Router dependencies ───────────────────────

from pydantic import BaseModel  # noqa: E402 (late import for readability)


async def get_agent_graph():
    """Dependency: get or create the agent graph."""
    return create_agent_graph()


# ── Endpoints ─────────────────────────────────


@router.post("/route", response_model=AgentRouteResponse)
async def route_request(
    request: AgentRouteRequest,
    current_user: UserResponse = Depends(get_current_user),
    graph=Depends(get_agent_graph),
):
    """
    Route a user message through the full multi-agent workflow.

    The request flows through:
    Router → (Planner if needed) → Execute Steps → Response → Memory Log
    """
    state = create_initial_state(
        user_id=current_user.id,
        message=request.message,
        conversation_id=request.conversation_id,
        attachments=request.attachments,
        metadata=request.metadata,
        max_retries=request.max_retries,
    )

    # Execute through the agent graph
    monitor = get_agent_monitor()
    monitor.start_session(state)

    try:
        result_state = await graph.aexecute(state)
    except Exception as exc:
        monitor.end_session(state)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")

    # Record session metrics
    monitor.end_session(result_state)
    try:
        await monitor.persist_session(result_state)
    except Exception:
        pass  # Non-critical

    # Build response
    return AgentRouteResponse(
        session_id=result_state.get("session_id", ""),
        response=result_state.get("final_response", "I processed your request."),
        category=result_state.get("category"),
        agents_used=list(result_state.get("results", {}).keys()),
        execution_path=result_state.get("graph_execution_path", []),
        total_latency_ms=result_state.get("total_latency_ms", 0.0),
        total_tokens=result_state.get("total_tokens", 0),
        errors=[
            {"agent": e["agent"], "error": e["error"]}
            for e in result_state.get("errors", [])
        ],
    )


@router.post("/process", response_model=AgentRouteResponse)
async def direct_agent_call(
    request: AgentRouteRequest,
    agent_name: str = Query(..., description="Agent to call directly"),
    current_user: UserResponse = Depends(get_current_user),
):
    """
    Call a specific agent directly, bypassing the router and planner.

    Useful for explicit agent selection from the UI.
    """
    registry = get_agent_registry()
    agent = registry.get(agent_name)

    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found. Available: {registry.get_names()}",
        )

    state = create_initial_state(
        user_id=current_user.id,
        message=request.message,
        conversation_id=request.conversation_id,
        attachments=request.attachments,
        metadata=request.metadata,
    )

    # Skip router — go directly to the specified agent
    state["selected_agents"] = [agent_name]
    state["category"] = agent_name.upper()

    monitor = get_agent_monitor()
    monitor.start_session(state)

    try:
        state = await agent.safe_process(state)
    except Exception as exc:
        monitor.end_session(state)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")

    # Compose response through the response agent
    from agents_v2.response_agent import ResponseAgent
    responder = ResponseAgent()
    state = await responder.safe_process(state)

    monitor.end_session(state)

    return AgentRouteResponse(
        session_id=state.get("session_id", ""),
        response=state.get("final_response", f"{agent_name} processed your request."),
        category=agent_name.upper(),
        agents_used=[agent_name],
        execution_path=[agent_name, "response"],
        total_latency_ms=state.get("total_latency_ms", 0.0),
        total_tokens=state.get("total_tokens", 0),
        errors=[
            {"agent": e["agent"], "error": e["error"]}
            for e in state.get("errors", [])
        ],
    )


@router.get("", response_model=List[AgentInfo])
async def list_agents(
    current_user: UserResponse = Depends(get_current_user),
):
    """List all available agents with their metadata."""
    registry = get_agent_registry()
    agents = []

    for name, agent in registry.get_all().items():
        agents.append(AgentInfo(
            name=agent.name,
            model=agent.model_name,
            description=agent.description,
            available=True,
        ))

    return agents


@router.get("/metrics")
async def get_agent_metrics(
    current_user: UserResponse = Depends(get_current_user),
):
    """Get global agent execution metrics and per-agent summaries."""
    monitor = get_agent_monitor()

    return {
        "global": monitor.get_global_stats(),
        "per_agent": {
            name: summary.to_dict()
            for name, summary in monitor.get_agent_summary().items()
        },
        "category_breakdown": monitor.get_category_breakdown(),
    }


@router.get("/logs")
async def get_agent_logs(
    limit: int = Query(50, ge=1, le=500),
    agent: Optional[str] = Query(None, description="Filter by agent name"),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: UserResponse = Depends(get_current_user),
):
    """Get agent execution logs from MongoDB."""
    from database.mongodb import mongodb

    query: Dict[str, Any] = {"user_id": current_user.id}
    if agent:
        query["agent_name"] = agent
    if status:
        query["status"] = status

    cursor = mongodb.agent_logs.find(query).sort("created_at", -1).limit(limit)
    logs = await cursor.to_list(length=limit)

    from database.schemas import serialize_doc
    return [serialize_doc(log) for log in logs]


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: str,
    current_user: UserResponse = Depends(get_current_user),
):
    """Get detailed metrics for a specific session."""
    from database.mongodb import mongodb

    session = await mongodb.analytics.find_one({
        "session_id": session_id,
        "user_id": current_user.id,
    })

    if not session:
        # Check in-memory monitor
        monitor = get_agent_monitor()
        metrics = monitor.get_session_metrics(session_id)
        if metrics:
            return metrics.to_dict()
        raise HTTPException(status_code=404, detail="Session not found")

    from database.schemas import serialize_doc
    return serialize_doc(session)
