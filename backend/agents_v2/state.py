"""
Agent State Definitions
=======================
TypedDict-based state for LangGraph workflows.
Carries all context across the agent execution graph.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4


class ExecutionStatus(str, Enum):
    """Status of a single agent execution node."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class WorkflowType(str, Enum):
    """Type of agent workflow to execute."""
    SINGLE = "single"          # One agent, direct response
    SEQUENTIAL = "sequential"  # Agents in sequence
    PARALLEL = "parallel"      # Agents in parallel groups
    HIERARCHICAL = "hierarchical"  # Planner → sub-agents → response


# ──────────────────────────────────────────────
# Nested state structures
# ──────────────────────────────────────────────


class AgentPlanStep(TypedDict):
    """A single step in a multi-agent execution plan."""
    step_id: str
    agent: str                    # Agent name (e.g. "research", "coding")
    input: str                    # Instruction for this step
    depends_on: List[str]         # Step IDs that must complete first
    expected_output: str          # What this step should produce
    max_retries: int              # Per-step retry limit
    timeout_seconds: int          # Per-step timeout
    parallel_group: Optional[str] # Group ID for parallel execution


class AgentPlan(TypedDict):
    """Full execution plan created by PlannerAgent."""
    goal: str
    workflow_type: WorkflowType
    steps: List[AgentPlanStep]
    parallel_groups: Dict[str, List[str]]  # group_id → [step_ids]


class AgentNodeResult(TypedDict):
    """Result of a single agent execution node."""
    agent_name: str
    status: ExecutionStatus
    output: Optional[str]
    error: Optional[str]
    tokens_used: int
    latency_ms: float
    metadata: Optional[Dict[str, Any]]


class AgentError(TypedDict):
    """Record of an error during agent execution."""
    step_id: str
    agent: str
    error: str
    retry_count: int
    timestamp: float


# ──────────────────────────────────────────────
# Main AgentState graph state
# ──────────────────────────────────────────────


class AgentState(TypedDict):
    """
    Complete state flowing through the LangGraph agent workflow.

    This single TypedDict is the entire state of the multi-agent system.
    Every LangGraph node reads from and writes to this state.
    """
    # ── Input ─────────────────────────────────
    user_id: str
    message: str
    conversation_id: Optional[str]
    attachments: Optional[List[Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]]

    # ── Router output ─────────────────────────
    category: Optional[str]         # TaskCategory as string
    confidence: float               # Classification confidence (0-1)
    detected_intent: Optional[str]  # Human-readable intent
    selected_agents: List[str]      # Agent(s) chosen for this request

    # ── Planner output ────────────────────────
    plan: Optional[AgentPlan]
    current_step_index: int
    execution_order: List[str]      # Ordered list of agent names to execute

    # ── Execution ─────────────────────────────
    results: Dict[str, AgentNodeResult]  # agent_name → result
    step_results: Dict[str, AgentNodeResult]  # step_id → result
    shared_context: Dict[str, Any]  # Cross-agent shared data

    # ── Memory context ────────────────────────
    memory_context: List[Dict[str, Any]]

    # ── Error recovery ────────────────────────
    errors: List[AgentError]
    retry_count: int
    max_retries: int
    fallback_activated: bool

    # ── Final ─────────────────────────────────
    final_response: Optional[str]
    response_agent: Optional[str]   # Which agent produced the response

    # ── Monitoring ────────────────────────────
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_tokens: int
    total_latency_ms: float
    graph_execution_path: List[str]  # Ordered list of nodes traversed


def create_initial_state(
    user_id: str,
    message: str,
    conversation_id: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> AgentState:
    """Factory to create a fresh AgentState with defaults."""
    return AgentState(
        # Input
        user_id=user_id,
        message=message,
        conversation_id=conversation_id,
        attachments=attachments or [],
        metadata=metadata or {},
        # Router
        category=None,
        confidence=0.0,
        detected_intent=None,
        selected_agents=[],
        # Planner
        plan=None,
        current_step_index=0,
        execution_order=[],
        # Execution
        results={},
        step_results={},
        shared_context={},
        # Memory
        memory_context=[],
        # Error
        errors=[],
        retry_count=0,
        max_retries=max_retries,
        fallback_activated=False,
        # Final
        final_response=None,
        response_agent=None,
        # Monitoring
        session_id=uuid4().hex[:12],
        start_time=0.0,
        end_time=None,
        total_tokens=0,
        total_latency_ms=0.0,
        graph_execution_path=[],
    )
