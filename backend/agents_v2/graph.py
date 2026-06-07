"""
LangGraph Agent Workflow
========================
Production-grade LangGraph state machine that orchestrates all agents.

Graph structure:
    Entry → RouterAgent
              ├── [direct single-agent] → Agent → ResponseAgent → MemoryAgent → Exit
              └── [complex/multi] → PlannerAgent → ExecuteStep(loop) → ResponseAgent → MemoryAgent → Exit

The ExecuteStep node dynamically iterates through plan steps,
calling the appropriate agent for each step and managing dependencies.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from backend.agents_v2.state import (
    AgentState,
    ExecutionStatus,
    WorkflowType,
    create_initial_state,
)
from backend.agents_v2.registry import get_agent_registry


# ──────────────────────────────────────────────
# Agent execution tracking
# ──────────────────────────────────────────────

_EXECUTED_STEP_IDS: set = set()


def _reset_execution_tracking() -> None:
    """Reset step tracking (call at start of each top-level request)."""
    _EXECUTED_STEP_IDS.clear()


# ══════════════════════════════════════════════
# LangGraph Node Implementations
# ══════════════════════════════════════════════


async def router_node(state: AgentState) -> AgentState:
    """
    LangGraph node: RouterAgent.
    Classifies the request, selects agents, and determines workflow type.
    """
    registry = get_agent_registry()
    router = registry.get("router")
    if not router:
        state["category"] = "GENERAL_CHAT"
        state["selected_agents"] = ["utility"]
        state["confidence"] = 0.5
        return state

    state["start_time"] = time.time()
    return await router.safe_process(state)


async def planner_node(state: AgentState) -> AgentState:
    """
    LangGraph node: PlannerAgent.
    Decomposes complex tasks into multi-step execution plans.
    Only invoked when router determines the task needs planning.
    """
    registry = get_agent_registry()
    planner = registry.get("planner")
    if not planner:
        state["final_response"] = "Planner is not available. Please try a simpler request."
        return state

    return await planner.safe_process(state)


async def execute_step_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Execute a single step from the plan.
    Reads state["current_step_index"], dispatches to the right agent,
    stores the result, and increments the counter.

    This node is invoked in a loop via conditional edges:
    execute_step → (more steps) → execute_step
                 → (done)       → response
    """
    plan = state.get("plan")
    if not plan or state["current_step_index"] >= len(plan["steps"]):
        # No plan or all steps done — move to response
        return state

    step = plan["steps"][state["current_step_index"]]
    agent_name = step["agent"]
    step_id = step["step_id"]

    # Check if this step's dependencies are met
    deps = step.get("depends_on", [])
    for dep_id in deps:
        dep_result = state["step_results"].get(dep_id)
        if not dep_result or dep_result["status"] != ExecutionStatus.SUCCESS:
            # Dependency not met — skip this step
            state["step_results"][step_id] = {
                "agent_name": agent_name,
                "status": ExecutionStatus.SKIPPED,
                "output": None,
                "error": f"Dependency not met: {dep_id}",
                "tokens_used": 0,
                "latency_ms": 0.0,
                "metadata": {"step_id": step_id},
            }
            state["current_step_index"] += 1
            return state

    # Dispatch to the right agent
    registry = get_agent_registry()
    agent = registry.get(agent_name)

    if not agent:
        state["step_results"][step_id] = {
            "agent_name": agent_name,
            "status": ExecutionStatus.FAILED,
            "output": None,
            "error": f"Agent '{agent_name}' not found in registry",
            "tokens_used": 0,
            "latency_ms": 0.0,
            "metadata": {"step_id": step_id},
        }
        state["current_step_index"] += 1
        return state

    # Inject step-specific input into shared context
    state["shared_context"]["current_step_input"] = step["input"]
    state["shared_context"]["current_step_expected"] = step["expected_output"]

    # Execute the agent
    try:
        state = await agent.safe_process(state)
        result = state["results"].get(agent_name, {})
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["step_id"] = step_id
        state["step_results"][step_id] = result
    except Exception as exc:
        state["step_results"][step_id] = {
            "agent_name": agent_name,
            "status": ExecutionStatus.FAILED,
            "output": None,
            "error": str(exc),
            "tokens_used": 0,
            "latency_ms": 0.0,
            "metadata": {"step_id": step_id},
        }

    state["current_step_index"] += 1
    return state


async def response_node(state: AgentState) -> AgentState:
    """
    LangGraph node: ResponseAgent.
    Composes the final response from all agent results.
    """
    registry = get_agent_registry()
    responder = registry.get("response")
    if not responder:
        # Fallback: build basic response from available results
        parts = []
        for step_id, result in state.get("step_results", {}).items():
            if result.get("output"):
                parts.append(result["output"])
        state["final_response"] = "\n\n".join(parts) if parts else "I processed your request."
        return state

    state = await responder.safe_process(state)
    return state


async def memory_log_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Log the conversation exchange to short-term memory.
    Runs as the final step to ensure every interaction is remembered.
    """
    from memory.short_term import stm

    user_message = state.get("message", "")
    final_response = state.get("final_response", "")

    if user_message and final_response:
        try:
            await stm.store(
                user_id=state["user_id"],
                content=f"User: {user_message}\nJARVIS: {final_response[:500]}",
                tags=["conversation", "agent_response"],
                importance_score=0.4,
                metadata={
                    "session_id": state.get("session_id"),
                    "category": state.get("category"),
                    "agents_used": list(state.get("results", {}).keys()),
                    "execution_path": state.get("graph_execution_path", []),
                },
            )
        except Exception:
            pass  # Memory storage failure should not break the response

    state["end_time"] = time.time()
    state["total_latency_ms"] = round(
        (state["end_time"] - state["start_time"]) * 1000, 2
    )
    return state


# ══════════════════════════════════════════════
# Conditional edge functions
# ══════════════════════════════════════════════


def route_from_router(state: AgentState) -> str:
    """
    Decide where to go after the Router node.

    Returns:
        "planner"  → Complex/multi-agent task needs planning
        "direct"   → Single agent or general chat → go to response
    """
    selected = state.get("selected_agents", [])
    category = state.get("category", "")

    # Complex tasks needing planning
    if category in ("PLANNING", "COMPLEX"):
        return "planner"

    # Multi-agent tasks
    if len(selected) > 1:
        return "planner"

    # General chat or single-agent tasks
    category_single = {
        "CODING": "direct",
        "CODE_REVIEW": "direct",
        "RESEARCH": "direct",
        "SUMMARIZATION": "direct",
        "VISION": "direct",
        "OCR": "direct",
        "MEMORY": "direct",
        "TASK_MANAGEMENT": "direct",
        "EXTRACTION": "direct",
    }
    return category_single.get(category, "direct")


def route_after_planner(state: AgentState) -> str:
    """
    Decide where to go after the Planner node.

    Returns:
        "execute"  → Start executing the plan
        "response" → No plan needed (single step already handled)
    """
    if state.get("plan") and len(state["plan"].get("steps", [])) > 0:
        return "execute"
    return "response"


def route_after_step(state: AgentState) -> str:
    """
    Decide whether to continue execution or finish.

    Returns:
        "continue" → More steps to execute
        "response" → All steps done, compose final response
    """
    plan = state.get("plan")
    if plan and state["current_step_index"] < len(plan["steps"]):
        return "continue"
    return "response"


# ══════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════


class AgentGraph:
    """
    Wrapper around the LangGraph workflow.

    Provides:
    - build(): Construct the compiled graph
    - aexecute(state): Run a request through the graph
    """

    def __init__(self):
        self._compiled_graph = None

    def build(self):
        """Build and compile the LangGraph workflow."""
        try:
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(AgentState)

            # Register nodes
            workflow.add_node("router", router_node)
            workflow.add_node("planner", planner_node)
            workflow.add_node("execute_step", execute_step_node)
            workflow.add_node("response", response_node)
            workflow.add_node("memory_log", memory_log_node)

            # Entry point
            workflow.set_entry_point("router")

            # Router → conditional routing
            workflow.add_conditional_edges(
                "router",
                route_from_router,
                {
                    "planner": "planner",
                    "direct": "response",
                },
            )

            # Planner → execute or skip to response
            workflow.add_conditional_edges(
                "planner",
                route_after_planner,
                {
                    "execute": "execute_step",
                    "response": "response",
                },
            )

            # Execute step loop
            workflow.add_conditional_edges(
                "execute_step",
                route_after_step,
                {
                    "continue": "execute_step",
                    "response": "response",
                },
            )

            # Response → memory log → end
            workflow.add_edge("response", "memory_log")
            workflow.add_edge("memory_log", END)

            self._compiled_graph = workflow.compile()
            return self._compiled_graph

        except ImportError:
            # LangGraph not installed — use fallback executor
            self._compiled_graph = None
            return None

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Execute a request through the agent workflow.

        Uses LangGraph if available, otherwise falls back to
        a simple sequential executor.
        """
        _reset_execution_tracking()
        state["start_time"] = time.time()

        if self._compiled_graph:
            try:
                result = await self._compiled_graph.ainvoke(state)
                return result
            except Exception as graph_error:
                # Fall through to sequential executor
                state.setdefault("errors", []).append({
                    "step_id": "graph",
                    "agent": "graph",
                    "error": f"LangGraph execution failed: {graph_error}",
                    "retry_count": 0,
                    "timestamp": time.time(),
                })

        # Sequential fallback executor
        return await self._fallback_execute(state)

    async def _fallback_execute(self, state: AgentState) -> AgentState:
        """
        Fallback executor that runs agents sequentially when LangGraph
        is not available or the graph execution fails.
        """
        registry = get_agent_registry()

        # Step 1: Router
        router = registry.get("router")
        if router:
            state = await router.safe_process(state)

        # Step 2: Planner (if needed)
        selected = state.get("selected_agents", [])
        category = state.get("category", "")
        needs_planning = category in ("PLANNING", "COMPLEX") or len(selected) > 1

        if needs_planning:
            planner = registry.get("planner")
            if planner:
                state = await planner.safe_process(state)

        # Step 3: Execute plan or direct agent
        plan = state.get("plan")
        if plan and plan.get("steps"):
            for _ in range(len(plan["steps"])):
                state = await execute_step_node(state)
        elif len(selected) == 1:
            agent = registry.get(selected[0])
            if agent:
                state = await agent.safe_process(state)
        elif not selected or category in ("GENERAL_CHAT", "ROUTING"):
            # General chat — use utility agent
            utility = registry.get("utility")
            if utility:
                state = await utility.safe_process(state)

        # Step 4: Response composition
        state = await response_node(state)

        # Step 5: Memory logging
        state = await memory_log_node(state)

        return state


# ── Factory ───────────────────────────────────

_graph_instance: Optional[AgentGraph] = None


def create_agent_graph() -> AgentGraph:
    """Get or create the global AgentGraph singleton."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = AgentGraph()
        _graph_instance.build()
    return _graph_instance
