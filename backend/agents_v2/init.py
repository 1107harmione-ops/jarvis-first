"""
Agent System Initialization
============================
Initializes and registers all agents at application startup.
This is the single entry point for setting up the multi-agent system.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents_v2.registry import get_agent_registry
from agents_v2.graph import create_agent_graph

logger = logging.getLogger("jarvis.agents.init")


async def initialize_agent_system() -> None:
    """
    Initialize and register all agents.

    Called during application startup (main.py lifespan).
    Registers every agent with the AgentRegistry and builds the LangGraph.
    """
    registry = get_agent_registry()
    logger.info("Initializing agent system...")

    # ── Import agent classes ──────────────────
    # (lazy imports to avoid circular dependencies at module level)

    try:
        from agents_v2.router_agent import RouterAgent
        from agents_v2.planner_agent import PlannerAgent
        from agents_v2.coding_agent import CodingAgent
        from agents_v2.research_agent import ResearchAgent
        from agents_v2.vision_agent import VisionAgent
        from agents_v2.memory_agent import MemoryAgent
        from agents_v2.task_agent import TaskAgent
        from agents_v2.utility_agent import UtilityAgent
        from agents_v2.response_agent import ResponseAgent

        # ── Register agents ───────────────────────
        registry.register(RouterAgent())
        registry.register(PlannerAgent())
        registry.register(CodingAgent())
        registry.register(ResearchAgent())
        registry.register(VisionAgent())
        registry.register(MemoryAgent())
        registry.register(TaskAgent())
        registry.register(UtilityAgent())
        registry.register(ResponseAgent())

        logger.info(f"Registered {registry.count} agents: {registry.get_names()}")

        # ── Build LangGraph ───────────────────────
        graph = create_agent_graph()
        if graph._compiled_graph:
            logger.info("LangGraph compiled successfully")
        else:
            logger.warning("LangGraph not available — using fallback executor")

    except Exception as e:
        logger.error(f"Failed to initialize agent system: {e}", exc_info=True)
        raise


async def shutdown_agent_system() -> None:
    """Cleanup agent system resources during shutdown."""
    from agents_v2.monitor import get_agent_monitor

    monitor = get_agent_monitor()
    monitor.clear()
    logger.info("Agent system shut down")
