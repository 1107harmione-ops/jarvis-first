"""
Agent Registry
==============
Central registry for agent discovery and lookup.
Agents register themselves with their metadata so the graph,
router, and monitoring system can discover them dynamically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from agents_v2.base import BaseAgent


class AgentRegistry:
    """
    Singleton registry for all available agents.

    Agents register themselves with their class, model, and capabilities.
    The router queries this registry to decide which agent to route to.
    """

    _instance: Optional["AgentRegistry"] = None

    def __new__(cls) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: Dict[str, BaseAgent] = {}
            cls._instance._initialized = False
        return cls._instance

    def register(self, agent: BaseAgent) -> None:
        """Register an agent instance by its name."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def get_all(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return dict(self._agents)

    def get_names(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self._agents.keys())

    def get_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        Get agents that have a specific capability.
        Capabilities are matched against agent descriptions.
        """
        capability_lower = capability.lower()
        results = []
        for agent in self._agents.values():
            if capability_lower in agent.description.lower():
                results.append(agent)
        return results

    def is_registered(self, name: str) -> bool:
        return name in self._agents

    @property
    def count(self) -> int:
        return len(self._agents)

    def reset(self) -> None:
        """Clear all registered agents (useful for testing)."""
        self._agents.clear()


_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get or create the global AgentRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
