"""
Agent Monitoring & Observability
=================================
Tracks agent execution metrics, latency, token usage, success rates,
and provides a dashboard API endpoint for real-time monitoring.

All metrics are stored in MongoDB's `analytics` and `agent_logs` collections.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.agents_v2.state import AgentState, ExecutionStatus
from backend.agents_v2.registry import get_agent_registry


class AgentMonitor:
    """
    Agent monitoring and observability system.

    Collects metrics per agent, per session, and globally.
    Provides query methods for dashboard display.
    """

    def __init__(self):
        self._session_metrics: Dict[str, "SessionMetrics"] = {}
        self._active_sessions: Dict[str, float] = {}  # session_id → start_time

    # ── Per-session tracking ──────────────────

    def start_session(self, state: AgentState) -> str:
        """Record the start of an agent execution session."""
        session_id = state.get("session_id", "unknown")
        self._active_sessions[session_id] = time.time()
        return session_id

    def end_session(self, state: AgentState) -> "SessionMetrics":
        """Record the end of a session and compute final metrics."""
        session_id = state.get("session_id", "unknown")
        start = self._active_sessions.pop(session_id, state.get("start_time", time.time()))
        end = state.get("end_time", time.time())
        duration = end - start

        metrics = SessionMetrics(
            session_id=session_id,
            user_id=state.get("user_id", "unknown"),
            category=state.get("category", "unknown"),
            duration_seconds=round(duration, 3),
            total_tokens=state.get("total_tokens", 0),
            agent_count=len(state.get("results", {})),
            execution_path=list(state.get("graph_execution_path", [])),
            success=self._compute_success(state),
            error_count=len(state.get("errors", [])),
            agents_used=[],
        )

        # Per-agent breakdown
        for agent_name, result in state.get("results", {}).items():
            metrics.agents_used.append(AgentMetric(
                agent_name=agent_name,
                status=result.get("status", ExecutionStatus.FAILED),
                latency_ms=result.get("latency_ms", 0),
                tokens_used=result.get("tokens_used", 0),
                error=result.get("error"),
            ))

        self._session_metrics[session_id] = metrics
        return metrics

    def _compute_success(self, state: AgentState) -> bool:
        """Determine if the overall execution was successful."""
        errors = state.get("errors", [])
        if errors:
            return False
        for result in state.get("results", {}).values():
            if result.get("status") == ExecutionStatus.FAILED:
                return False
        return True

    # ── Dashboard queries ─────────────────────

    def get_session_metrics(self, session_id: str) -> Optional["SessionMetrics"]:
        """Get metrics for a specific session."""
        return self._session_metrics.get(session_id)

    def get_recent_sessions(self, limit: int = 20) -> List["SessionMetrics"]:
        """Get most recent session metrics."""
        sorted_sessions = sorted(
            self._session_metrics.values(),
            key=lambda m: m.duration_seconds,
            reverse=True,
        )
        return sorted_sessions[:limit]

    def get_agent_summary(self) -> Dict[str, "AgentSummary"]:
        """
        Get aggregate metrics per agent.

        Returns {agent_name: AgentSummary}.
        """
        summaries: Dict[str, AgentSummary] = {}

        for metrics in self._session_metrics.values():
            for agent in metrics.agents_used:
                if agent.agent_name not in summaries:
                    summaries[agent.agent_name] = AgentSummary(
                        agent_name=agent.agent_name,
                        total_calls=0,
                        successful_calls=0,
                        failed_calls=0,
                        total_latency_ms=0.0,
                        total_tokens=0,
                        avg_latency_ms=0.0,
                        avg_tokens_per_call=0,
                    )
                summary = summaries[agent.agent_name]
                summary.total_calls += 1
                summary.total_latency_ms += agent.latency_ms
                summary.total_tokens += agent.tokens_used

                if agent.status == ExecutionStatus.SUCCESS:
                    summary.successful_calls += 1
                else:
                    summary.failed_calls += 1

        # Compute averages
        for summary in summaries.values():
            if summary.total_calls > 0:
                summary.avg_latency_ms = round(
                    summary.total_latency_ms / summary.total_calls, 2
                )
                summary.avg_tokens_per_call = summary.total_tokens // summary.total_calls

        return summaries

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global execution statistics."""
        total_sessions = len(self._session_metrics)
        if total_sessions == 0:
            return self._empty_global_stats()

        successful = sum(1 for m in self._session_metrics.values() if m.success)
        total_duration = sum(m.duration_seconds for m in self._session_metrics.values())
        total_tokens = sum(m.total_tokens for m in self._session_metrics.values())
        total_errors = sum(m.error_count for m in self._session_metrics.values())

        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful,
            "failed_sessions": total_sessions - successful,
            "success_rate": round(successful / total_sessions * 100, 1) if total_sessions else 0.0,
            "total_duration_seconds": round(total_duration, 3),
            "average_duration_seconds": round(total_duration / total_sessions, 3) if total_sessions else 0.0,
            "total_tokens_used": total_tokens,
            "average_tokens_per_session": total_tokens // total_sessions if total_sessions else 0,
            "total_errors": total_errors,
            "unique_agents": len(set(
                a.agent_name for m in self._session_metrics.values() for a in m.agents_used
            )),
            "active_sessions": len(self._active_sessions),
        }

    def _empty_global_stats(self) -> Dict[str, Any]:
        return {
            "total_sessions": 0,
            "successful_sessions": 0,
            "failed_sessions": 0,
            "success_rate": 0.0,
            "total_duration_seconds": 0.0,
            "average_duration_seconds": 0.0,
            "total_tokens_used": 0,
            "average_tokens_per_session": 0,
            "total_errors": 0,
            "unique_agents": 0,
            "active_sessions": 0,
        }

    def get_category_breakdown(self) -> Dict[str, int]:
        """Get count of sessions per request category."""
        breakdown: Dict[str, int] = {}
        for metrics in self._session_metrics.values():
            cat = metrics.category or "unknown"
            breakdown[cat] = breakdown.get(cat, 0) + 1
        return breakdown

    # ── Persistence ───────────────────────────

    async def persist_session(self, state: AgentState) -> None:
        """Persist session metrics to MongoDB."""
        try:
            from database.mongodb import mongodb

            metrics = self._session_metrics.get(state.get("session_id", ""))
            if not metrics:
                return

            doc = {
                "session_id": metrics.session_id,
                "user_id": metrics.user_id,
                "category": metrics.category,
                "duration_seconds": metrics.duration_seconds,
                "total_tokens": metrics.total_tokens,
                "success": metrics.success,
                "error_count": metrics.error_count,
                "execution_path": metrics.execution_path,
                "agents": [
                    {
                        "name": a.agent_name,
                        "status": a.status.value,
                        "latency_ms": a.latency_ms,
                        "tokens": a.tokens_used,
                        "error": a.error,
                    }
                    for a in metrics.agents_used
                ],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await mongodb.analytics.insert_one(doc)
        except Exception:
            pass  # Persistence failure is non-critical

    def clear(self) -> None:
        """Clear all in-memory metrics."""
        self._session_metrics.clear()
        self._active_sessions.clear()


# ── Data classes ──────────────────────────────


class SessionMetrics:
    """Metrics for a single agent execution session."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        category: str,
        duration_seconds: float,
        total_tokens: int,
        agent_count: int,
        execution_path: List[str],
        success: bool,
        error_count: int,
        agents_used: Optional[List["AgentMetric"]] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.category = category
        self.duration_seconds = duration_seconds
        self.total_tokens = total_tokens
        self.agent_count = agent_count
        self.execution_path = execution_path
        self.success = success
        self.error_count = error_count
        self.agents_used = agents_used or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "category": self.category,
            "duration_seconds": self.duration_seconds,
            "total_tokens": self.total_tokens,
            "agent_count": self.agent_count,
            "execution_path": self.execution_path,
            "success": self.success,
            "error_count": self.error_count,
            "agents": [a.to_dict() for a in self.agents_used],
        }


class AgentMetric:
    """Metrics for a single agent call within a session."""

    def __init__(
        self,
        agent_name: str,
        status: ExecutionStatus,
        latency_ms: float,
        tokens_used: int,
        error: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.status = status
        self.latency_ms = latency_ms
        self.tokens_used = tokens_used
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


class AgentSummary:
    """Aggregate metrics for a single agent across sessions."""

    def __init__(
        self,
        agent_name: str,
        total_calls: int = 0,
        successful_calls: int = 0,
        failed_calls: int = 0,
        total_latency_ms: float = 0.0,
        total_tokens: int = 0,
        avg_latency_ms: float = 0.0,
        avg_tokens_per_call: int = 0,
    ):
        self.agent_name = agent_name
        self.total_calls = total_calls
        self.successful_calls = successful_calls
        self.failed_calls = failed_calls
        self.total_latency_ms = total_latency_ms
        self.total_tokens = total_tokens
        self.avg_latency_ms = avg_latency_ms
        self.avg_tokens_per_call = avg_tokens_per_call

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round(self.successful_calls / self.total_calls * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens_per_call": self.avg_tokens_per_call,
        }


# ── Singleton ─────────────────────────────────

_agent_monitor: Optional[AgentMonitor] = None


def get_agent_monitor() -> AgentMonitor:
    """Get or create the global AgentMonitor singleton."""
    global _agent_monitor
    if _agent_monitor is None:
        _agent_monitor = AgentMonitor()
    return _agent_monitor
