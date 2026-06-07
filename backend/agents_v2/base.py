"""
Base Agent Abstraction
======================
All agents inherit from BaseAgent, which provides:
- Standard process() interface
- Memory integration (read context, log actions)
- Token usage tracking
- Error handling wrapper
- LLM call helper with fallback
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.agents_v2.state import (
    AgentNodeResult,
    AgentState,
    ExecutionStatus,
    create_initial_state,
)
from backend.agents_v2.tools import AgentTools


class BaseAgent(ABC):
    """
    Abstract base class for all JARVIS agents.

    Each agent subclass must implement `process()` which takes the
    current AgentState and returns an updated AgentState with results
    populated in state["results"][self.name].
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        system_prompt: str,
        description: str,
        tools: Optional[AgentTools] = None,
    ):
        self.name = name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.description = description
        self.tools = tools or AgentTools()

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Execute this agent's logic on the current state.

        Must update state["results"][self.name] with an AgentNodeResult
        and may modify any other fields (shared_context, memory_context, etc.)
        """
        ...

    # ── Execution wrapper with error handling ──────────────────

    async def safe_process(self, state: AgentState) -> AgentState:
        """
        Wraps process() with timing, error handling, and result recording.
        This is what the LangGraph node calls.
        """
        start = time.monotonic()
        step_id = f"{self.name}_{int(start * 1000)}"

        # Record entry in execution path
        state.setdefault("graph_execution_path", []).append(self.name)

        try:
            # Execute agent logic
            state = await self.process(state)

            latency = (time.monotonic() - start) * 1000
            result: AgentNodeResult = {
                "agent_name": self.name,
                "status": ExecutionStatus.SUCCESS,
                "output": state.get("final_response"),
                "error": None,
                "tokens_used": state.get("total_tokens", 0),
                "latency_ms": round(latency, 2),
                "metadata": {
                    "step_id": step_id,
                    "model": self.model_name,
                },
            }
            state["results"][self.name] = result
            state["step_results"][step_id] = result
            state["total_latency_ms"] += latency

        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

            result: AgentNodeResult = {
                "agent_name": self.name,
                "status": ExecutionStatus.FAILED,
                "output": None,
                "error": error_msg,
                "tokens_used": state.get("total_tokens", 0),
                "latency_ms": round(latency, 2),
                "metadata": {"step_id": step_id, "model": self.model_name},
            }
            state["results"][self.name] = result
            state["step_results"][step_id] = result
            state.setdefault("errors", []).append({
                "step_id": step_id,
                "agent": self.name,
                "error": error_msg,
                "retry_count": state.get("retry_count", 0),
                "timestamp": time.time(),
            })
            state["total_latency_ms"] += latency

        return state

    # ── LLM helper ────────────────────────────────────────────

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> Tuple[str, int]:
        """
        Call the appropriate LLM for this agent.

        Returns (response_text, tokens_used).
        Abstracts over model selection so agents don't import LLMs directly.
        """
        from llm.router import llm_router

        category = self.name.upper()  # e.g. "coding" → "CODING"
        model = llm_router.select_model(category)

        if stream:
            collected = []
            async for chunk in await llm_router.route(
                messages, category=category, stream=True, temperature=temperature,
            ):
                collected.append(chunk)
            text = "".join(collected)
        else:
            result = await llm_router.route(
                messages, category=category, stream=False, temperature=temperature,
                max_tokens=max_tokens,
            )
            text = result.get("content", result.get("text", ""))

        # Rough token estimate
        tokens_used = len(text.split()) * 2 + sum(len(m.get("content", "").split()) for m in messages) * 2
        return text, tokens_used

    async def _call_llm_with_fallback(
        self,
        messages: List[Dict[str, str]],
        primary_category: str,
        fallback_category: str = "GENERAL_CHAT",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, bool]:
        """
        Call LLM with automatic fallback to a different model.

        Returns (response_text, tokens_used, used_fallback).
        """
        from llm.router import llm_router

        try:
            text, tokens = await self._call_llm(messages, temperature, max_tokens)
            return text, tokens, False
        except Exception as primary_error:
            # Fallback to general chat model (DeepSeek)
            try:
                model = llm_router.select_model(fallback_category)
                result = await llm_router.route(
                    messages, category=fallback_category, stream=False,
                    temperature=temperature, max_tokens=max_tokens,
                )
                text = result.get("content", result.get("text", ""))
                tokens = len(text.split()) * 2 + sum(len(m.get("content", "").split()) for m in messages) * 2
                return text, tokens, True
            except Exception:
                raise primary_error  # Raise original error if fallback also fails

    # ── Memory helpers ────────────────────────────────────────

    async def _load_memory_context(self, state: AgentState) -> List[Dict[str, Any]]:
        """Load recent memory context for the user and inject into state."""
        try:
            from memory.short_term import stm
            context = await stm.get_context_window(state["user_id"], max_items=15)
            state["memory_context"] = context
            return context
        except Exception:
            state["memory_context"] = []
            return []

    async def _store_agent_log(
        self,
        state: AgentState,
        action: str,
        input_summary: str,
        output_summary: str,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """Log agent execution to the agent_logs collection."""
        try:
            from database.mongodb import mongodb
            from database.schemas import new_agent_log_doc

            log_doc = new_agent_log_doc(
                agent_name=self.name,
                session_id=state.get("session_id", "unknown"),
                action=action,
                user_id=state["user_id"],
                input_summary=input_summary[:200],
                output_summary=output_summary[:500] if output_summary else "",
                tokens_used=state.get("total_tokens", 0),
                duration_ms=state["results"].get(self.name, {}).get("latency_ms", 0),
                status=status,
                error=error[:500] if error else None,
            )
            await mongodb.agent_logs.insert_one(log_doc)
        except Exception:
            pass  # Logging failure should not break execution

    # ── Utility ───────────────────────────────────────────────

    def _build_system_messages(self, user_message: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Build the standard message list with system prompt and context."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user_message})
        return messages

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.name}] model={self.model_name}>"
