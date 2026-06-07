"""
Response Agent
==============
Composes the final user-facing response from all agent execution results.

The ResponseAgent is the terminal node in the LangGraph workflow. It:
1. Collects outputs from all executed agents (``state["step_results"]``)
2. Aggregates shared context and memory context
3. Uses DeepSeek to compose a coherent, personality-rich response
4. Handles partial results gracefully when some agents have failed
5. Updates ``state["final_response"]`` and ``state["response_agent"]``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.agents_v2.state import AgentState, ExecutionStatus
from backend.agents_v2.base import BaseAgent
from backend.llm.deepseek import deepseek


class ResponseAgent(BaseAgent):
    """Composes final responses from multi-agent execution results.

    This agent runs after all other agents have completed (or failed).  It
    synthesises their outputs into a single coherent answer, adding JARVIS's
    characteristic personality.

    Behaviour:
    - Builds a structured summary of every step result (successful or failed)
    - Injects shared context and memory context for continuity
    - Uses DeepSeek with a personality-rich system prompt
    - Falls back to a concatenation of results if the LLM call fails
    - Logs the final response to memory for future context
    """

    _RESPONSE_SYSTEM_PROMPT: str = (
        "You are JARVIS, an elite AI assistant with a sharp, witty, and highly "
        "competent personality. You speak with confidence and precision, like "
        "a cross between Tony Stark's JARVIS and a world-class technical "
        "lead.\n\n"

        "Your communication style:\n"
        "- Be direct and confident — you know your stuff\n"
        "- Use a touch of wit and personality, but never at the cost of clarity\n"
        "- Structure complex answers with clear sections, bullet points, or "
        "numbered steps when helpful\n"
        "- Acknowledge what the user asked for, then deliver\n"
        "- If something failed, be transparent about it and offer alternatives\n"
        "- Use technical precision when discussing code, architecture, or data\n"
        "- Keep responses concise but complete — respect the user's time\n\n"

        "You are composing the FINAL response from multiple agent outputs. "
        "Synthesise the information naturally — do not just list what each agent "
        "did.  Present the answer as if you figured it out yourself."
    )

    def __init__(self) -> None:
        super().__init__(
            name="response",
            model_name="deepseek",
            system_prompt=self._RESPONSE_SYSTEM_PROMPT,
            description="Composes final responses from agent results",
        )

    async def process(self, state: AgentState) -> AgentState:
        """Compose the final response from all execution results.

        Steps:
        1. Build a structured summary of all step results
        2. Gather shared context and memory context
        3. Call DeepSeek with the aggregated information
        4. Handle partial failures gracefully
        5. Update ``state["final_response"]`` and ``state["response_agent"]``

        Args:
            state: The final ``AgentState`` after all agents have executed.

        Returns:
            Updated ``AgentState`` with ``final_response`` and
            ``response_agent`` populated.
        """
        start_time = time.monotonic()

        # ── Step 1: Build execution summary ─────────────────────────────
        execution_summary = self._build_execution_summary(state)
        shared_context = state.get("shared_context", {})
        memory_context = state.get("memory_context", [])
        user_message = state.get("message", "")
        plan = state.get("plan")

        # ── Step 2: Detect failure status ────────────────────────────────
        all_successful, partial_failures = self._check_execution_status(state)

        # ── Step 3: Compose via DeepSeek ─────────────────────────────────
        try:
            response_text = await self._compose_response(
                user_message=user_message,
                execution_summary=execution_summary,
                shared_context=shared_context,
                memory_context=memory_context,
                plan=plan,
                all_successful=all_successful,
                partial_failures=partial_failures,
            )
        except Exception:
            # If LLM fails, build a simple fallback response
            response_text = self._fallback_response(
                execution_summary, partial_failures,
            )

        # ── Step 4: Compute response latency ─────────────────────────────
        latency_ms = (time.monotonic() - start_time) * 1000

        # ── Step 5: Persist to state ─────────────────────────────────────
        state["final_response"] = response_text
        state["response_agent"] = self.name
        state["end_time"] = time.time()
        state["total_latency_ms"] = state.get("total_latency_ms", 0.0) + latency_ms
        state.setdefault("graph_execution_path", []).append(self.name)

        return state

    # ── Execution summary ───────────────────────────────────────────────

    def _build_execution_summary(self, state: AgentState) -> str:
        """Build a structured text summary of all step results.

        Iterates through ``state["step_results"]`` in insertion order
        (step_results preserves the order keys were added) and formats
        each as a readable block.
        """
        step_results = state.get("step_results", {})
        plan = state.get("plan")

        if not step_results:
            return "No agent results were produced."

        lines: List[str] = []
        plan_steps = {s["step_id"]: s for s in (plan.get("steps", []) if plan else [])}

        lines.append(f"Total steps executed: {len(step_results)}")
        lines.append("")

        for step_id, result in step_results.items():
            agent_name = result.get("agent_name", "unknown")
            status = result.get("status", ExecutionStatus.UNKNOWN)

            # Parse status enum or string
            if isinstance(status, ExecutionStatus):
                status_str = status.value
            else:
                status_str = str(status)

            output = result.get("output") or ""
            error = result.get("error")
            tokens = result.get("tokens_used", 0)
            latency = result.get("latency_ms", 0.0)

            # Include the step's original input if available
            step_info = plan_steps.get(step_id, {})
            step_input = step_info.get("input", "")

            lines.append(f"--- Step: {step_id} [{agent_name}] ({status_str}) ---")
            if step_input:
                lines.append(f"  Task: {step_input[:200]}")
            if output:
                # Truncate very long outputs for the summary
                if len(output) > 1000:
                    lines.append(f"  Output: {output[:1000]}...")
                else:
                    lines.append(f"  Output: {output}")
            if error:
                lines.append(f"  Error: {error[:300]}")
            lines.append(f"  (tokens: {tokens}, latency: {latency:.1f}ms)")
            lines.append("")

        return "\n".join(lines)

    def _check_execution_status(self, state: AgentState) -> tuple:
        """Check overall execution status.

        Returns:
            A tuple of ``(all_successful, partial_failures)`` where
            ``partial_failures`` is a list of (step_id, agent_name, error)
            tuples for failed steps.
        """
        step_results = state.get("step_results", {})
        failures: List[tuple] = []

        for step_id, result in step_results.items():
            status = result.get("status", ExecutionStatus.FAILED)
            if isinstance(status, ExecutionStatus):
                status_val = status.value
            else:
                status_val = str(status)

            if status_val in ("failed", "timeout", "skipped"):
                failures.append((
                    step_id,
                    result.get("agent_name", "unknown"),
                    result.get("error", "No error details"),
                ))

        all_successful = len(failures) == 0
        return all_successful, failures

    # ── Response composition ────────────────────────────────────────────

    async def _compose_response(
        self,
        user_message: str,
        execution_summary: str,
        shared_context: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        plan: Optional[Dict[str, Any]],
        all_successful: bool,
        partial_failures: List[tuple],
    ) -> str:
        """Use DeepSeek to compose a personality-rich final response.

        Builds a comprehensive prompt with all execution context and
        delegates to ``deepseek.chat()`` for the actual composition.
        """
        # Build the user message for the response LLM
        parts: List[str] = [
            "## Original user request",
            user_message,
            "",
            "## Agent execution summary",
            execution_summary,
        ]

        # Add plan context if available
        if plan:
            goal = plan.get("goal", "")
            steps = plan.get("steps", [])
            parts.append("## Original plan")
            parts.append(f"Goal: {goal}")
            for step in steps:
                parts.append(
                    f"  - {step.get('step_id', '?')}: "
                    f"[{step.get('agent', '?')}] {step.get('input', '')[:150]}"
                )
            parts.append("")

        # Add shared context
        if shared_context:
            ctx_str = "\n".join(
                f"  {k}: {str(v)[:200]}"
                for k, v in shared_context.items()
            )
            parts.append("## Shared context")
            parts.append(ctx_str)
            parts.append("")

        # Add recent memory
        if memory_context:
            recent = memory_context[-5:]
            mem_lines = [
                f"  - {item.get('content', str(item))[:200]}" for item in recent
            ]
            parts.append("## Recent memory")  # (misspelling intentional to match field name)
            parts.extend(mem_lines)
            parts.append("")

        # Failure note
        if not all_successful:
            failure_lines = [
                f"  - Step {sid} ({agent}): {err[:150]}"
                for sid, agent, err in partial_failures
            ]
            parts.append("## Partial failures (handle gracefully)")
            parts.extend(failure_lines)
            parts.append(
                "Note: Some steps failed. Acknowledge this briefly but "
                "focus on what did succeed."
            )
            parts.append("")

        parts.append(
            "Now compose your final response to the user. Be yourself — "
            "confident, helpful, and sharp. Synthesise the information "
            "naturally."
        )

        user_content = "\n".join(parts)

        messages = [
            {"role": "system", "content": self._RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        result = await deepseek.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )

        # Extract text from response
        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = result.get("content", result.get("text", ""))

        if not text:
            text = self._fallback_response(execution_summary, partial_failures)

        return text

    # ── Fallback ────────────────────────────────────────────────────────

    def _fallback_response(
        self,
        execution_summary: str,
        partial_failures: List[tuple],
    ) -> str:
        """Build a simple concatenated fallback response when the LLM call fails.

        This ensures the user always receives a response even if the
        composition model is unavailable.
        """
        lines: List[str] = [
            "Here's what I found:",
            "",
        ]

        # Extract outputs from the summary
        for line in execution_summary.split("\n"):
            if line.startswith("  Output:"):
                lines.append(line.replace("  Output:", "").strip())
            elif line.startswith("  Error:"):
                lines.append(f"[Note: {line.replace('  Error:', '').strip()}]")

        if not partial_failures:
            lines.append("")
            lines.append("Everything executed successfully.")

        return "\n".join(lines)

    async def _store_response_to_memory(self, state: AgentState) -> None:
        """Store the final response into short-term memory for context continuity.

        This is called after the response is composed so future requests
        have context.
        """
        try:
            from memory.short_term import stm

            response_text = state.get("final_response", "")
            if response_text and len(response_text) > 20:
                await stm.add(
                    user_id=state["user_id"],
                    role="assistant",
                    content=response_text[:500],
                    metadata={
                        "agent": self.name,
                        "session_id": state.get("session_id", ""),
                        "category": state.get("category", ""),
                    },
                )
        except Exception:
            # Memory storage failure must not break the response
            pass
