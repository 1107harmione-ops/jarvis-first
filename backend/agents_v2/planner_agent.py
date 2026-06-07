"""
Planner Agent
=============
Decomposes complex tasks into executable multi-step plans.

The PlannerAgent receives tasks that the router has identified as
complex or multi-agent. It:
1. Uses DeepSeek with structured output to produce a detailed plan
2. Detects dependencies between steps for correct ordering
3. Groups independent steps for parallel execution
4. Populates ``state["plan"]`` and ``state["execution_order"]``

The resulting ``AgentPlan`` drives the LangGraph execution phase.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.agents_v2.state import (
    AgentPlan,
    AgentPlanStep,
    AgentState,
    ExecutionStatus,
    WorkflowType,
)
from backend.agents_v2.base import BaseAgent
from backend.llm.deepseek import deepseek


class PlannerAgent(BaseAgent):
    """Breaks complex tasks into structured, executable plans.

    This agent is invoked when the router detects that a task requires
    multiple steps across one or more agents. It produces an ``AgentPlan``
    that the LangGraph executor follows, handling both sequential chains
    and parallel group execution.

    The planner analyses:
    - The user's goal and success criteria
    - Which agents are needed and in what order
    - Dependencies between steps (e.g. research must finish before coding)
    - Opportunities for parallel execution
    """

    # System prompt used when calling DeepSeek for plan generation.
    _PLANNER_SYSTEM_PROMPT: str = (
        "You are JARVIS's Planner Agent, responsible for breaking down complex "
        "user requests into clear, executable multi-step plans.\n\n"

        "Rules:\n"
        "1. Identify the overall goal of the request.\n"
        "2. Break it down into discrete steps. Each step must be assigned to "
        "exactly one agent.\n"
        "3. Determine dependencies: if step B needs output from step A, "
        "B.depends_on must include A's step_id.\n"
        "4. Group independent steps (no dependency chain between them) into "
        "parallel groups so they can execute concurrently.\n"
        "5. Choose the right workflow type:\n"
        "   - 'single' for one-step tasks\n"
        "   - 'sequential' for ordered, dependent steps\n"
        "   - 'parallel' when steps have no interdependencies\n"
        "   - 'hierarchical' for complex tasks with sub-plans\n\n"

        "Available agents:\n"
        "- coding: Code generation, review, debug, architecture\n"
        "- research: Web search, deep research, summarization\n"
        "- vision: Image analysis, OCR, screenshot analysis\n"
        "- memory: Store, recall, search memories\n"
        "- task: Task management, reminders, scheduling\n"
        "- planner: Complex multi-step tasks, project planning\n"
        "- utility: Quick answers, math, formatting, simple questions\n\n"

        "Respond with valid JSON only, no markdown wrapping."
    )

    def __init__(self) -> None:
        super().__init__(
            name="planner",
            model_name="deepseek",
            system_prompt=self._PLANNER_SYSTEM_PROMPT,
            description="Breaks complex tasks into executable multi-agent plans",
        )

        # Maximum number of steps allowed per plan to prevent runaway plans.
        self._max_steps: int = 15

    async def process(self, state: AgentState) -> AgentState:
        """Decompose the user's request into a structured plan.

        Steps:
        1. Build a prompt with the user's message, attachment info, and memory
        2. Call ``deepseek.extract_json()`` to obtain a structured plan
        3. Validate, normalise, and sanitise the plan
        4. Detect dependencies and build parallel execution groups
        5. Populate ``state["plan"]`` and ``state["execution_order"]``

        Args:
            state: The current ``AgentState`` with ``message`` and optionally
                   ``attachments`` and ``memory_context``.

        Returns:
            Updated ``AgentState`` with ``plan`` and ``execution_order`` populated.
        """
        message = state.get("message", "")
        attachments = state.get("attachments", [])
        memory_context = state.get("memory_context", [])

        # ── Step 1: Build plan request context ───────────────────────────
        user_content = self._build_plan_prompt(message, attachments, memory_context)

        # ── Step 2: Generate plan via DeepSeek ────────────────────────────
        try:
            raw_plan = await deepseek.extract_json(
                system_prompt=self._PLANNER_SYSTEM_PROMPT,
                user_message=user_content,
            )
        except Exception as exc:
            # If structured extraction fails, create a minimal fallback plan
            raw_plan = self._fallback_plan(message, str(exc))

        # ── Step 3: Validate and normalise ───────────────────────────────
        plan = self._parse_plan(raw_plan, message)
        if plan is None:
            plan = self._fallback_plan(message, "Plan parsing failed")

        # ── Step 4: Detect dependencies and parallel groups ──────────────
        plan = self._resolve_dependencies(plan)
        plan = self._build_parallel_groups(plan)

        # ── Step 5: Build execution order ────────────────────────────────
        execution_order = self._compute_execution_order(plan)

        # ── Step 6: Persist to state ─────────────────────────────────────
        state["plan"] = plan
        state["execution_order"] = execution_order
        state["current_step_index"] = 0
        state.setdefault("graph_execution_path", []).append(self.name)

        return state

    # ── Prompt building ─────────────────────────────────────────────────

    def _build_plan_prompt(
        self,
        message: str,
        attachments: List[Dict[str, Any]],
        memory_context: List[Dict[str, Any]],
    ) -> str:
        """Build a detailed user prompt for plan generation.

        Includes attachment descriptions and recent memory context to help
        the planner produce context-aware plans.
        """
        parts: List[str] = [
            f"User request: {message}",
        ]

        if attachments:
            attachment_lines = []
            for att in attachments:
                if isinstance(att, dict):
                    name = att.get("name") or att.get("path") or att.get("url", "unknown")
                    att_type = att.get("type", att.get("mime_type", "unknown"))
                    attachment_lines.append(f"  - {name} ({att_type})")
                elif isinstance(att, str):
                    attachment_lines.append(f"  - {att}")
            if attachment_lines:
                parts.append("Attachments:\n" + "\n".join(attachment_lines))

        if memory_context:
            recent = memory_context[-5:]
            ctx_lines = [
                f"  - {item.get('content', str(item))[:200]}"
                for item in recent
            ]
            parts.append("Recent memory context:\n" + "\n".join(ctx_lines))

        # Append the required output schema
        parts.append(
            "\nProduce a plan in this exact JSON structure:\n"
            "{\n"
            '  "goal": "Overall goal of the task",\n'
            '  "workflow_type": "single|sequential|parallel|hierarchical",\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_id": "step_1",\n'
            '      "agent": "agent_name",\n'
            '      "input": "Detailed instruction for this step",\n'
            '      "expected_output": "What this step should produce",\n'
            '      "depends_on": ["step_id_of_prerequisite"],\n'
            '      "max_retries": 2,\n'
            '      "timeout_seconds": 60\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Maximum {self._max_steps} steps."
        )

        return "\n\n".join(parts)

    # ── Plan parsing ────────────────────────────────────────────────────

    def _parse_plan(self, raw: Dict[str, Any], original_message: str) -> Optional[AgentPlan]:
        """Validate and normalise the raw JSON from DeepSeek into an ``AgentPlan``.

        Returns ``None`` if the structure is unusable.
        """
        # Validate top-level structure
        goal = raw.get("goal", "").strip()
        if not goal:
            goal = original_message[:200]

        # Parse workflow type
        workflow_type_str = raw.get("workflow_type", "sequential")
        try:
            workflow_type = WorkflowType(workflow_type_str)
        except ValueError:
            workflow_type = WorkflowType.SEQUENTIAL

        # Parse steps
        raw_steps = raw.get("steps", [])
        if not raw_steps or not isinstance(raw_steps, list):
            return None

        steps: List[AgentPlanStep] = []
        for idx, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                continue

            step = self._normalise_step(raw_step, idx)
            if step is not None:
                steps.append(step)

        if not steps:
            return None

        # For single-step plans, correct the workflow type
        if len(steps) == 1 and workflow_type != WorkflowType.SINGLE:
            workflow_type = WorkflowType.SINGLE

        # Build parallel_groups map (may be enriched by _build_parallel_groups)
        parallel_groups: Dict[str, List[str]] = {}

        return AgentPlan(
            goal=goal,
            workflow_type=workflow_type,
            steps=steps,
            parallel_groups=parallel_groups,
        )

    def _normalise_step(self, raw_step: Dict[str, Any], index: int) -> Optional[AgentPlanStep]:
        """Normalise a single step dict into an ``AgentPlanStep``.

        Fills sensible defaults for missing fields.
        """
        step_id = raw_step.get("step_id", "").strip()
        if not step_id:
            step_id = f"step_{index + 1}"

        agent = raw_step.get("agent", "").strip().lower()
        if not agent:
            agent = "utility"

        # Validate agent name against allowed set
        valid_agents = {
            "coding", "research", "vision", "memory",
            "task", "planner", "utility",
        }
        if agent not in valid_agents:
            agent = "utility"

        step_input = raw_step.get("input", "").strip()
        if not step_input:
            step_input = f"Execute step {step_id}"

        expected_output = raw_step.get("expected_output", "").strip()
        if not expected_output:
            expected_output = f"Output of {step_id}"

        raw_depends = raw_step.get("depends_on", [])
        depends_on: List[str] = (
            [str(d).strip() for d in raw_depends if d]
            if isinstance(raw_depends, list)
            else []
        )

        max_retries = int(raw_step.get("max_retries", 2))
        timeout_seconds = int(raw_step.get("timeout_seconds", 60))

        return AgentPlanStep(
            step_id=step_id,
            agent=agent,
            input=step_input,
            depends_on=depends_on,
            expected_output=expected_output,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            parallel_group=None,
        )

    # ── Dependency resolution ──────────────────────────────────────────

    def _resolve_dependencies(self, plan: AgentPlan) -> AgentPlan:
        """Resolve and validate step dependencies.

        Ensures all ``depends_on`` references point to valid step IDs.
        Orphan references are removed.  Steps are reordered so that every
        step appears after all of its dependencies (topological sort).
        """
        steps = list(plan["steps"])
        step_ids = {s["step_id"] for s in steps}

        # Remove invalid dependency references
        for step in steps:
            step["depends_on"] = [
                dep for dep in step["depends_on"] if dep in step_ids
            ]

        # Topological sort (Kahn's algorithm)
        valid_ids = {s["step_id"] for s in steps}
        in_degree: Dict[str, int] = {s["step_id"]: 0 for s in steps}
        adjacency: Dict[str, List[str]] = {s["step_id"]: [] for s in steps}

        for step in steps:
            for dep in step["depends_on"]:
                if dep in valid_ids:
                    adjacency.setdefault(dep, []).append(step["step_id"])
                    in_degree[step["step_id"]] = in_degree.get(step["step_id"], 0) + 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        sorted_ids: List[str] = []

        while queue:
            # Sort within same dependency level for determinism
            queue.sort()
            current = queue.pop(0)
            sorted_ids.append(current)
            for neighbour in adjacency.get(current, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        # If sort excluded some IDs (cycle), append them at the end
        remaining = [s["step_id"] for s in steps if s["step_id"] not in sorted_ids]
        sorted_ids.extend(remaining)

        # Rebuild steps in sorted order
        step_map = {s["step_id"]: s for s in steps}
        sorted_steps = [step_map[sid] for sid in sorted_ids if sid in step_map]

        return AgentPlan(
            goal=plan["goal"],
            workflow_type=plan["workflow_type"],
            steps=sorted_steps,
            parallel_groups=plan.get("parallel_groups", {}),
        )

    # ── Parallel group construction ─────────────────────────────────────

    def _build_parallel_groups(self, plan: AgentPlan) -> AgentPlan:
        """Group independent steps for parallel execution.

        Steps that have no dependency chain between them are assigned to
        the same parallel group.  Each group can execute concurrently
        within the LangGraph executor.
        """
        steps = list(plan["steps"])
        if len(steps) <= 1:
            return plan

        # Assign a "level" to each step based on its longest dependency chain
        step_map = {s["step_id"]: s for s in steps}
        levels: Dict[str, int] = {}

        def _compute_level(step_id: str) -> int:
            """Compute the longest dependency chain length for a step."""
            if step_id in levels:
                return levels[step_id]
            step = step_map.get(step_id)
            if not step or not step["depends_on"]:
                levels[step_id] = 0
                return 0
            max_dep = max(_compute_level(dep) for dep in step["depends_on"])
            levels[step_id] = max_dep + 1
            return max_dep + 1

        for step in steps:
            _compute_level(step["step_id"])

        # Group steps by level — all steps at the same level can run in parallel
        groups: Dict[str, List[str]] = {}
        for step in steps:
            level = levels.get(step["step_id"], 0)
            group_id = f"group_{level}"
            groups.setdefault(group_id, []).append(step["step_id"])

        # Assign group IDs back to steps
        for step in steps:
            level = levels.get(step["step_id"], 0)
            step["parallel_group"] = f"group_{level}"

        # Single-step groups don't need parallel execution
        single_groups = {gid for gid, sids in groups.items() if len(sids) <= 1}
        for step in steps:
            if step["parallel_group"] in single_groups:
                step["parallel_group"] = None

        # Remove empty/single groups from the map
        parallel_groups = {
            gid: sids for gid, sids in groups.items()
            if len(sids) > 1
        }

        return AgentPlan(
            goal=plan["goal"],
            workflow_type=plan["workflow_type"],
            steps=steps,
            parallel_groups=parallel_groups,
        )

    # ── Execution ordering ─────────────────────────────────────────────

    def _compute_execution_order(self, plan: AgentPlan) -> List[str]:
        """Compute the ordered list of step IDs for execution.

        The order respects dependencies: steps appear after all their
        dependencies.  This is the list the LangGraph executor iterates over.
        """
        return [step["step_id"] for step in plan["steps"]]

    # ── Fallback ────────────────────────────────────────────────────────

    def _fallback_plan(self, message: str, reason: str) -> Dict[str, Any]:
        """Create a minimal fallback plan when structured generation fails.

        The fallback creates a single-step utility-agent plan so the system
        can still respond to the user.
        """
        return {
            "goal": message[:200],
            "workflow_type": "single",
            "steps": [
                {
                    "step_id": "step_1",
                    "agent": "utility",
                    "input": message,
                    "expected_output": "A helpful response to the user",
                    "depends_on": [],
                    "max_retries": 2,
                    "timeout_seconds": 60,
                },
            ],
        }
