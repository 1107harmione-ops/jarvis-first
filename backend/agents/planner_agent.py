"""
Planner Agent — project planning, architecture design, roadmap creation, and task decomposition.
Breaks down complex requests into actionable plans.
"""

from __future__ import annotations

import json
import time
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc
from backend.llm.deepseek import deepseek
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PlannerAgent:
    """Specialized agent for planning and architecture design.

    Handles:
    - Project planning and roadmaps
    - Architecture design
    - Task decomposition
    - Sprint/iteration planning
    - Technical specification generation
    """

    def __init__(self) -> None:
        self.name = "planner_agent"

    async def process(
        self,
        user_id: str,
        message: str,
        context: list[dict[str, str]] | None = None,
        attachments: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a planning request."""
        start = time.monotonic()
        session_id = session_id or f"planner_{int(start)}"

        plan_type = self._detect_plan_type(message)
        logger.info(
            "Planner agent processing",
            extra={"plan_type": plan_type, "session_id": session_id, "user_id": user_id},
        )

        try:
            if plan_type == "architecture":
                result = await self._architecture_plan(message, context)
            elif plan_type == "roadmap":
                result = await self._roadmap(message, context)
            elif plan_type == "decomposition":
                result = await self._decomposition(message, context)
            elif plan_type == "specification":
                result = await self._specification(message, context)
            elif plan_type == "sprint":
                result = await self._sprint_plan(message, context)
            else:
                result = await self._general_plan(message, context)

            elapsed = (time.monotonic() - start) * 1000
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=plan_type,
                    input_summary=message[:200],
                    output_summary=result.get("content", "")[:200],
                    duration_ms=elapsed,
                    status="success",
                )
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Planner agent failed", extra={"error": str(exc), "session_id": session_id})
            return {"content": f"Planning failed: {str(exc)}", "agent": self.name, "error": str(exc)}

    async def _architecture_plan(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Create an architecture design plan."""
        system = """You are a senior software architect. Design a comprehensive architecture for the described system.

Return a structured plan covering:
1. **System Overview** — purpose and goals
2. **Architecture Diagram** (ASCII/description) — components and data flow
3. **Technology Stack** — with rationale for each choice
4. **Component Design** — each major component with responsibilities
5. **Data Model** — entities and relationships
6. **API Design** — endpoints and contracts
7. **Security Considerations**
8. **Scalability Strategy**
9. **Deployment Architecture**
10. **Implementation Phases**

Be specific and actionable. Include trade-offs where relevant."""

        messages = self._build_messages(system, message, context)
        response = await deepseek.chat(messages, temperature=0.5, max_tokens=8192)
        content = response["choices"][0]["message"]["content"]

        return {
            "content": content,
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"plan_type": "architecture"},
        }

    async def _roadmap(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Create a project roadmap."""
        system = """You are a product and engineering strategist. Create a detailed project roadmap.

Structure:
1. **Vision & Goals** — what we're building and why
2. **Phases** (timeline-based):
   - Phase 1: Foundation (weeks 1-2)
   - Phase 2: Core Features (weeks 3-4)
   - Phase 3: Enhancement (weeks 5-6)
   - Phase 4: Polish & Scale (weeks 7-8)
3. **Milestones** with deliverables
4. **Dependencies** between phases
5. **Resource Requirements**
6. **Risk Assessment**
7. **Success Metrics"""

        messages = self._build_messages(system, message, context)
        response = await deepseek.chat(messages, temperature=0.5, max_tokens=4096)

        return {
            "content": response["choices"][0]["message"]["content"],
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"plan_type": "roadmap"},
        }

    async def _decomposition(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Break down a complex task into actionable steps."""
        system = """You are a task decomposition expert. Break down the described project/task into small, actionable steps.

Return as structured JSON:
{
    "project": "Project name",
    "overview": "Brief description",
    "steps": [
        {
            "id": 1,
            "title": "Step title",
            "description": "What to do",
            "effort": "small|medium|large",
            "dependencies": [],
            "deliverable": "What this produces"
        }
    ],
    "estimated_total_effort": "X days/weeks",
    "recommended_approach": "Parallel or sequential execution strategy"
}

Make steps concrete and individually completable."""

        messages = self._build_messages(system, message, context)
        response = await deepseek.chat(messages, temperature=0.3, max_tokens=4096)
        content = response["choices"][0]["message"]["content"]

        # Try to parse and format the JSON
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content

            parsed = json.loads(json_str)
            steps = parsed.get("steps", [])
            formatted = f"""## {parsed.get('project', 'Project Plan')}

{parsed.get('overview', '')}

### Steps ({len(steps)} total)
"""
            for step in steps:
                effort_emoji = {"small": "🟢", "medium": "🟡", "large": "🔴"}
                formatted += f"""
**{step.get('id', '?')}. {step.get('title', '')}**
{effort_emoji.get(step.get('effort', 'medium'), '⚪')} Effort: {step.get('effort', 'medium')}
   {step.get('description', '')}
   📦 Deliverable: {step.get('deliverable', 'N/A')}
"""
            formatted += f"\n**Estimated total effort:** {parsed.get('estimated_total_effort', 'TBD')}\n"
            formatted += f"\n**Strategy:** {parsed.get('recommended_approach', 'Sequential execution')}"
            content = formatted
        except (json.JSONDecodeError, KeyError):
            pass  # Use raw response

        return {
            "content": content,
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"plan_type": "decomposition"},
        }

    async def _specification(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Generate a technical specification document."""
        system = """You are a technical writer. Create a detailed technical specification.

Include:
1. **Title & Version**
2. **Overview** — purpose, scope, audience
3. **Technical Requirements**
4. **System Architecture** — components, interactions
5. **Data Models** — schemas, relationships
6. **API Specifications** — endpoints, request/response formats
7. **Error Handling**
8. **Testing Strategy**
9. **Deployment Notes**
10. **Future Considerations"""

        messages = self._build_messages(system, message, context)
        response = await deepseek.chat(messages, temperature=0.3, max_tokens=8192)

        return {
            "content": response["choices"][0]["message"]["content"],
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"plan_type": "specification"},
        }

    async def _sprint_plan(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Create a sprint/iteration plan."""
        system = """You are an Agile sprint planner. Create a detailed sprint plan.

Structure:
1. **Sprint Goal** — what we aim to achieve
2. **Sprint Duration** — recommended length
3. **User Stories** (as "As a... I want... So that...")
4. **Task Breakdown** — technical tasks for each story
5. **Estimated Story Points**
6. **Team Allocation**
7. **Risks & Mitigations**
8. **Definition of Done**
9. **Review Criteria"""

        messages = self._build_messages(system, message, context)
        response = await deepseek.chat(messages, temperature=0.5, max_tokens=4096)

        return {
            "content": response["choices"][0]["message"]["content"],
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"plan_type": "sprint"},
        }

    async def _general_plan(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """General planning response."""
        system = """You are JARVIS's planning module. Help the user plan their project or task.

Provide:
- Clear structure and actionable steps
- Timeline estimates where relevant
- Dependencies and risks
- Recommended approach

Be practical and focused on execution."""

        messages = self._build_messages(system, message, context)
        response = await deepseek.chat(messages, temperature=0.5, max_tokens=4096)

        return {
            "content": response["choices"][0]["message"]["content"],
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"plan_type": "general"},
        }

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        context: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Build message list with context."""
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            for msg in context:
                if msg["role"] != "system":
                    messages.append(msg)
        # Ensure the user message is last
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_message})
        elif messages[-1]["role"] == "user":
            # Replace last user message
            messages[-1]["content"] = user_message
        return messages

    def _detect_plan_type(self, message: str) -> str:
        """Detect the type of planning request."""
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["architecture", "system design", "architect", "tech stack"]):
            return "architecture"
        if any(kw in msg_lower for kw in ["roadmap", "timeline", "milestone", "phase", "long-term"]):
            return "roadmap"
        if any(kw in msg_lower for kw in ["break down", "decompose", "step by step", "task list", "subtasks"]):
            return "decomposition"
        if any(kw in msg_lower for kw in ["specification", "spec", "technical spec", "design doc"]):
            return "specification"
        if any(kw in msg_lower for kw in ["sprint", "iteration", "agile", "story", "two-week"]):
            return "sprint"
        return "general"


# Global singleton
planner_agent = PlannerAgent()
