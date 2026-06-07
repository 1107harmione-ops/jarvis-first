"""
Task Agent
==========
Manages tasks, reminders, and scheduling through natural language.

Capabilities:
- Create tasks with title, description, priority, due date
- List / filter tasks by status or priority
- Update task fields (title, description, priority, due date)
- Complete tasks
- Delete tasks
- Get task statistics

Uses DeepSeek for structured data extraction from natural language,
and delegates all persistence to the shared TaskService.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID as _UUID

from backend.agents_v2.state import AgentState, ExecutionStatus
from backend.agents_v2.base import BaseAgent
from backend.agents_v2.registry import get_agent_registry
from backend.agents_v2.tools import AgentTools
from backend.llm.deepseek import deepseek
from backend.database.mongodb import mongodb
from backend.memory.short_term import stm
from backend.memory.long_term import ltm
from backend.memory.vector_memory import vector_memory
from backend.services.memory_service import memory_service
from backend.services.task_service import task_service
from backend.config.settings import settings


class TaskAgent(BaseAgent):
    """
    Task Agent — natural-language task management.

    Detects the user's task intent from their message, extracts structured
    fields via DeepSeek (for ``create`` and ``update``), and routes all
    operations through the shared ``TaskService`` singleton.

    Injects task results into ``state["shared_context"]["tasks"]`` so
    downstream agents or the response builder can access them.

    Operation types
    ---------------
    ``create``    parse NL fields, call ``task_service.create_task()``
    ``list``      call ``task_service.list_tasks()`` with optional filters
    ``update``    locate task by ID or title, apply field changes
    ``complete``  locate task by ID or title, mark done
    ``delete``    locate task by ID or title, remove
    ``stats``     aggregate counts via ``task_service.get_stats()``
    """

    # ── Intent patterns ───────────────────────────────────────

    _CREATE = re.compile(
        r"(?:remind\s*me|create\s*(?:a\s*)?task|"
        r"add\s*(?:a\s*)?task|new\s*task|"
        r"schedule|set\s*(?:a\s*)?reminder|"
        r"make\s*(?:a\s*)?note\s+to)",
        re.IGNORECASE,
    )
    _LIST = re.compile(
        r"(?:what\s*(?:are|'re|is)\s*(?:my|the)\s*tasks|"
        r"show\s*(?:my\s*)?tasks|list\s*tasks|"
        r"(?:pending|active|open)\s*tasks|"
        r"what.*(?:due|todo|to-do|to do))",
        re.IGNORECASE,
    )
    _UPDATE = re.compile(
        r"(?:update|change|modify|edit|reschedule|"
        r"move|rename|rep\w*\s*task)",
        re.IGNORECASE,
    )
    _COMPLETE = re.compile(
        r"(?:mark\s*(?:as\s*)?(?:done|completed|finished)|"
        r"complete\s*task|done\s*with|"
        r"finish\s*task|set\s*done)",
        re.IGNORECASE,
    )
    _DELETE = re.compile(
        r"(?:delete|remove|cancel|erase|destroy|trash)\s*task",
        re.IGNORECASE,
    )
    _STATS = re.compile(
        r"(?:task\s*(?:stats?|statistics|summary|report|count)|"
        r"how\s+many\s+tasks|"
        r"show\s*(?:my\s*)?task\s*(?:stats|status))",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__(
            name="task",
            model_name="deepseek",
            system_prompt="""You are JARVIS's Task Agent. You manage tasks, reminders, and scheduling.

You can:
- Create tasks: "remind me to X at Y" / "create a task to X"
- List tasks: "what are my tasks" / "show pending tasks"
- Update tasks: "change the due date of X to Y"
- Complete tasks: "mark X as done" / "complete task X"
- Delete tasks: "remove task X"
- Get stats: "how many tasks do I have"

Always parse the user's intent and extract structured task data.""",
            description="Manages tasks, reminders, and scheduling",
        )

    # ── Main entry point ──────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Detect task operation, execute it, and return updated state."""
        message = state["message"]
        user_id = state["user_id"]
        operation, query = self._detect_operation(message)

        handler_map: Dict[str, Any] = {
            "create": self._handle_create,
            "list": self._handle_list,
            "update": self._handle_update,
            "complete": self._handle_complete,
            "delete": self._handle_delete,
            "stats": self._handle_stats,
        }

        handler = handler_map.get(operation)

        if handler is None:
            state["final_response"] = (
                "I'm not sure what task operation you'd like. "
                "Try: create, list, complete, update, delete, or stats."
            )
            return state

        try:
            result = await handler(user_id, query if operation != "stats" else message)
        except Exception as exc:
            result = {
                "response": (
                    f"Sorry, I encountered an error processing your task "
                    f"request: {exc}"
                )
            }

        # Inject into shared context for downstream use
        state.setdefault("shared_context", {})
        state["shared_context"]["tasks"] = result.get("tasks", [])
        state["final_response"] = result.get("response", "")

        # Log
        await self._store_agent_log(
            state=state,
            action=f"task_{operation}",
            input_summary=message[:200],
            output_summary=result.get("response", "")[:200],
        )

        return state

    # ── Operation detection ───────────────────────────────────

    def _detect_operation(self, message: str) -> Tuple[str, str]:
        """
        Classify the task intent.

        Returns
            (operation_key, cleaned_query)
        """
        msg = message.strip()

        for pattern, op in [
            (self._STATS, "stats"),
            (self._COMPLETE, "complete"),
            (self._DELETE, "delete"),
            (self._UPDATE, "update"),
            (self._LIST, "list"),
            (self._CREATE, "create"),
        ]:
            match = pattern.search(msg)
            if match:
                query = msg[match.end() :].strip().lstrip(":,;. ")
                return op, query

        return "unknown", msg

    # ── Create ────────────────────────────────────────────────

    async def _handle_create(self, user_id: str, query: str) -> Dict[str, Any]:
        """Parse NL task fields via DeepSeek and create the task."""
        if not query:
            return {"response": "What task would you like me to create?", "tasks": []}

        try:
            fields = await deepseek.extract_json(
                system_prompt="""Extract task details from the user's request.

Respond with **valid JSON only** — no markdown, no explanation:

{
  "title": "Short task title",
  "description": "Optional longer description or empty string",
  "priority": "low" | "medium" | "high",
  "due_date": "ISO 8601 date string or null if not specified",
  "tags": ["tag1", "tag2"] or empty array
}

Rules:
- ALWAYS provide a title.
- due_date should be a full ISO 8601 string (e.g. "2026-06-10T17:00:00Z")
  or null.
- priority defaults to "medium" if not specified.""",
                user_message=query,
            )
        except Exception:
            fields = {"title": query, "description": "", "priority": "medium", "due_date": None, "tags": []}

        title = str(fields.get("title", query))[:200]
        description = str(fields.get("description", ""))[:1000]
        priority = str(fields.get("priority", "medium")).lower()
        if priority not in ("low", "medium", "high"):
            priority = "medium"

        due_at: Optional[datetime] = None
        raw_due = fields.get("due_date")
        if raw_due:
            try:
                due_at = datetime.fromisoformat(str(raw_due).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                due_at = None

        tags = list(fields.get("tags", [])) if isinstance(fields.get("tags"), list) else []

        created = await task_service.create_task(
            user_id=user_id,
            title=title,
            description=description,
            priority=priority,
            due_at=due_at,
            tags=tags,
        )

        response = f"✅ Task created: **{title}** (priority: {priority})"
        if due_at:
            response += f", due {due_at.strftime('%b %d at %H:%M')}"
        return {"response": response, "tasks": [created]}

    # ── List ──────────────────────────────────────────────────

    async def _handle_list(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        List tasks with optional status / priority filters derived
        from the query text.
        """
        status: Optional[str] = None
        priority: Optional[str] = None

        q_lower = query.lower()

        # Status hints
        if any(w in q_lower for w in ("pending", "active", "open", "incomplete", "not done")):
            status = "pending"
        elif any(w in q_lower for w in ("completed", "done", "finished")):
            status = "completed"
        elif any(w in q_lower for w in ("failed", "errored")):
            status = "failed"

        # Priority hints
        if "high" in q_lower:
            priority = "high"
        elif "low" in q_lower:
            priority = "low"
        elif "medium" in q_lower or "normal" in q_lower:
            priority = "medium"

        tasks = await task_service.list_tasks(
            user_id=user_id,
            status=status,
            priority=priority,
            limit=50,
        )

        if not tasks:
            if status:
                return {"response": f"No **{status}** tasks found.", "tasks": []}
            return {"response": "No tasks found. Create one with *remind me to ...*", "tasks": []}

        # Build a pretty summary
        lines: List[str] = [f"📋 **Tasks** ({len(tasks)}):\n"]
        for t in tasks:
            title = t.get("title", "Untitled")
            pri = t.get("priority", "medium")
            due = t.get("due_at")
            due_str = f" — due {due[:10]}" if due else ""
            lines.append(f"- [{pri}] {title}{due_str}")

        return {"response": "\n".join(lines), "tasks": tasks}

    # ── Update ────────────────────────────────────────────────

    async def _handle_update(self, user_id: str, query: str) -> Dict[str, Any]:
        """Find the task by ID or title and apply the requested changes."""
        if not query:
            return {"response": "Which task would you like to update?", "tasks": []}

        task, _ = await self._find_task(user_id, query)
        if task is None:
            return {"response": f"I couldn't find a task matching *{query[:50]}*.", "tasks": []}

        task_id = task.get("id", task.get("_id"))
        if not task_id:
            return {"response": "Could not determine task ID.", "tasks": []}

        # Let DeepSeek parse what fields to update
        try:
            update_fields = await deepseek.extract_json(
                system_prompt="""You are updating a task. The user wants to change some fields.

Current task: {title}

Respond with **valid JSON only**:
{
  "title": "new title or null to keep",
  "description": "new description or null",
  "priority": "low|medium|high or null",
  "due_date": "ISO 8601 string or null",
  "tags": ["tag1", ...] or null
}

Only include fields that the user explicitly wants to change. Set others to null.""".format(title=task.get("title", "Untitled")),
                user_message=query,
            )
        except Exception:
            return {"response": "I couldn't parse the update details. Please try again.", "tasks": [task]}

        # Build updates dict, skipping nulls
        updates: Dict[str, Any] = {}
        if update_fields.get("title"):
            updates["title"] = str(update_fields["title"])[:200]
        if update_fields.get("description"):
            updates["description"] = str(update_fields["description"])[:1000]
        if update_fields.get("priority") in ("low", "medium", "high"):
            updates["priority"] = update_fields["priority"]

        raw_due = update_fields.get("due_date")
        if raw_due:
            try:
                updates["due_at"] = datetime.fromisoformat(str(raw_due).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if update_fields.get("tags") is not None:
            updates["tags"] = list(update_fields["tags"]) if isinstance(update_fields["tags"], list) else []

        if not updates:
            return {"response": "I didn't detect any changes to apply.", "tasks": [task]}

        updated = await task_service.update_task(task_id, user_id, updates)
        if updated:
            return {"response": f"✅ Task **{updated.get('title')}** updated.", "tasks": [updated]}
        return {"response": "Task update failed.", "tasks": [task]}

    # ── Complete ──────────────────────────────────────────────

    async def _handle_complete(self, user_id: str, query: str) -> Dict[str, Any]:
        """Mark a task as completed by ID or title."""
        if not query:
            return {"response": "Which task would you like to complete?", "tasks": []}

        task, match_type = await self._find_task(user_id, query)
        if task is None:
            return {"response": f"I couldn't find a task matching *{query[:50]}*.", "tasks": []}

        task_id = task.get("id", task.get("_id"))
        if not task_id:
            return {"response": "Could not determine task ID.", "tasks": []}

        success = await task_service.complete_task(task_id, user_id)
        if success:
            return {
                "response": f"✅ Task **{task.get('title')}** marked as completed.",
                "tasks": [task],
            }
        return {"response": f"Failed to complete task **{task.get('title')}**.", "tasks": [task]}

    # ── Delete ────────────────────────────────────────────────

    async def _handle_delete(self, user_id: str, query: str) -> Dict[str, Any]:
        """Delete a task by ID or title."""
        if not query:
            return {"response": "Which task would you like to delete?", "tasks": []}

        task, match_type = await self._find_task(user_id, query)
        if task is None:
            return {"response": f"I couldn't find a task matching *{query[:50]}*.", "tasks": []}

        task_id = task.get("id", task.get("_id"))
        if not task_id:
            return {"response": "Could not determine task ID.", "tasks": []}

        success = await task_service.delete_task(task_id, user_id)
        if success:
            return {"response": f"🗑️ Task **{task.get('title')}** deleted.", "tasks": []}
        return {"response": f"Failed to delete task **{task.get('title')}**.", "tasks": [task]}

    # ── Stats ─────────────────────────────────────────────────

    async def _handle_stats(self, user_id: str, _query: str) -> Dict[str, Any]:
        """Aggregate task counts."""
        stats = await task_service.get_stats(user_id)

        response = (
            "📊 **Task Statistics**\n\n"
            f"• Total:     **{stats.get('total', 0):,}**\n"
            f"• Pending:   **{stats.get('pending', 0):,}**\n"
            f"• Completed: **{stats.get('completed', 0):,}**\n"
            f"• Failed:    **{stats.get('failed', 0):,}**\n"
            f"• Cancelled: **{stats.get('cancelled', 0):,}**"
        )
        return {"response": response, "tasks": []}

    # ── Helpers ───────────────────────────────────────────────

    async def _find_task(
        self, user_id: str, identifier: str
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Try to locate a task by:
        1. Exact ID (ObjectId hex or UUID)
        2. Title text match among pending tasks

        Returns
            (task_dict or None, match_type: "id" | "title" | "")
        """
        # --- Attempt ID match ---
        # MongoDB ObjectId is a 24-char hex string
        cleaned = identifier.strip().strip("'\"")

        # Check if it looks like a MongoDB ObjectId
        if re.match(r"^[0-9a-fA-F]{24}$", cleaned):
            task = await task_service.get_task(cleaned, user_id)
            if task:
                return task, "id"

        # Handle `_id` or `id` from serialized docs
        task = await task_service.get_task(cleaned, user_id)
        if task:
            return task, "id"

        # --- Fallback: search by title ---
        tasks = await task_service.list_tasks(user_id, limit=100)
        q_lower = cleaned.lower()
        for t in tasks:
            title = (t.get("title") or "").lower()
            if q_lower in title or title in q_lower:
                return t, "title"

        return None, ""
