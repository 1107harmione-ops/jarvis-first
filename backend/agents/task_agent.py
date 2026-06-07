"""
Task Agent — manages reminders, scheduled tasks, recurring tasks, and background jobs.
Integrates with MongoDB for persistent task storage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc, new_task_doc, serialize_doc, update_task_doc
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TaskAgent:
    """Specialized agent for task management.

    Handles CRUD for tasks, reminders, scheduled items, and recurring jobs.
    """

    def __init__(self) -> None:
        self.name = "task_agent"

    async def process(
        self,
        user_id: str,
        message: str,
        context: list[dict[str, str]] | None = None,
        attachments: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a task-related request."""
        start = time.monotonic()
        session_id = session_id or f"task_{int(start)}"

        task_type = self._detect_task_type(message)
        logger.info(
            "Task agent processing",
            extra={"task_type": task_type, "session_id": session_id, "user_id": user_id},
        )

        try:
            if task_type == "create":
                result = await self._create_task(user_id, message)
            elif task_type == "list":
                result = await self._list_tasks(user_id, message)
            elif task_type == "update":
                result = await self._update_task(user_id, message)
            elif task_type == "delete":
                result = await self._delete_task(user_id, message)
            elif task_type == "complete":
                result = await self._complete_task(user_id, message)
            else:
                result = await self._general_task_response(message)

            elapsed = (time.monotonic() - start) * 1000
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary=result.get("content", "")[:200],
                    duration_ms=elapsed,
                    status="success",
                )
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Task agent failed", extra={"error": str(exc), "session_id": session_id})
            return {"content": f"Task operation failed: {str(exc)}", "agent": self.name, "error": str(exc)}

    async def _create_task(self, user_id: str, message: str) -> dict[str, Any]:
        """Create a new task from natural language."""
        # Use LLM to extract task details
        from backend.llm.deepseek import deepseek

        extraction_prompt = f"""Extract task details from this request. Return JSON:
{{
    "title": "task title",
    "description": "detailed description",
    "priority": "low|medium|high|critical",
    "due_at": "ISO datetime or null",
    "scheduled_at": "ISO datetime or null",
    "recurring": "cron expression or null",
    "tags": ["tag1", "tag2"]
}}

Request: {message}"""

        try:
            extracted = await deepseek.extract_json(
                "You are a task extraction assistant. Extract structured task data from natural language.",
                extraction_prompt,
            )
        except Exception:
            extracted = {"title": message[:200], "description": message, "priority": "medium"}

        # Parse dates from natural language
        due_at = None
        if extracted.get("due_at"):
            try:
                due_at = datetime.fromisoformat(extracted["due_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                due_at = None

        scheduled_at = None
        if extracted.get("scheduled_at"):
            try:
                scheduled_at = datetime.fromisoformat(extracted["scheduled_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                scheduled_at = None

        doc = new_task_doc(
            user_id=user_id,
            title=extracted.get("title", message[:200]),
            description=extracted.get("description", ""),
            priority=extracted.get("priority", "medium"),
            due_at=due_at,
            scheduled_at=scheduled_at,
            recurring=extracted.get("recurring"),
            tags=extracted.get("tags", []),
        )
        result = await mongodb.tasks.insert_one(doc)
        task_id = str(result.inserted_id)

        content = f"""✅ **Task Created!**

**{extracted.get('title', message[:100])}**
- Priority: {extracted.get('priority', 'medium').upper()}
- Status: Pending
- Due: {due_at.strftime('%Y-%m-%d %H:%M UTC') if due_at else 'No due date'}
- Recurring: {extracted.get('recurring', 'No')}
- ID: `{task_id}`"""

        return {"content": content, "agent": self.name, "metadata": {"task_id": task_id, "action": "created"}}

    async def _list_tasks(self, user_id: str, message: str) -> dict[str, Any]:
        """List tasks with optional filters."""
        # Parse filter from message
        status_filter = None
        if "completed" in message.lower():
            status_filter = "completed"
        elif "pending" in message.lower() or "active" in message.lower():
            status_filter = "pending"
        elif "failed" in message.lower():
            status_filter = "failed"

        query: dict[str, Any] = {"user_id": user_id}
        if status_filter:
            query["status"] = status_filter

        cursor = mongodb.tasks.find(query, sort=[("created_at", -1)]).limit(20)
        tasks = await cursor.to_list(length=20)

        if not tasks:
            return {"content": "You have no tasks. Create one with a reminder or task request!", "agent": self.name}

        content = f"## Your Tasks ({len(tasks)})\n\n"
        for task in tasks:
            t = serialize_doc(task)
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌", "cancelled": "🚫"}
            emoji = status_emoji.get(t.get("status", "pending"), "📋")
            due = t.get("due_at", "")
            if isinstance(due, str) and due:
                due_str = due[:10]
            else:
                due_str = "No due date"
            content += f"{emoji} **{t.get('title', 'Untitled')}**\n"
            content += f"   Priority: {t.get('priority', 'medium').upper()} | Due: {due_str}\n"
            content += f"   ID: `{t.get('id', '')}`\n\n"

        return {"content": content, "agent": self.name, "metadata": {"task_count": len(tasks)}}

    async def _update_task(self, user_id: str, message: str) -> dict[str, Any]:
        """Update an existing task."""
        # Extract task ID and new values
        import re
        task_id_match = re.search(r'`([a-f0-9]+)`', message)
        if not task_id_match:
            return {"content": "Please specify the task ID to update (e.g., `abc123`).", "agent": self.name}

        task_id = task_id_match.group(1)

        # Build update from message
        from backend.llm.deepseek import deepseek
        try:
            extracted = await deepseek.extract_json(
                "Extract task fields to update from the request. Return JSON with only fields to change.",
                f"Update task {task_id}: {message}",
            )
        except Exception:
            extracted = {}

        updates: dict[str, Any] = {}
        if "title" in extracted:
            updates["title"] = extracted["title"]
        if "description" in extracted:
            updates["description"] = extracted["description"]
        if "priority" in extracted:
            updates["priority"] = extracted["priority"]
        if "status" in extracted:
            updates["status"] = extracted["status"]
        if extracted.get("due_at"):
            try:
                updates["due_at"] = datetime.fromisoformat(extracted["due_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        if not updates:
            return {"content": "No changes specified. Tell me what to update (title, priority, status, etc.).", "agent": self.name}

        result = await mongodb.tasks.update_one(
            {"_id": task_id, "user_id": user_id},
            {"$set": update_task_doc(**updates)},
        )

        if result.modified_count > 0:
            return {"content": f"✅ Task `{task_id}` updated successfully.", "agent": self.name, "metadata": {"task_id": task_id, "action": "updated"}}
        return {"content": f"Task `{task_id}` not found or no changes needed.", "agent": self.name}

    async def _delete_task(self, user_id: str, message: str) -> dict[str, Any]:
        """Delete a task."""
        import re
        task_id_match = re.search(r'`([a-f0-9]+)`', message)
        if not task_id_match:
            return {"content": "Please specify the task ID to delete (e.g., `abc123`).", "agent": self.name}

        task_id = task_id_match.group(1)
        result = await mongodb.tasks.delete_one({"_id": task_id, "user_id": user_id})

        if result.deleted_count > 0:
            return {"content": f"🗑️ Task `{task_id}` deleted.", "agent": self.name, "metadata": {"task_id": task_id, "action": "deleted"}}
        return {"content": f"Task `{task_id}` not found.", "agent": self.name}

    async def _complete_task(self, user_id: str, message: str) -> dict[str, Any]:
        """Mark a task as completed."""
        import re
        task_id_match = re.search(r'`([a-f0-9]+)`', message)
        if not task_id_match:
            # Mark most recent pending task as complete
            cursor = mongodb.tasks.find(
                {"user_id": user_id, "status": "pending"},
                sort=[("created_at", -1)],
                limit=1,
            )
            tasks = await cursor.to_list(length=1)
            if tasks:
                task_id = str(tasks[0]["_id"])
            else:
                return {"content": "No pending tasks to complete.", "agent": self.name}
        else:
            task_id = task_id_match.group(1)

        result = await mongodb.tasks.update_one(
            {"_id": task_id, "user_id": user_id},
            {"$set": update_task_doc(status="completed", completed_at=datetime.now(timezone.utc))},
        )

        if result.modified_count > 0:
            return {"content": f"✅ Task `{task_id}` marked as completed!", "agent": self.name, "metadata": {"task_id": task_id, "action": "completed"}}
        return {"content": f"Task `{task_id}` not found.", "agent": self.name}

    async def _general_task_response(self, message: str) -> dict[str, Any]:
        """General task help response."""
        return {
            "content": """I can help you manage tasks! Here's what I can do:

**Create tasks:** "Remind me to buy groceries tomorrow at 5pm"
**List tasks:** "Show my pending tasks"
**Complete tasks:** "Mark task \`abc123\` as done"
**Update tasks:** "Change priority of \`abc123\` to high"
**Delete tasks:** "Delete task \`abc123\`"

You can also set recurring tasks: "Remind me to stand up every hour" or "Daily standup reminder at 9am" """,
            "agent": self.name,
        }

    def _detect_task_type(self, message: str) -> str:
        """Detect the task operation type."""
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["remind", "create task", "new task", "set reminder", "schedule", "add task"]):
            return "create"
        if any(kw in msg_lower for kw in ["show", "list", "my tasks", "what tasks", "pending tasks"]):
            return "list"
        if any(kw in msg_lower for kw in ["update", "change", "modify", "edit"]):
            return "update"
        if any(kw in msg_lower for kw in ["delete", "remove", "cancel task"]):
            return "delete"
        if any(kw in msg_lower for kw in ["complete", "mark done", "mark as done", "finish"]):
            return "complete"
        return "general"


task_agent = TaskAgent()
