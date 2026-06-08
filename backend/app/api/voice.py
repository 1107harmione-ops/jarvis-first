"""Voice command API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.logger import get_logger
from app.notes.schemas import NoteCreate, NoteRead
from app.notes.service import note_service
from app.memory.service import memory_service
from app.reminders.schemas import ReminderCreate, ReminderRead
from app.reminders.service import reminder_service
from app.search.service import search_service
from app.tasks.schemas import TaskCreate, TaskRead
from app.tasks.service import task_service
from app.voice.router import IntentType, intent_router

logger = get_logger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])


class VoiceCommandRequest(BaseModel):
    text: str


class VoiceCommandResponse(BaseModel):
    intent: str
    spoken_response: str
    data: dict | None = None


@router.post("/command", response_model=VoiceCommandResponse)
async def process_voice_command(
    req: VoiceCommandRequest,
    db: AsyncSession = Depends(get_db),
):
    """Process a voice command text and execute the intent."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty command text")

    # Route intent
    result = intent_router.route(text)
    logger.info(
        "voice_command_routed",
        intent=result.type.value,
        confidence=result.confidence,
        text=text,
    )

    if not result.is_known():
        return VoiceCommandResponse(
            intent="UNKNOWN",
            spoken_response="Sorry, I didn't understand that. Can you rephrase?",
        )

    # Execute based on intent
    try:
        response_text, data = await _execute_intent(result, db)
    except Exception as e:
        logger.error("voice_command_error", intent=result.type.value, error=str(e))
        return VoiceCommandResponse(
            intent=result.type.value,
            spoken_response=f"Sorry, I encountered an error: {str(e)}",
        )

    return VoiceCommandResponse(
        intent=result.type.value,
        spoken_response=response_text,
        data=data,
    )


async def _execute_intent(
    result,
    db: AsyncSession,
) -> tuple[str, dict | None]:
    """Execute a routed intent and return (spoken_response, data)."""
    intent = result.type

    if intent == IntentType.TASK_CREATE:
        title = result.entities.get("title", result.raw_text)
        task = await task_service.create(db, TaskCreate(title=title))
        return f"Task created: {task.title}", {"task": TaskRead.model_validate(task).model_dump()}

    elif intent == IntentType.TASK_LIST:
        tasks, total = await task_service.list(db)
        if total == 0:
            return "You have no tasks.", {"tasks": [], "total": 0}
        task_titles = [t.title for t in tasks[:5]]
        summary = f"You have {total} tasks. " if total > 5 else ""
        task_list = ", ".join(task_titles)
        return f"{summary}Your tasks: {task_list}", {
            "tasks": [TaskRead.model_validate(t).model_dump() for t in tasks],
            "total": total,
        }

    elif intent == IntentType.TASK_COMPLETE:
        title_hint = result.entities.get("title", "").lower()
        tasks, total = await task_service.list(db, status="pending")
        if total == 0:
            return "You have no pending tasks to complete.", None
        target = next((t for t in tasks if title_hint in t.title.lower()), tasks[0])
        completed = await task_service.complete(db, target.id)
        return f"Task completed: {completed.title}", {"task": TaskRead.model_validate(completed).model_dump()}

    elif intent == IntentType.TASK_DELETE:
        title_hint = result.entities.get("title", "").lower()
        tasks, total = await task_service.list(db)
        if total == 0:
            return "You have no tasks to delete.", None
        target = next((t for t in tasks if title_hint in t.title.lower()), tasks[0])
        await task_service.delete(db, target.id)
        return f"Task deleted: {target.title}", None

    elif intent == IntentType.TASK_SEARCH:
        query = result.entities.get("query", result.raw_text.replace("search", "").replace("find", "").strip())
        tasks, total = await task_service.search(db, query)
        if total == 0:
            return f"No tasks found matching '{query}'.", {"tasks": [], "total": 0}
        task_list = ", ".join(t.title for t in tasks[:5])
        return f"Found {total} task{'s' if total != 1 else ''}: {task_list}", {
            "tasks": [TaskRead.model_validate(t).model_dump() for t in tasks],
            "total": total,
        }

    elif intent == IntentType.NOTE_CREATE:
        title = result.entities.get("title", result.raw_text)
        note = await note_service.create(db, NoteCreate(title=title))
        return f"Note created: {note.title}", {"note": NoteRead.model_validate(note).model_dump()}

    elif intent == IntentType.NOTE_SEARCH:
        query = result.entities.get("query", result.raw_text.replace("search", "").replace("find", "").strip())
        notes, total = await note_service.search(db, query)
        if total == 0:
            return f"No notes found matching '{query}'.", {"notes": [], "total": 0}
        note_list = ", ".join(n.title for n in notes[:5])
        return f"Found {total} note{'s' if total != 1 else ''}: {note_list}", {
            "notes": [NoteRead.model_validate(n).model_dump() for n in notes],
            "total": total,
        }

    elif intent == IntentType.NOTE_UPDATE:
        title_hint = result.entities.get("title", "").lower()
        notes, total = await note_service.list(db)
        if total == 0:
            return "You have no notes to update.", None
        target = next((n for n in notes if title_hint in n.title.lower()), notes[0])
        return f"Note '{target.title}' found. Please provide the updated content.", {
            "note": NoteRead.model_validate(target).model_dump(),
        }

    elif intent == IntentType.NOTE_DELETE:
        title_hint = result.entities.get("title", "").lower()
        notes, total = await note_service.list(db)
        if total == 0:
            return "You have no notes to delete.", None
        target = next((n for n in notes if title_hint in n.title.lower()), notes[0])
        await note_service.delete(db, target.id)
        return f"Note deleted: {target.title}", None

    elif intent == IntentType.REMINDER_CREATE:
        import datetime
        import re as _re
        text_lower = result.raw_text.lower()
        reminder_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        if "tomorrow" in text_lower:
            reminder_time = datetime.datetime.now() + datetime.timedelta(days=1)
            reminder_time = reminder_time.replace(hour=9, minute=0, second=0)
        elif "in " in text_lower:
            hour_match = _re.search(r"in\s+(\d+)\s+hour", text_lower)
            if hour_match:
                reminder_time = datetime.datetime.now() + datetime.timedelta(hours=int(hour_match.group(1)))
        title = result.raw_text.replace("remind me", "").replace("set a reminder", "").replace("to", "").strip()
        title = _re.sub(r"\s+(tomorrow|in\s+\d+\s+hours?)", "", title).strip()
        if not title:
            title = "Reminder"
        data = ReminderCreate(title=title, reminder_time=reminder_time)
        reminder = await reminder_service.create(db, data)
        time_str = reminder_time.strftime("%B %d at %I:%M %p")
        return f"Reminder set for {time_str}: {title}", {
            "reminder": ReminderRead.model_validate(reminder).model_dump(),
        }

    elif intent == IntentType.MEMORY_SAVE:
        fact = result.raw_text.replace("remember that", "").replace("remember", "").replace("save that", "").strip()
        if not fact:
            fact = result.raw_text
        entry = await memory_service.store(db, fact)
        if entry:
            return f"I'll remember that: {fact[:100]}", {"entry": {"id": entry.id, "fact": entry.fact}}
        return f"Okay.", None

    elif intent == IntentType.MEMORY_RECALL:
        query = result.raw_text.replace("what do you know about", "").replace("recall", "").replace("what do you know", "").strip()
        if query:
            entries, total = await memory_service.search(db, query)
            if total == 0:
                return f"I don't have any memories about {query}.", {"entries": [], "total": 0}
            facts = ". ".join(e.fact for e in entries[:3])
            return f"I know {total} things: {facts}", {
                "entries": [{"id": e.id, "fact": e.fact, "importance": e.importance} for e in entries],
                "total": total,
            }
        entries, total = await memory_service.list(db)
        if total == 0:
            return "I don't know anything about you yet.", {"entries": [], "total": 0}
        facts = ". ".join(e.fact for e in entries[:3])
        return f"I know {total} things about you. {facts}", {
            "entries": [{"id": e.id, "fact": e.fact} for e in entries],
            "total": total,
        }

    elif intent == IntentType.MEMORY_FORGET:
        query = result.raw_text.replace("forget that", "").replace("forget", "").strip()
        if query:
            count = await memory_service.forget_by_fact(db, query)
            return f"I forgot {count} memory about {query}." if count else f"I don't have any memories about {query}.", None
        return "What should I forget?", None

    elif intent == IntentType.GLOBAL_SEARCH:
        query = result.entities.get("query", result.raw_text.replace("search for", "").replace("find", "").replace("look for", "").replace("look up", "").strip())
        if not query:
            return "What would you like me to search for?", None
        results = await search_service.search(db, query)
        if not results:
            return f"No results found for '{query}'.", {"results": [], "total": 0}
        snippets = [f"{r['type']}: {r['title'][:60]}" for r in results[:5]]
        summary = f"Found {len(results)} result{'s' if len(results) != 1 else ''}: " + ", ".join(snippets)
        return summary, {"results": results, "total": len(results)}

    else:
        return f"Command understood but not yet implemented.", None
