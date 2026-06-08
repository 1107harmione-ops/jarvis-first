"""Voice command API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.logger import get_logger
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

    else:
        return f"Command understood but not yet implemented.", None
