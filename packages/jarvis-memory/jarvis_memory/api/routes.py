"""FastAPI route definitions for all REST endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from jarvis_memory.database import db
from jarvis_memory.models.conversation import ConversationDocument
from jarvis_memory.models.knowledge import KnowledgeDocument
from jarvis_memory.models.memory import MemoryDocument, MemoryType
from jarvis_memory.models.message import MessageDocument
from jarvis_memory.models.task import TaskDocument
from jarvis_memory.models.user import UserDocument
from jarvis_memory.repositories.conversation_repo import ConversationRepository
from jarvis_memory.repositories.knowledge_repo import KnowledgeRepository
from jarvis_memory.repositories.memory_repo import MemoryRepository
from jarvis_memory.repositories.message_repo import MessageRepository
from jarvis_memory.repositories.task_repo import TaskRepository
from jarvis_memory.repositories.user_repo import UserRepository
from jarvis_memory.services.context_builder import ContextBuilder
from jarvis_memory.services.consolidation_service import ConsolidationService
from jarvis_memory.services.embedding_service import EmbeddingService
from jarvis_memory.services.memory_service import MemoryService
from jarvis_memory.services.retrieval_service import RetrievalService
from jarvis_memory.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency helpers — lazy-init singletons
# ---------------------------------------------------------------------------

def _get_user_repo() -> UserRepository:
    return UserRepository(db.get_collection("users"))


def _get_conversation_repo() -> ConversationRepository:
    return ConversationRepository(db.get_collection("conversations"))


def _get_message_repo() -> MessageRepository:
    return MessageRepository(db.get_collection("messages"))


def _get_memory_repo() -> MemoryRepository:
    return MemoryRepository(db.get_collection("memories"))


def _get_task_repo() -> TaskRepository:
    return TaskRepository(db.get_collection("tasks"))


def _get_knowledge_repo() -> KnowledgeRepository:
    return KnowledgeRepository(db.get_collection("knowledge"))


def _get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def _get_scoring_service() -> ScoringService:
    return ScoringService(embedding_service=_get_embedding_service())


def _get_memory_service() -> MemoryService:
    return MemoryService(
        memory_repo=_get_memory_repo(),
        embedding_service=_get_embedding_service(),
        scoring_service=_get_scoring_service(),
    )


def _get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        memory_repo=_get_memory_repo(),
        knowledge_repo=_get_knowledge_repo(),
        embedding_service=_get_embedding_service(),
        scoring_service=_get_scoring_service(),
    )


def _get_context_builder() -> ContextBuilder:
    return ContextBuilder(
        retrieval_service=_get_retrieval_service(),
        embedding_service=_get_embedding_service(),
        user_repo=_get_user_repo(),
        conversation_repo=_get_conversation_repo(),
        message_repo=_get_message_repo(),
        task_repo=_get_task_repo(),
    )


def _get_consolidation_service() -> ConsolidationService:
    return ConsolidationService(
        memory_repo=_get_memory_repo(),
        embedding_service=_get_embedding_service(),
        scoring_service=_get_scoring_service(),
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password_hash: str | None = None
    profile: dict[str, Any] | None = None
    preferences: dict[str, Any] | None = None


class UserUpdateRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    profile: dict[str, Any] | None = None
    preferences: dict[str, Any] | None = None
    voice_settings: dict[str, Any] | None = None


class ConversationCreateRequest(BaseModel):
    user_id: str
    session_id: str = "default"
    title: str = ""


class MessageCreateRequest(BaseModel):
    user_id: str
    role: str
    content: str
    content_type: str = "text"
    tokens: int = 0


class MemoryCreateRequest(BaseModel):
    user_id: str
    memory_type: MemoryType = "short_term"
    content: str
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    source: str | None = None
    source_id: str | None = None


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    memory_type: MemoryType | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    importance_score: float | None = None


class MemorySearchRequest(BaseModel):
    user_id: str
    query: str
    memory_types: list[MemoryType] | None = None
    top_k: int = 10


class ContextRequest(BaseModel):
    user_id: str
    conversation_id: str
    query: str
    max_tokens: int = 3000


class TaskCreateRequest(BaseModel):
    user_id: str
    title: str
    description: str = ""
    status: str = "pending"
    priority: int = 0
    task_type: str = "one_off"
    due_at: str | None = None


class TaskUpdateRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: int | None = None
    due_at: str | None = None


class KnowledgeCreateRequest(BaseModel):
    user_id: str | None = None
    title: str
    content: str
    summary: str | None = None
    url: str | None = None
    source_type: str = "document"
    tags: list[str] | None = None


class KnowledgeSearchRequest(BaseModel):
    user_id: str | None = None
    query: str
    top_k: int = 5


# ---------------------------------------------------------------------------
# Routes — Users
# ---------------------------------------------------------------------------

@router.post("/users", summary="Create user")
async def create_user(body: UserCreateRequest):
    """Register a new user."""
    repo = _get_user_repo()
    existing = await repo.find_by_email(body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = UserDocument(
        username=body.username,
        email=body.email,
        password_hash=body.password_hash,
    )
    if body.profile:
        user.profile = type(user.profile)(**body.profile)
    if body.preferences:
        user.preferences = type(user.preferences)(**body.preferences)

    created = await repo.create(user)
    return created.model_dump(by_alias=True)


@router.get("/users/{user_id}", summary="Get user")
async def get_user(user_id: str):
    """Retrieve a user by external ID."""
    repo = _get_user_repo()
    user = await repo.get_by_field("user_id", user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user.model_dump(by_alias=True)


@router.put("/users/{user_id}", summary="Update user")
async def update_user(user_id: str, body: UserUpdateRequest):
    """Update user profile and/or preferences."""
    repo = _get_user_repo()
    updates: dict[str, Any] = {}
    if body.username is not None:
        updates["username"] = body.username
    if body.email is not None:
        updates["email"] = body.email
    if body.profile is not None:
        updates["profile"] = body.profile
    if body.preferences is not None:
        updates["preferences"] = body.preferences
    if body.voice_settings is not None:
        updates["voice_settings"] = body.voice_settings

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Find by user_id, update using _id
    existing = await repo.get_by_field("user_id", user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="User not found")
    updated = await repo.update(existing.id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated.model_dump(by_alias=True)


# ---------------------------------------------------------------------------
# Routes — Conversations
# ---------------------------------------------------------------------------

@router.post("/conversations", summary="Create conversation")
async def create_conversation(body: ConversationCreateRequest):
    """Start a new conversation."""
    repo = _get_conversation_repo()
    conv = ConversationDocument(
        user_id=body.user_id,
        session_id=body.session_id,
        title=body.title,
    )
    created = await repo.create(conv)
    return created.model_dump(by_alias=True)


@router.get("/conversations/{conv_id}", summary="Get conversation")
async def get_conversation(conv_id: str):
    """Retrieve a conversation and its recent messages."""
    repo = _get_conversation_repo()
    conv = await repo.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_repo = _get_message_repo()
    messages = await msg_repo.get_conversation_messages(conv.conversation_id, limit=50)

    result = conv.model_dump(by_alias=True)
    result["messages"] = [m.model_dump(by_alias=True) for m in messages]
    return result


@router.post("/conversations/{conv_id}/messages", summary="Add message")
async def add_message(conv_id: str, body: MessageCreateRequest):
    """Add a message to a conversation."""
    conv_repo = _get_conversation_repo()

    # Verify conversation exists (by _id)
    conv = await conv_repo.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_repo = _get_message_repo()
    msg = MessageDocument(
        conversation_id=conv.conversation_id,
        user_id=body.user_id,
        role=body.role,  # type: ignore
        content=body.content,
        content_type=body.content_type,
        tokens=body.tokens,
    )
    created = await msg_repo.create(msg)

    # Update conversation counters
    await conv_repo.add_message_count(conv_id, tokens=body.tokens)

    return created.model_dump(by_alias=True)


@router.get("/conversations/{conv_id}/messages", summary="Get messages")
async def get_conversation_messages(
    conv_id: str,
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    """Retrieve messages for a conversation."""
    conv_repo = _get_conversation_repo()
    conv = await conv_repo.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_repo = _get_message_repo()
    messages = await msg_repo.get_conversation_messages(
        conv.conversation_id, limit=limit, skip=skip
    )
    return [m.model_dump(by_alias=True) for m in messages]


# ---------------------------------------------------------------------------
# Routes — Memories
# ---------------------------------------------------------------------------

@router.post("/memories", summary="Create memory")
async def create_memory(body: MemoryCreateRequest):
    """Create a new memory with auto-generated embedding and importance score."""
    service = _get_memory_service()
    memory = await service.create_memory(
        user_id=body.user_id,
        memory_type=body.memory_type,
        content=body.content,
        tags=body.tags,
        metadata=body.metadata,
        source=body.source,
        source_id=body.source_id,
    )
    return memory.model_dump(by_alias=True)


@router.get("/memories/{memory_id}", summary="Get memory")
async def get_memory(memory_id: str):
    """Retrieve a memory by its ID."""
    service = _get_memory_service()
    memory = await service.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory.model_dump(by_alias=True)


@router.put("/memories/{memory_id}", summary="Update memory")
async def update_memory(memory_id: str, body: MemoryUpdateRequest):
    """Update a memory."""
    service = _get_memory_service()
    updates: dict[str, Any] = {}
    if body.content is not None:
        updates["content"] = body.content
    if body.memory_type is not None:
        updates["memory_type"] = body.memory_type
    if body.tags is not None:
        updates["tags"] = body.tags
    if body.metadata is not None:
        updates["metadata"] = body.metadata
    if body.importance_score is not None:
        updates["importance_score"] = body.importance_score

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = await service.update_memory(memory_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return updated.model_dump(by_alias=True)


@router.delete("/memories/{memory_id}", summary="Delete memory")
async def delete_memory(memory_id: str):
    """Delete a memory by its ID."""
    service = _get_memory_service()
    deleted = await service.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@router.post("/memories/search", summary="Hybrid search memories")
async def search_memories(body: MemorySearchRequest):
    """Hybrid vector + keyword search across user memories."""
    service = _get_memory_service()
    results = await service.search_memories(
        user_id=body.user_id,
        query=body.query,
        memory_types=body.memory_types,
        top_k=body.top_k,
    )
    return {"results": results, "total": len(results)}


# ---------------------------------------------------------------------------
# Routes — Context
# ---------------------------------------------------------------------------

@router.post("/context", summary="Build context payload")
async def build_context(body: ContextRequest):
    """Build a structured context payload for LLM injection."""
    builder = _get_context_builder()
    payload = await builder.build_context(
        user_id=body.user_id,
        conversation_id=body.conversation_id,
        query=body.query,
        max_tokens=body.max_tokens,
    )
    return payload.to_dict()


# ---------------------------------------------------------------------------
# Routes — Tasks
# ---------------------------------------------------------------------------

@router.get("/tasks", summary="List tasks")
async def list_tasks(
    user_id: str = Query(..., description="User ID"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
):
    """List tasks for a user, optionally filtered by status."""
    repo = _get_task_repo()
    tasks = await repo.get_user_tasks(
        user_id, status=status, limit=limit, skip=skip
    )
    return [t.model_dump(by_alias=True) for t in tasks]


@router.post("/tasks", summary="Create task")
async def create_task(body: TaskCreateRequest):
    """Create a new task."""
    repo = _get_task_repo()
    from datetime import datetime

    due_at = None
    if body.due_at:
        try:
            due_at = datetime.fromisoformat(body.due_at)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid due_at format")

    task = TaskDocument(
        user_id=body.user_id,
        title=body.title,
        description=body.description,
        status=body.status,  # type: ignore
        priority=body.priority,
        task_type=body.task_type,
        due_at=due_at,
    )
    created = await repo.create(task)
    return created.model_dump(by_alias=True)


@router.put("/tasks/{task_id}", summary="Update task")
async def update_task(task_id: str, body: TaskUpdateRequest):
    """Update a task."""
    repo = _get_task_repo()
    from datetime import datetime

    updates: dict[str, Any] = {}
    if body.title is not None:
        updates["title"] = body.title
    if body.description is not None:
        updates["description"] = body.description
    if body.status is not None:
        updates["status"] = body.status
    if body.priority is not None:
        updates["priority"] = body.priority
    if body.due_at is not None:
        try:
            updates["due_at"] = datetime.fromisoformat(body.due_at)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid due_at format")
    if body.status == "completed":
        updates["completed_at"] = datetime.utcnow()

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    existing = await repo.get(task_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Task not found")

    updated = await repo.update(task_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return updated.model_dump(by_alias=True)


# ---------------------------------------------------------------------------
# Routes — Knowledge
# ---------------------------------------------------------------------------

@router.post("/knowledge", summary="Add knowledge document")
async def add_knowledge(body: KnowledgeCreateRequest):
    """Add a knowledge document with auto-generated embedding."""
    repo = _get_knowledge_repo()
    embedder = _get_embedding_service()

    embedding = await embedder.embed(body.content)

    doc = KnowledgeDocument(
        user_id=body.user_id,
        title=body.title,
        content=body.content,
        summary=body.summary,
        url=body.url,
        source_type=body.source_type,  # type: ignore
        tags=body.tags or [],
        embedding=embedding,
    )
    created = await repo.create(doc)
    return created.model_dump(by_alias=True)


@router.get("/knowledge/search", summary="Search knowledge")
async def search_knowledge(
    user_id: str | None = Query(None, description="User ID (optional)"),
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50),
):
    """Search knowledge documents using hybrid embedding + keyword."""
    retrieval = _get_retrieval_service()
    results = await retrieval.search_knowledge(
        user_id=user_id, query=query, top_k=top_k
    )
    return {"results": results, "total": len(results)}


# ---------------------------------------------------------------------------
# Routes — Consolidation
# ---------------------------------------------------------------------------

@router.post("/consolidate", summary="Run memory consolidation")
async def run_consolidation(
    user_id: str = Query(..., description="User ID to consolidate for"),
):
    """Trigger STM → LTM consolidation for a user."""
    service = _get_consolidation_service()
    count = await service.consolidate(user_id)
    return {"user_id": user_id, "consolidated_count": count}
