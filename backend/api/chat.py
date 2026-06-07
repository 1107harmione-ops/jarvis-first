"""
Chat API — conversation management and message exchange endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.agents.router_agent import router_agent
from backend.database.mongodb import mongodb
from backend.database.schemas import (
    new_conversation_doc,
    new_message_doc,
    serialize_doc,
)
from backend.database.models import (
    ChatRequest,
    ConversationCreate,
    ConversationResponse,
    MessageResponse,
)
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/conversations")
async def create_conversation(
    body: ConversationCreate,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new conversation."""
    doc = new_conversation_doc(
        user_id=user["id"],
        title=body.title,
        model=body.model,
    )
    result = await mongodb.conversations.insert_one(doc)
    conv = serialize_doc(doc)
    return {"success": True, "data": conv}


@router.get("/conversations")
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """List user's conversations (paginated, most recent first)."""
    skip = (page - 1) * page_size
    cursor = mongodb.conversations.find(
        {"user_id": user["id"], "is_archived": {"$ne": True}},
        sort=[("updated_at", -1)],
        skip=skip,
        limit=page_size,
    )
    docs = await cursor.to_list(length=page_size)
    total = await mongodb.conversations.count_documents({"user_id": user["id"]})

    return {
        "success": True,
        "data": {
            "items": [serialize_doc(d) for d in docs],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        },
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get a conversation with its messages."""
    conv = await mongodb.conversations.find_one(
        {"_id": conversation_id, "user_id": user["id"]}
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages_cursor = mongodb.messages.find(
        {"conversation_id": conversation_id},
        sort=[("created_at", 1)],
    )
    messages = await messages_cursor.to_list(length=500)

    return {
        "success": True,
        "data": {
            "conversation": serialize_doc(conv),
            "messages": [serialize_doc(m) for m in messages],
        },
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Delete a conversation and its messages."""
    result = await mongodb.conversations.delete_one(
        {"_id": conversation_id, "user_id": user["id"]}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await mongodb.messages.delete_many({"conversation_id": conversation_id})
    return {"success": True, "message": "Conversation deleted"}


@router.post("/send")
async def send_message(
    body: ChatRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Send a message and get an AI response.

    Routes through the multi-agent system automatically.
    """
    response = await router_agent.process(
        user_id=user["id"],
        message=body.message,
        conversation_id=body.conversation_id,
        stream=False,
        attachments=body.attachments,
        metadata=body.metadata,
    )

    return {
        "success": True,
        "data": {
            "content": response.get("content", ""),
            "agent": response.get("agent", "router"),
            "conversation_id": response.get("conversation_id"),
            "category": response.get("category", "general"),
            "duration_ms": response.get("duration_ms", 0),
            "tokens_used": response.get("tokens_used"),
        },
    }


@router.get("/messages/{conversation_id}")
async def get_messages(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=500),
    before_id: str | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get messages in a conversation (paginated, newest first)."""
    query: dict[str, Any] = {"conversation_id": conversation_id}
    if before_id:
        query["_id"] = {"$lt": before_id}

    cursor = mongodb.messages.find(
        query, sort=[("created_at", -1)], limit=limit
    )
    messages = await cursor.to_list(length=limit)
    messages.reverse()  # Return chronological order

    return {
        "success": True,
        "data": {
            "messages": [serialize_doc(m) for m in messages],
            "has_more": len(messages) == limit,
        },
    }
