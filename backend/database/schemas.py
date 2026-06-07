"""
MongoDB document schemas — plain dict builders for CRUD operations.
These bridge Pydantic models to Motor/MongoDB documents.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId


def now_utc() -> datetime:
    """Current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def serialize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert a MongoDB document for JSON serialization.
    Converts ObjectId and datetime fields to strings.
    """
    serialized: dict[str, Any] = {}
    for key, value in doc.items():
        if key == "_id":
            serialized["id"] = str(value)
        elif isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            serialized[key] = [
                serialize_doc(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict):
            serialized[key] = serialize_doc(value)
        else:
            serialized[key] = value
    return serialized


# ── User ─────────────────────────────────────────────────────────


def new_user_doc(
    email: str,
    password_hash: str,
    name: str,
    role: str = "user",
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "email": email,
        "password_hash": password_hash,
        "name": name,
        "role": role,
        "api_key_hashed": None,
        "is_active": True,
        "settings": {},
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


# ── Conversation ─────────────────────────────────────────────────


def new_conversation_doc(
    user_id: str,
    title: str = "New Conversation",
    model: str = "deepseek-chat",
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "title": title,
        "model": model,
        "message_count": 0,
        "is_archived": False,
        "metadata": {},
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


def update_conversation_doc(
    title: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {"updated_at": now_utc()}
    if title is not None:
        doc["title"] = title
    if model is not None:
        doc["model"] = model
    return doc


# ── Message ──────────────────────────────────────────────────────


def new_message_doc(
    conversation_id: str,
    role: str,
    content: str,
    agent: str | None = None,
    attachments: list[str] | None = None,
    tokens_used: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "agent": agent,
        "attachments": attachments or [],
        "tokens_used": tokens_used,
        "metadata": metadata or {},
        "created_at": now_utc(),
    }


# ── Memory ───────────────────────────────────────────────────────


def new_memory_doc(
    user_id: str,
    content: str,
    memory_type: str = "short_term",
    importance_score: float = 0.5,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    summary: str | None = None,
    source: str | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "content": content,
        "memory_type": memory_type,
        "importance_score": importance_score,
        "tags": tags or [],
        "embedding": embedding,
        "summary": summary,
        "source": source,
        "access_count": 0,
        "last_accessed": None,
        "consolidated": False,
        "decay_rate": 0.1,
        "expires_at": expires_at,
        "metadata": metadata or {},
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


# ── Task ─────────────────────────────────────────────────────────


def new_task_doc(
    user_id: str,
    title: str,
    description: str = "",
    priority: str = "medium",
    due_at: datetime | None = None,
    scheduled_at: datetime | None = None,
    recurring: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "title": title,
        "description": description,
        "status": "pending",
        "priority": priority,
        "due_at": due_at,
        "scheduled_at": scheduled_at,
        "recurring": recurring,
        "tags": tags or [],
        "retry_count": 0,
        "completed_at": None,
        "error": None,
        "metadata": {},
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


def update_task_doc(**kwargs: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {"updated_at": now_utc()}
    for key, value in kwargs.items():
        if value is not None:
            doc[key] = value
    return doc


# ── Agent Log ────────────────────────────────────────────────────


def new_agent_log_doc(
    agent_name: str,
    session_id: str,
    action: str,
    user_id: str | None = None,
    input_summary: str = "",
    output_summary: str = "",
    tokens_used: int = 0,
    duration_ms: float = 0.0,
    status: str = "success",
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "agent_name": agent_name,
        "session_id": session_id,
        "user_id": user_id,
        "action": action,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "tokens_used": tokens_used,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
        "metadata": metadata or {},
        "created_at": now_utc(),
    }


# ── Knowledge ────────────────────────────────────────────────────


def new_knowledge_doc(
    title: str,
    content: str,
    source: str,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "title": title,
        "content": content,
        "source": source,
        "tags": tags or [],
        "embedding": embedding,
        "url": url,
        "access_count": 0,
        "metadata": metadata or {},
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


# ── Research ─────────────────────────────────────────────


def new_research_report_doc(
    user_id: str,
    session_id: str,
    query: str,
    research_type: str,
    depth: str,
    executive_summary: str = "",
    key_findings: list[str] | None = None,
    detailed_analysis: str | None = None,
    pros: list[str] | None = None,
    cons: list[str] | None = None,
    recommendations: list[str] | None = None,
    conclusions: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    fact_check: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "session_id": session_id,
        "title": "",
        "query": query,
        "research_type": research_type,
        "depth": depth,
        "executive_summary": executive_summary,
        "key_findings": key_findings or [],
        "detailed_analysis": detailed_analysis,
        "pros": pros,
        "cons": cons,
        "recommendations": recommendations,
        "conclusions": conclusions,
        "sources": sources or [],
        "source_count": len(sources) if sources else 0,
        "fact_check": fact_check,
        "metadata": metadata or {},
        "tags": tags or [],
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }


def new_research_source_doc(
    url: str,
    title: str,
    snippet: str,
    domain: str,
    content: str = "",
    published_date: datetime | None = None,
    author: str | None = None,
    authority_score: float = 0.0,
    freshness_score: float = 0.0,
    accuracy_score: float = 0.0,
    relevance_score: float = 0.0,
    popularity_score: float = 0.0,
    overall_score: float = 0.0,
    query: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "url": url,
        "title": title,
        "snippet": snippet,
        "content": content,
        "domain": domain,
        "published_date": published_date,
        "author": author,
        "authority_score": authority_score,
        "freshness_score": freshness_score,
        "accuracy_score": accuracy_score,
        "relevance_score": relevance_score,
        "popularity_score": popularity_score,
        "overall_score": overall_score,
        "query": query,
        "tags": tags or [],
        "access_count": 0,
        "created_at": now_utc(),
    }


def new_research_cache_doc(
    cache_key: str,
    query: str,
    research_type: str,
    ttl: datetime,
    report_id: str | None = None,
    synthesis: str | None = None,
    source_urls: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "_id": ObjectId(),
        "cache_key": cache_key,
        "query": query,
        "research_type": research_type,
        "report_id": report_id,
        "synthesis": synthesis,
        "source_urls": source_urls or [],
        "ttl": ttl,
        "created_at": now_utc(),
    }
