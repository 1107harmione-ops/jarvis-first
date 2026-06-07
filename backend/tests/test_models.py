"""
Tests for database models and schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.database.models import (
    AgentType,
    ChatRequest,
    MemoryCreate,
    MemoryType,
    TaskCreate,
    TaskPriority,
    TaskStatus,
    UserCreate,
    UserLogin,
    VoiceSessionCreate,
)
from backend.database.schemas import (
    new_conversation_doc,
    new_memory_doc,
    new_message_doc,
    new_task_doc,
    new_user_doc,
    serialize_doc,
)


class TestModels:
    """Tests for Pydantic models."""

    def test_user_create_valid(self) -> None:
        user = UserCreate(email="test@example.com", password="StrongPass1", name="Test")
        assert user.email == "test@example.com"

    def test_user_create_invalid_email(self) -> None:
        with pytest.raises(ValueError):
            UserCreate(email="invalid", password="StrongPass1", name="Test")

    def test_user_create_weak_password(self) -> None:
        with pytest.raises(ValueError):
            UserCreate(email="test@example.com", password="weak", name="Test")

    def test_user_login(self) -> None:
        login = UserLogin(email="test@example.com", password="pass")
        assert login.email == "test@example.com"

    def test_chat_request_valid(self) -> None:
        req = ChatRequest(message="Hello", agent=AgentType.ROUTER)
        assert req.message == "Hello"
        assert req.agent == AgentType.ROUTER
        assert req.stream is False

    def test_chat_request_with_attachments(self) -> None:
        req = ChatRequest(
            message="Analyze this",
            attachments=["http://example.com/image.jpg"],
        )
        assert len(req.attachments) == 1

    def test_memory_create(self) -> None:
        mem = MemoryCreate(
            content="Test memory",
            memory_type=MemoryType.LONG_TERM,
            importance_score=0.8,
            tags=["test"],
        )
        assert mem.memory_type == MemoryType.LONG_TERM

    def test_task_create(self) -> None:
        task = TaskCreate(
            title="Test task",
            priority=TaskPriority.HIGH,
            tags=["urgent"],
        )
        assert task.title == "Test task"
        assert task.priority == TaskPriority.HIGH

    def test_task_create_recurring(self) -> None:
        task = TaskCreate(
            title="Daily standup",
            recurring="0 9 * * 1-5",
        )
        assert task.recurring == "0 9 * * 1-5"

    def test_voice_session_create(self) -> None:
        session = VoiceSessionCreate(language="hi", wake_word_enabled=True)
        assert session.language == "hi"
        assert session.wake_word_enabled is True


class TestSchemas:
    """Tests for database document builders."""

    def test_new_user_doc(self) -> None:
        doc = new_user_doc(
            email="test@test.com",
            password_hash="hashed_pw",
            name="Tester",
        )
        assert doc["email"] == "test@test.com"
        assert doc["role"] == "user"
        assert doc["is_active"] is True
        assert "_id" in doc

    def test_new_conversation_doc(self) -> None:
        doc = new_conversation_doc(
            user_id="user123",
            title="Test Conversation",
            model="deepseek-chat",
        )
        assert doc["user_id"] == "user123"
        assert doc["title"] == "Test Conversation"
        assert doc["message_count"] == 0

    def test_new_message_doc(self) -> None:
        doc = new_message_doc(
            conversation_id="conv123",
            role="user",
            content="Hello!",
        )
        assert doc["conversation_id"] == "conv123"
        assert doc["role"] == "user"
        assert doc["content"] == "Hello!"

    def test_new_message_with_agent(self) -> None:
        doc = new_message_doc(
            conversation_id="conv123",
            role="assistant",
            content="Hi there!",
            agent="coding_agent",
            tokens_used=150,
        )
        assert doc["agent"] == "coding_agent"
        assert doc["tokens_used"] == 150

    def test_new_memory_doc_stm(self) -> None:
        doc = new_memory_doc(
            user_id="user123",
            content="Short term memory",
            memory_type="short_term",
            expires_at=datetime.now(timezone.utc),
        )
        assert doc["memory_type"] == "short_term"
        assert doc["expires_at"] is not None

    def test_new_memory_doc_ltm(self) -> None:
        doc = new_memory_doc(
            user_id="user123",
            content="Long term memory",
            memory_type="long_term",
            importance_score=0.8,
        )
        assert doc["memory_type"] == "long_term"
        assert doc["importance_score"] == 0.8
        assert doc["expires_at"] is None

    def test_new_task_doc(self) -> None:
        doc = new_task_doc(
            user_id="user123",
            title="Buy groceries",
            priority="high",
        )
        assert doc["title"] == "Buy groceries"
        assert doc["status"] == "pending"
        assert doc["priority"] == "high"

    def test_new_task_recurring(self) -> None:
        doc = new_task_doc(
            user_id="user123",
            title="Daily standup",
            recurring="0 9 * * 1-5",
        )
        assert doc["recurring"] == "0 9 * * 1-5"

    def test_serialize_doc_with_objectid(self) -> None:
        doc = new_user_doc(email="a@b.com", password_hash="pw", name="A")
        serialized = serialize_doc(doc)
        assert "id" in serialized
        assert serialized["id"] == str(doc["_id"])
        assert "_id" not in serialized

    def test_serialize_doc_with_datetime(self) -> None:
        doc = new_conversation_doc(user_id="u1")
        serialized = serialize_doc(doc)
        assert isinstance(serialized["created_at"], str)
