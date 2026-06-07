"""LLM context builder — assembles a structured context payload."""

import logging
from dataclasses import dataclass, field
from typing import Any

from jarvis_memory.repositories.conversation_repo import ConversationRepository
from jarvis_memory.repositories.message_repo import MessageRepository
from jarvis_memory.repositories.task_repo import TaskRepository
from jarvis_memory.repositories.user_repo import UserRepository
from jarvis_memory.services.retrieval_service import RetrievalService, ScoredMemory
from jarvis_memory.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ContextPayload:
    """Structured context payload for LLM prompt injection."""

    user_context: dict[str, Any] = field(default_factory=dict)
    conversation_context: dict[str, Any] = field(default_factory=dict)
    memory_context: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    task_context: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    knowledge_context: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the payload as a plain dict."""
        return {
            "user_context": self.user_context,
            "conversation_context": self.conversation_context,
            "memory_context": self.memory_context,
            "task_context": self.task_context,
            "knowledge_context": self.knowledge_context,
        }


class ContextBuilder:
    """Assembles structured context for LLM injection.

    Gathers user profile, conversation history, relevant memories, active
    tasks, and knowledge documents — then trims the result to a token
    budget.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        embedding_service: EmbeddingService,
        user_repo: UserRepository,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        task_repo: TaskRepository,
    ) -> None:
        self._retrieval = retrieval_service
        self._embedder = embedding_service
        self._user_repo = user_repo
        self._conversation_repo = conversation_repo
        self._message_repo = message_repo
        self._task_repo = task_repo

    async def build_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        max_tokens: int = 3000,
    ) -> ContextPayload:
        """Assemble full context payload for LLM injection.

        Args:
            user_id: The user's external ID.
            conversation_id: The conversation ID.
            query: The current user query text.
            query_embedding: Pre-computed query embedding.
            max_tokens: Maximum token budget for the payload.

        Returns:
            A ``ContextPayload`` instance.
        """
        # 1. User profile + preferences (always included)
        user = await self._get_user_context(user_id)

        # 2. Conversation history
        conversation = await self._get_conversation_context(conversation_id)

        # 3. Relevant memories via hybrid search
        if query_embedding is None:
            query_embedding = await self._embedder.embed(query)

        memories = await self._retrieval.hybrid_search(
            user_id, query, query_embedding, top_k=15
        )
        memory_groups = self._group_memories_by_type(memories)

        # 4. Active tasks
        tasks = await self._get_task_context(user_id)

        # 5. Knowledge documents
        knowledge = await self._retrieval.search_knowledge(
            user_id, query, query_embedding, top_k=5
        )

        # 6. Assemble payload
        payload = ContextPayload(
            user_context=user,
            conversation_context=conversation,
            memory_context=memory_groups,
            task_context=tasks,
            knowledge_context=knowledge,
        )

        # 7. Trim to token budget
        return await self._trim_to_budget(payload, max_tokens)

    async def _get_user_context(self, user_id: str) -> dict[str, Any]:
        """Fetch user profile and preferences.

        Args:
            user_id: The user's external ID.

        Returns:
            A dict with user context fields.
        """
        user = await self._user_repo.get_by_field("user_id", user_id)
        if user is None:
            return {"user_id": user_id, "username": "unknown"}

        return {
            "user_id": user.user_id,
            "username": user.username,
            "profile": {
                "name": user.profile.name,
                "timezone": user.profile.timezone,
                "location": user.profile.location,
            },
            "preferences": {
                "language": user.preferences.language,
                "response_style": user.preferences.response_style,
                "voice_speed": user.preferences.voice_speed,
                "theme": user.preferences.theme,
            },
        }

    async def _get_conversation_context(
        self,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Fetch conversation metadata and recent messages.

        Args:
            conversation_id: The conversation ID.

        Returns:
            A dict with conversation context fields.
        """
        conversation = await self._conversation_repo.get_by_field(
            "conversation_id", conversation_id
        )

        messages = await self._message_repo.get_conversation_messages(
            conversation_id, limit=20
        )

        recent_messages = [
            {
                "role": msg.role,
                "content": msg.content[:500],  # Truncate long messages
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            for msg in messages[-10:]
        ]

        return {
            "current_conversation_id": conversation_id,
            "title": conversation.title if conversation else "",
            "summary": conversation.summary if conversation else "",
            "message_count": conversation.message_count if conversation else 0,
            "recent_messages": recent_messages,
        }

    async def _get_task_context(
        self,
        user_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch pending and recently completed tasks.

        Args:
            user_id: The user's external ID.

        Returns:
            A dict with ``pending`` and ``completed_recently`` lists.
        """
        pending = await self._task_repo.get_pending_tasks_by_priority(
            user_id, limit=10
        )
        completed = await self._task_repo.get_completed_tasks(
            user_id, limit=5
        )

        return {
            "pending": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "priority": t.priority,
                    "due_at": t.due_at.isoformat() if t.due_at else None,
                    "status": t.status,
                }
                for t in pending
            ],
            "completed_recently": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "completed_at": t.completed_at.isoformat()
                    if t.completed_at
                    else None,
                }
                for t in completed
            ],
        }

    def _group_memories_by_type(
        self,
        memories: list[ScoredMemory],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group scored memories by their ``memory_type``.

        Args:
            memories: List of scored memories.

        Returns:
            A dict mapping memory type keys to lists of memory dicts.
        """
        groups: dict[str, list[dict[str, Any]]] = {}

        for mem in memories:
            mtype = mem.memory_type or "short_term"
            entry = {
                "content": mem.content,
                "importance": round(mem.importance_score, 3),
                "score": round(mem.score, 3),
                "tags": mem.tags,
            }
            groups.setdefault(mtype, []).append(entry)

        return groups

    async def _trim_to_budget(
        self,
        payload: ContextPayload,
        max_tokens: int,
    ) -> ContextPayload:
        """Trim the context payload to fit within *max_tokens*.

        Uses a rough token estimate (4 chars ≈ 1 token). Trims in order:
        1. Knowledge documents (least relevant)
        2. Task context (older completed tasks)
        3. Memory context (lowest scored entries)
        4. Conversation history (older messages)

        Args:
            payload: The full context payload.
            max_tokens: Maximum allowed tokens.

        Returns:
            A trimmed ``ContextPayload``.
        """
        current_tokens = self._estimate_tokens(payload)
        if current_tokens <= max_tokens:
            return payload

        logger.info(
            "Context payload %d tokens exceeds budget %d; trimming",
            current_tokens,
            max_tokens,
        )

        # 1. Trim knowledge (keep top 2)
        while payload.knowledge_context and self._estimate_tokens(payload) > max_tokens:
            payload.knowledge_context.pop()

        # 2. Trim completed tasks
        while payload.task_context.get("completed_recently") and self._estimate_tokens(payload) > max_tokens:
            payload.task_context["completed_recently"].pop()

        # 3. Trim memory context (remove lowest-scored entries)
        for mtype in list(payload.memory_context.keys()):
            group = payload.memory_context[mtype]
            group.sort(key=lambda x: x.get("score", 0), reverse=True)
            while group and self._estimate_tokens(payload) > max_tokens:
                group.pop()

        # 4. Trim conversation history (remove oldest)
        messages = payload.conversation_context.get("recent_messages", [])
        while messages and self._estimate_tokens(payload) > max_tokens:
            messages.pop(0)

        return payload

    def _estimate_tokens(self, payload: ContextPayload) -> int:
        """Estimate token count from serialized payload length.

        Approximates 1 token ≈ 4 characters.

        Args:
            payload: The context payload.

        Returns:
            Estimated token count.
        """
        import json

        text = json.dumps(payload.to_dict(), default=str)
        return len(text) // 4
