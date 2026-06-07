"""pytest fixtures for jarvis-memory tests.

Uses ``mongomock`` under an async adapter that mimics Motor's interface.
"""

import asyncio
from typing import Any

import pytest
import pytest_asyncio

from jarvis_memory.database import db
from jarvis_memory.models.conversation import ConversationDocument
from jarvis_memory.models.knowledge import KnowledgeDocument
from jarvis_memory.models.memory import MemoryDocument
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


# ---------------------------------------------------------------------------
# Async adapter — wraps mongomock's pymongo-API client as a Motor-like object
# ---------------------------------------------------------------------------

class _AsyncCollection:
    """Async adapter wrapping a mongomock ``Collection`` to behave like
    ``AsyncIOMotorCollection``."""

    def __init__(self, collection: Any):
        self._col = collection

    async def insert_one(self, document: dict, *args, **kwargs) -> Any:
        return await asyncio.to_thread(self._col.insert_one, document, *args, **kwargs)

    async def find_one(self, filter: dict, *args, **kwargs) -> Any:
        return await asyncio.to_thread(self._col.find_one, filter, *args, **kwargs)

    async def find_one_and_update(self, filter: dict, update: dict, *args, **kwargs) -> Any:
        return await asyncio.to_thread(
            self._col.find_one_and_update, filter, update, *args, **kwargs
        )

    async def delete_one(self, filter: dict, *args, **kwargs) -> Any:
        return await asyncio.to_thread(self._col.delete_one, filter, *args, **kwargs)

    def find(self, *args, **kwargs):
        """Return a synchronous cursor that we wrap with async next / to_list."""
        cursor = self._col.find(*args, **kwargs)
        return _AsyncCursor(cursor)

    def aggregate(self, pipeline: list[dict], *args, **kwargs):
        cursor = self._col.aggregate(pipeline, *args, **kwargs)
        return _AsyncCursor(cursor)

    async def count_documents(self, filter: dict, *args, **kwargs) -> int:
        return await asyncio.to_thread(self._col.count_documents, filter, *args, **kwargs)

    async def create_index(self, keys, *args, **kwargs) -> str:
        return await asyncio.to_thread(self._col.create_index, keys, *args, **kwargs)

    async def list_indexes(self, *args, **kwargs) -> list:
        return await asyncio.to_thread(lambda: list(self._col.list_indexes()), *args, **kwargs)

    async def drop(self, *args, **kwargs) -> None:
        return await asyncio.to_thread(self._col.drop, *args, **kwargs)


class _AsyncCursor:
    """Async wrapper around a pymongo Cursor."""

    def __init__(self, cursor: Any):
        self._cursor = cursor

    async def to_list(self, length: int | None = None) -> list:
        return await asyncio.to_thread(lambda: list(self._cursor))

    def sort(self, key_or_list, direction=None):
        self._cursor = self._cursor.sort(key_or_list, direction)
        return self

    def skip(self, n: int):
        self._cursor = self._cursor.skip(n)
        return self

    def limit(self, n: int):
        self._cursor = self._cursor.limit(n)
        return self


class _AsyncDb:
    """Async adapter wrapping a mongomock ``Database``."""

    def __init__(self, mongo_client: Any, name: str):
        self._db = mongo_client[name]

    def get_collection(self, name: str) -> _AsyncCollection:
        return _AsyncCollection(self._db[name])

    def __getitem__(self, name: str) -> _AsyncCollection:
        return self.get_collection(name)

    async def list_collection_names(self) -> list[str]:
        return await asyncio.to_thread(self._db.list_collection_names)


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Connect to an in-memory mongomock database and clean between tests."""
    import mongomock

    client = mongomock.MongoClient()
    async_db = _AsyncDb(client, "jarvis_test")

    db.client = client  # Keep sync client for reference
    db.db = async_db   # Our async wrapper

    yield

    # Drop all collections between tests for isolation
    if db.db is not None:
        collections = await db.db.list_collection_names()
        for coll in collections:
            col = db.db.get_collection(coll)
            await col.drop()

    db.client = None
    db.db = None


# ---------------------------------------------------------------------------
# Collection fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def memory_collection() -> _AsyncCollection:
    return db.get_collection("memories")


@pytest_asyncio.fixture
async def user_collection() -> _AsyncCollection:
    return db.get_collection("users")


@pytest_asyncio.fixture
async def conversation_collection() -> _AsyncCollection:
    return db.get_collection("conversations")


@pytest_asyncio.fixture
async def message_collection() -> _AsyncCollection:
    return db.get_collection("messages")


@pytest_asyncio.fixture
async def task_collection() -> _AsyncCollection:
    return db.get_collection("tasks")


@pytest_asyncio.fixture
async def knowledge_collection() -> _AsyncCollection:
    return db.get_collection("knowledge")


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def user_repo(user_collection) -> UserRepository:
    return UserRepository(user_collection)


@pytest_asyncio.fixture
async def conversation_repo(conversation_collection) -> ConversationRepository:
    return ConversationRepository(conversation_collection)


@pytest_asyncio.fixture
async def message_repo(message_collection) -> MessageRepository:
    return MessageRepository(message_collection)


@pytest_asyncio.fixture
async def memory_repo(memory_collection) -> MemoryRepository:
    return MemoryRepository(memory_collection)


@pytest_asyncio.fixture
async def task_repo(task_collection) -> TaskRepository:
    return TaskRepository(task_collection)


@pytest_asyncio.fixture
async def knowledge_repo(knowledge_collection) -> KnowledgeRepository:
    return KnowledgeRepository(knowledge_collection)


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

@pytest.fixture
def embedding_service() -> EmbeddingService:
    return EmbeddingService()


@pytest.fixture
def scoring_service(embedding_service) -> ScoringService:
    return ScoringService(embedding_service=embedding_service)


@pytest_asyncio.fixture
async def memory_service(
    memory_repo,
    embedding_service,
    scoring_service,
) -> MemoryService:
    return MemoryService(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        scoring_service=scoring_service,
    )


@pytest_asyncio.fixture
async def retrieval_service(
    memory_repo,
    knowledge_repo,
    embedding_service,
    scoring_service,
) -> RetrievalService:
    return RetrievalService(
        memory_repo=memory_repo,
        knowledge_repo=knowledge_repo,
        embedding_service=embedding_service,
        scoring_service=scoring_service,
    )


@pytest_asyncio.fixture
async def context_builder(
    retrieval_service,
    embedding_service,
    user_repo,
    conversation_repo,
    message_repo,
    task_repo,
) -> ContextBuilder:
    return ContextBuilder(
        retrieval_service=retrieval_service,
        embedding_service=embedding_service,
        user_repo=user_repo,
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        task_repo=task_repo,
    )


@pytest_asyncio.fixture
async def consolidation_service(
    memory_repo,
    embedding_service,
    scoring_service,
) -> ConsolidationService:
    return ConsolidationService(
        memory_repo=memory_repo,
        embedding_service=embedding_service,
        scoring_service=scoring_service,
        min_importance=0.4,
        age_hours=0,  # Make consolidation testable immediately
    )


# ---------------------------------------------------------------------------
# Seed data helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def seed_user(user_repo: UserRepository) -> UserDocument:
    """Create and return a test user."""
    user = UserDocument(
        user_id="test-user-001",
        username="tony",
        email="tony@stark.com",
    )
    return await user_repo.create(user)


@pytest_asyncio.fixture
async def seed_conversation(
    conversation_repo: ConversationRepository,
    seed_user: UserDocument,
) -> ConversationDocument:
    """Create and return a test conversation."""
    conv = ConversationDocument(
        user_id=seed_user.user_id,
        title="Test Conversation",
    )
    return await conversation_repo.create(conv)


@pytest_asyncio.fixture
async def seed_memories(
    memory_repo: MemoryRepository,
    seed_user: UserDocument,
) -> list[MemoryDocument]:
    """Create multiple test memories of different types."""
    memories = [
        MemoryDocument(
            user_id=seed_user.user_id,
            memory_type="short_term",
            content="User asked about today's weather forecast",
            importance_score=0.7,
            tags=["weather", "recent"],
        ),
        MemoryDocument(
            user_id=seed_user.user_id,
            memory_type="long_term",
            content="User prefers morning meetings before noon",
            importance_score=0.85,
            tags=["preference", "schedule"],
        ),
        MemoryDocument(
            user_id=seed_user.user_id,
            memory_type="user_preference",
            content="User prefers dark mode and concise responses",
            importance_score=0.9,
            tags=["preference"],
        ),
        MemoryDocument(
            user_id=seed_user.user_id,
            memory_type="episodic",
            content="User had a great meeting with Stark Industries on June 1",
            importance_score=0.75,
            tags=["meeting", "important"],
        ),
    ]
    created = []
    for mem in memories:
        c = await memory_repo.create(mem)
        created.append(c)
    return created


@pytest_asyncio.fixture
async def seed_tasks(
    task_repo: TaskRepository,
    seed_user: UserDocument,
) -> list[TaskDocument]:
    """Create test tasks."""
    tasks = [
        TaskDocument(
            user_id=seed_user.user_id,
            title="Review Q3 report",
            status="pending",
            priority=4,
        ),
        TaskDocument(
            user_id=seed_user.user_id,
            title="Email CFO about budget",
            status="completed",
            priority=3,
        ),
        TaskDocument(
            user_id=seed_user.user_id,
            title="Buy groceries",
            status="pending",
            priority=1,
        ),
    ]
    created = []
    for t in tasks:
        c = await task_repo.create(t)
        created.append(c)
    return created
