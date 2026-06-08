# Phase 1: Voice Task Manager MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working voice task manager with FastAPI + SQLite + Vosk STT + Edge TTS

**Architecture:** Clean layered architecture — API layer → Service layer → Database layer. Voice input flows through STT → Intent Router → Task Service → TTS response. Single-user, no auth.

**Tech Stack:** FastAPI, SQLAlchemy + SQLite, Pydantic, Vosk, Edge TTS, structlog

---

## File Structure

### New Files to Create

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app with lifespan
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings from .env
│   │   ├── logger.py              # Structured logging
│   │   ├── exceptions.py          # Custom exception classes
│   │   └── deps.py                # FastAPI dependency injection
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py          # SQLite engine + session factory
│   │   ├── models.py              # Task SQLAlchemy model
│   │   └── migrations.py          # create_all tables
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── models.py              # Task ORM model (re-export from database)
│   │   ├── schemas.py             # TaskCreate, TaskRead, TaskUpdate
│   │   ├── service.py             # CRUD business logic
│   │   └── api.py                 # REST endpoints
│   │
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── router.py              # Intent router (keyword → regex → fuzzy)
│   │   ├── stt.py                 # Vosk STT wrapper
│   │   └── tts.py                 # Edge TTS wrapper
│   │
│   └── api/
│       ├── __init__.py
│       └── voice.py               # Voice command endpoint (text-in/text-out for now)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test fixtures (test DB, test client)
│   ├── test_tasks.py              # Task CRUD tests
│   ├── test_router.py             # Intent router tests
│   └── test_voice.py              # Voice pipeline integration test
│
├── .env                           # Local config (create from .env.example)
├── .env.example                   # Template
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── start.sh
```

### Modified Files

```
backend/__init__.py                # (already exists, will be overwritten)
backend/main.py                    # (already exists, will be replaced)
```

---

### Task 1: Project Scaffolding — Config, Logging, Exceptions, Database

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/logger.py`
- Create: `backend/app/core/exceptions.py`
- Create: `backend/app/core/deps.py`
- Create: `backend/app/database/__init__.py`
- Create: `backend/app/database/connection.py`
- Create: `backend/app/database/models.py`
- Create: `backend/app/database/migrations.py`
- Create: `backend/.env.example`
- Create: `backend/requirements.txt`
- Modify: `backend/__init__.py` (make it empty)

- [ ] **Step 1: Create requirements.txt**

```txt
# Jarvis Voice Productivity Assistant — Phase 1
# Python 3.12+

# Web Framework
fastapi>=0.111.0,<1.0.0
uvicorn[standard]>=0.29.0,<1.0.0
pydantic>=2.7.0,<3.0.0
pydantic-settings>=2.3.0,<3.0.0

# Database
SQLAlchemy>=2.0.0,<3.0.0
aiosqlite>=0.20.0,<1.0.0

# Voice STT
vosk>=0.3.45

# Voice TTS
edge-tts>=6.1.0,<7.0.0

# Structured Logging
structlog>=24.1.0,<25.0.0

# Fuzzy Matching
thefuzz>=0.22.0,<1.0.0
python-Levenshtein>=0.25.0,<1.0.0

# Testing
pytest>=8.0.0,<9.0.0
pytest-asyncio>=0.23.0,<1.0.0
httpx>=0.27.0,<1.0.0
```

- [ ] **Step 2: Create app/__init__.py** (empty)

- [ ] **Step 3: Create core/__init__.py** (empty)

- [ ] **Step 4: Create core/config.py**

```python
"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    APP_NAME: str = "Jarvis"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./jarvis.db"

    # Vosk STT
    VOSK_MODEL_PATH: str = "./models/vosk-model-small-en-us-0.15"
    VOSK_SAMPLE_RATE: int = 16000

    # TTS
    TTS_VOICE: str = "en-US-AriaNeural"  # Edge TTS voice

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "console"

    # Paths
    DATA_DIR: Path = Path("./data")


settings = Settings()
```

- [ ] **Step 5: Create core/logger.py**

```python
"""Structured logging configuration."""

from __future__ import annotations

import logging
import structlog
from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if settings.LOG_FORMAT == "console"
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name or __name__)
```

- [ ] **Step 6: Create core/exceptions.py**

```python
"""Application exception hierarchy."""

from __future__ import annotations


class JarvisError(Exception):
    """Base exception for all Jarvis errors."""
    pass


class NotFoundError(JarvisError):
    """Resource not found."""
    def __init__(self, resource: str, resource_id: int | str):
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} with id {resource_id} not found")


class ValidationError(JarvisError):
    """Input validation failed."""
    pass


class VoiceProcessingError(JarvisError):
    """Voice processing (STT/TTS) failed."""
    pass


class IntentNotFoundError(JarvisError):
    """Could not determine intent from voice input."""
    def __init__(self, text: str):
        self.text = text
        super().__init__(f"Could not determine intent from: {text}")
```

- [ ] **Step 7: Create core/deps.py**

```python
"""FastAPI dependency injection."""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
```

- [ ] **Step 8: Create database/__init__.py** (empty)

- [ ] **Step 9: Create database/connection.py**

```python
"""Async SQLite database connection management."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass
```

- [ ] **Step 10: Create database/models.py**

```python
"""Database ORM models."""

from __future__ import annotations

import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, func

from app.database.connection import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, default="", server_default="")
    status = Column(String(20), default="pending", server_default="pending")
    priority = Column(String(20), default="medium", server_default="medium")
    due_date = Column(DateTime, nullable=True)
    tags = Column(String(500), default="", server_default="")
    category = Column(String(100), default="general", server_default="general")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title='{self.title}', status='{self.status}')>"
```

- [ ] **Step 11: Create database/migrations.py**

```python
"""Database migration / schema creation."""

from __future__ import annotations

from app.core.logger import get_logger
from app.database.connection import Base, engine

logger = get_logger(__name__)


async def create_tables() -> None:
    """Create all tables defined in ORM models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
```

- [ ] **Step 12: Create .env.example**

```env
# Jarvis Voice Assistant — Environment Configuration
APP_NAME=Jarvis
APP_VERSION=0.1.0
DEBUG=true

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite+aiosqlite:///./jarvis.db

# Vosk STT
VOSK_MODEL_PATH=./models/vosk-model-small-en-us-0.15
VOSK_SAMPLE_RATE=16000

# TTS
TTS_VOICE=en-US-AriaNeural

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

- [ ] **Step 13: Install dependencies and verify**

```bash
cd /root/backend && pip install -r requirements.txt
```

- [ ] **Step 14: Initialize backend/__init__.py** (empty file)

---

### Task 2: Task Domain — Schemas, Service, API

**Files:**
- Create: `backend/app/tasks/__init__.py`
- Create: `backend/app/tasks/schemas.py`
- Create: `backend/app/tasks/service.py`
- Create: `backend/app/tasks/api.py`
- Create: `backend/app/api/__init__.py`

- [ ] **Step 1: Create tasks/__init__.py** (empty)

- [ ] **Step 2: Create tasks/schemas.py**

```python
"""Task Pydantic schemas for API request/response."""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Task title")
    description: str = Field("", max_length=5000, description="Task description")
    priority: str = Field("medium", pattern=r"^(low|medium|high|urgent)$")
    due_date: Optional[datetime.datetime] = Field(None, description="ISO 8601 due date")
    tags: str = Field("", description="Comma-separated tags")
    category: str = Field("general", max_length=100)


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern=r"^(pending|completed|cancelled)$")
    priority: Optional[str] = Field(None, pattern=r"^(low|medium|high|urgent)$")
    due_date: Optional[datetime.datetime] = None
    tags: Optional[str] = None
    category: Optional[str] = None


class TaskRead(BaseModel):
    id: int
    title: str
    description: str
    status: str
    priority: str
    due_date: Optional[datetime.datetime] = None
    tags: str
    category: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    tasks: list[TaskRead]
    total: int
```

- [ ] **Step 3: Create tasks/service.py**

```python
"""Task business logic layer."""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.logger import get_logger
from app.database.models import Task
from app.tasks.schemas import TaskCreate, TaskUpdate

logger = get_logger(__name__)


class TaskService:
    """CRUD operations for tasks."""

    async def create(self, db: AsyncSession, data: TaskCreate) -> Task:
        """Create a new task."""
        task = Task(
            title=data.title,
            description=data.description,
            priority=data.priority,
            due_date=data.due_date,
            tags=data.tags,
            category=data.category,
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        logger.info("task_created", task_id=task.id, title=task.title)
        return task

    async def get(self, db: AsyncSession, task_id: int) -> Task:
        """Get a task by ID."""
        task = await db.get(Task, task_id)
        if not task:
            raise NotFoundError("Task", task_id)
        return task

    async def list(
        self,
        db: AsyncSession,
        status: Optional[str] = None,
        category: Optional[str] = None,
        priority: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Task], int]:
        """List tasks with optional filters and search."""
        query = select(Task)

        if status:
            query = query.where(Task.status == status)
        if category:
            query = query.where(Task.category == category)
        if priority:
            query = query.where(Task.priority == priority)
        if search:
            search_term = f"%{search}%"
            query = query.where(
                Task.title.ilike(search_term) | Task.description.ilike(search_term) | Task.tags.ilike(search_term)
            )

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply ordering, pagination
        query = query.order_by(Task.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        tasks = list(result.scalars().all())

        return tasks, total

    async def update(self, db: AsyncSession, task_id: int, data: TaskUpdate) -> Task:
        """Update a task."""
        task = await self.get(db, task_id)

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)

        task.updated_at = datetime.datetime.now()
        await db.commit()
        await db.refresh(task)
        logger.info("task_updated", task_id=task.id, changes=update_data)
        return task

    async def complete(self, db: AsyncSession, task_id: int) -> Task:
        """Mark a task as completed."""
        return await self.update(db, task_id, TaskUpdate(status="completed"))

    async def delete(self, db: AsyncSession, task_id: int) -> None:
        """Delete a task."""
        task = await self.get(db, task_id)
        await db.delete(task)
        await db.commit()
        logger.info("task_deleted", task_id=task_id)

    async def search(self, db: AsyncSession, query_str: str, limit: int = 20) -> tuple[list[Task], int]:
        """Search tasks by title, description, and tags."""
        return await self.list(db, search=query_str, limit=limit)


# Singleton
task_service = TaskService()
```

- [ ] **Step 4: Create tasks/api.py**

```python
"""Task REST API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.tasks.schemas import TaskCreate, TaskRead, TaskListResponse, TaskUpdate
from app.tasks.service import task_service

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("", response_model=TaskRead, status_code=201)
async def create_task(data: TaskCreate, db: AsyncSession = Depends(get_db)):
    """Create a new task."""
    return await task_service.create(db, data)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, pattern=r"^(pending|completed|cancelled)$"),
    category: Optional[str] = None,
    priority: Optional[str] = Query(None, pattern=r"^(low|medium|high|urgent)$"),
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List tasks with optional filters."""
    tasks, total = await task_service.list(db, status, category, priority, search, limit, offset)
    return TaskListResponse(tasks=tasks, total=total)


@router.get("/{task_id}", response_model=TaskRead)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Get a task by ID."""
    return await task_service.get(db, task_id)


@router.patch("/{task_id}", response_model=TaskRead)
async def update_task(task_id: int, data: TaskUpdate, db: AsyncSession = Depends(get_db)):
    """Update a task."""
    return await task_service.update(db, task_id, data)


@router.patch("/{task_id}/complete", response_model=TaskRead)
async def complete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Mark a task as completed."""
    return await task_service.complete(db, task_id)


@router.delete("/{task_id}", status_code=204)
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a task."""
    await task_service.delete(db, task_id)


@router.get("/search/", response_model=TaskListResponse)
async def search_tasks(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Search tasks by query string."""
    tasks, total = await task_service.search(db, q, limit)
    return TaskListResponse(tasks=tasks, total=total)
```

- [ ] **Step 5: Create api/__init__.py** (empty)

---

### Task 3: Voice Module — Intent Router

**Files:**
- Create: `backend/app/voice/__init__.py`
- Create: `backend/app/voice/router.py`

- [ ] **Step 1: Create voice/__init__.py** (empty)

- [ ] **Step 2: Create voice/router.py**

```python
"""Intent router — keyword → regex → fuzzy matching pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from thefuzz import fuzz


class IntentType(str, Enum):
    TASK_CREATE = "TASK_CREATE"
    TASK_LIST = "TASK_LIST"
    TASK_COMPLETE = "TASK_COMPLETE"
    TASK_DELETE = "TASK_DELETE"
    TASK_SEARCH = "TASK_SEARCH"

    NOTE_CREATE = "NOTE_CREATE"
    NOTE_SEARCH = "NOTE_SEARCH"

    REMINDER_CREATE = "REMINDER_CREATE"

    MEMORY_SAVE = "MEMORY_SAVE"
    MEMORY_RECALL = "MEMORY_RECALL"

    UNKNOWN = "UNKNOWN"


@dataclass
class IntentResult:
    type: IntentType
    confidence: float  # 0.0 to 1.0
    entities: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""

    def is_known(self) -> bool:
        return self.type != IntentType.UNKNOWN


# ── Pattern Definitions ────────────────────────────────────────

EXACT_PATTERNS: dict[str, IntentType] = {
    "show my tasks": IntentType.TASK_LIST,
    "show my pending tasks": IntentType.TASK_LIST,
    "show my completed tasks": IntentType.TASK_LIST,
    "list my tasks": IntentType.TASK_LIST,
    "what are my tasks": IntentType.TASK_LIST,
}

REGEX_PATTERNS: list[tuple[re.Pattern, IntentType, list[str]]] = [
    # TASK_CREATE
    (re.compile(r"(?:create|add|make|new)\s+(?:a\s+)?(?:task|todo)\s+(?:to\s+)?(?:called\s+)?(?:titled\s+)?['\"]?(.+?)['\"]?(?:\s+(?:with|by|due|priority|on|for))?(?:\s+(?:high|medium|low|urgent))?(?:\s+priority)?(?:\s+due\s+(.+?))?$", re.IGNORECASE),
     IntentType.TASK_CREATE, ["title", "due_date"]),

    # TASK_LIST
    (re.compile(r"(?:show|list|get|display|what(?:'s| is| are))\s+(?:my\s+)?(?:pending\s+)?(?:completed\s+)?(?:tasks|task|todos|todo)", re.IGNORECASE),
     IntentType.TASK_LIST, []),

    # TASK_COMPLETE
    (re.compile(r"(?:complete|mark\s+(?:as\s+)?done|finish|done)\s+(?:my\s+)?(?:task\s+)?(?:called\s+)?(?:titled\s+)?['\"]?(.+?)['\"]?$", re.IGNORECASE),
     IntentType.TASK_COMPLETE, ["title"]),

    # TASK_DELETE
    (re.compile(r"(?:delete|remove|erase)\s+(?:my\s+)?(?:task\s+)?(?:called\s+)?(?:titled\s+)?['\"]?(.+?)['\"]?$", re.IGNORECASE),
     IntentType.TASK_DELETE, ["title"]),

    # TASK_SEARCH
    (re.compile(r"(?:search|find|look\s+(?:for|up))\s+(?:my\s+)?(?:tasks?\s+(?:about|for|with|containing)\s+)?['\"]?(.+?)['\"]?$", re.IGNORECASE),
     IntentType.TASK_SEARCH, ["query"]),
]

FUZZY_THRESHOLD = 75

FUZZY_PATTERNS: list[tuple[str, IntentType]] = [
    ("create task", IntentType.TASK_CREATE),
    ("add task", IntentType.TASK_CREATE),
    ("new task", IntentType.TASK_CREATE),
    ("show tasks", IntentType.TASK_LIST),
    ("list tasks", IntentType.TASK_LIST),
    ("my tasks", IntentType.TASK_LIST),
    ("complete task", IntentType.TASK_COMPLETE),
    ("mark done", IntentType.TASK_COMPLETE),
    ("finish task", IntentType.TASK_COMPLETE),
    ("delete task", IntentType.TASK_COMPLETE),
    ("remove task", IntentType.TASK_DELETE),
    ("search task", IntentType.TASK_SEARCH),
    ("find task", IntentType.TASK_SEARCH),
    ("create note", IntentType.NOTE_CREATE),
    ("find note", IntentType.NOTE_SEARCH),
    ("remind me", IntentType.REMINDER_CREATE),
    ("remember", IntentType.MEMORY_SAVE),
    ("recall", IntentType.MEMORY_RECALL),
    ("what do you know", IntentType.MEMORY_RECALL),
]


class IntentRouter:
    """Routes natural language to structured intents."""

    def route(self, text: str) -> IntentResult:
        """Route text through the matching pipeline."""
        text = text.strip().lower()
        result = IntentResult(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)

        # 1. Exact match
        if text in EXACT_PATTERNS:
            result.type = EXACT_PATTERNS[text]
            result.confidence = 1.0
            return result

        # 2. Regex match
        for pattern, intent_type, entity_keys in REGEX_PATTERNS:
            match = pattern.search(text)
            if match:
                result.type = intent_type
                result.confidence = 0.95
                for i, key in enumerate(entity_keys):
                    if i < len(match.groups()) and match.group(i + 1):
                        result.entities[key] = match.group(i + 1).strip()
                return result

        # 3. Fuzzy match
        best_score = 0
        best_intent = IntentType.UNKNOWN
        for pattern_text, intent_type in FUZZY_PATTERNS:
            score = fuzz.ratio(text, pattern_text)
            if score > best_score:
                best_score = score
                best_intent = intent_type

        if best_score >= FUZZY_THRESHOLD:
            result.type = best_intent
            result.confidence = best_score / 100.0
            return result

        return result


# Singleton
intent_router = IntentRouter()
```

---

### Task 4: Voice Module — STT and TTS

**Files:**
- Create: `backend/app/voice/stt.py`
- Create: `backend/app/voice/tts.py`

- [ ] **Step 1: Create voice/stt.py**

```python
"""Vosk speech-to-text wrapper."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Optional

import vosk

from app.core.config import settings
from app.core.exceptions import VoiceProcessingError
from app.core.logger import get_logger

logger = get_logger(__name__)


class VoskSTT:
    """Offline speech-to-text using Vosk."""

    def __init__(self):
        self.model: Optional[vosk.Model] = None
        self.recognizer: Optional[vosk.KaldiRecognizer] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load the Vosk model."""
        model_path = Path(settings.VOSK_MODEL_PATH)
        if not model_path.exists():
            logger.warning("vosk_model_not_found", path=str(model_path))
            raise VoiceProcessingError(f"Vosk model not found at {model_path}")

        self.model = vosk.Model(str(model_path))
        self.recognizer = vosk.KaldiRecognizer(self.model, settings.VOSK_SAMPLE_RATE)
        self._initialized = True
        logger.info("vosk_model_loaded", path=str(model_path))

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe audio file to text."""
        if not self._initialized:
            raise VoiceProcessingError("Vosk model not initialized")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise VoiceProcessingError(f"Audio file not found: {audio_path}")

        wf = wave.open(str(audio_path), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise VoiceProcessingError("Audio must be WAV: mono, 16-bit")

        recognizer = vosk.KaldiRecognizer(self.model, wf.getframerate())
        recognizer.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        logger.info("stt_transcribed", text=text, audio=str(audio_path))
        return text

    async def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe raw PCM audio bytes."""
        if not self._initialized:
            raise VoiceProcessingError("Vosk model not initialized")

        recognizer = vosk.KaldiRecognizer(self.model, settings.VOSK_SAMPLE_RATE)

        if recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(recognizer.Result())
        else:
            result = json.loads(recognizer.FinalResult())

        text = result.get("text", "").strip()
        logger.info("stt_transcribed_bytes", text=text, length=len(audio_bytes))
        return text

    async def close(self) -> None:
        """Cleanup resources."""
        self.model = None
        self.recognizer = None
        self._initialized = False
        logger.info("vosk_model_unloaded")


# Singleton
vosk_stt = VoskSTT()
```

- [ ] **Step 2: Create voice/tts.py**

```python
"""Text-to-speech using Edge TTS with Piper TTS fallback."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EdgeTTS:
    """Text-to-speech using Edge TTS (online, high quality)."""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> None:
        """Check Edge TTS availability."""
        try:
            import edge_tts
            # Verify by listing voices
            voices = await edge_tts.list_voices()
            logger.info("edge_tts_available", voice_count=len(voices))
            self._initialized = True
        except Exception as e:
            logger.warning("edge_tts_unavailable", error=str(e))
            self._initialized = False

    async def synthesize(self, text: str, output_path: Optional[str | Path] = None) -> Path:
        """Synthesize text to speech audio file."""
        import edge_tts

        if output_path is None:
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            output_path = Path(path)
        else:
            output_path = Path(output_path)

        communicate = edge_tts.Communicate(text, settings.TTS_VOICE)
        await communicate.save(str(output_path))

        logger.info("tts_synthesized", text_len=len(text), output=str(output_path))
        return output_path

    async def close(self) -> None:
        """Cleanup."""
        self._initialized = False


# Singleton
edge_tts = EdgeTTS()
```

---

### Task 5: Voice API Endpoint

**Files:**
- Create: `backend/app/api/voice.py`

- [ ] **Step 1: Create api/voice.py**

```python
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
from app.voice.tts import edge_tts

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
    logger.info("voice_command_routed", intent=result.type.value, confidence=result.confidence, text=text)

    if not result.is_known():
        return VoiceCommandResponse(
            intent="UNKNOWN",
            spoken_response=f"Sorry, I didn't understand that. Can you rephrase?",
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
        query = result.entities.get("query", result.raw_text.replace("search", "").strip())
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
```

---

### Task 6: Main Application Entry Point

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/start.sh`

- [ ] **Step 1: Create app/main.py**

```python
"""Jarvis Voice Productivity Assistant — Main Application Entry Point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.database.migrations import create_tables

# Setup logging on import
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    # ── Startup ──
    logger.info(
        "Starting Jarvis Voice Assistant",
        version=settings.APP_VERSION,
    )

    # Create database tables
    await create_tables()

    # Initialize voice services (optional for now)
    try:
        from app.voice.tts import edge_tts
        await edge_tts.initialize()
    except Exception as e:
        logger.warning("tts_init_skipped", error=str(e))

    yield

    # ── Shutdown ──
    logger.info("Shutting down Jarvis Voice Assistant")

    try:
        from app.voice.tts import edge_tts
        await edge_tts.close()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Health Check ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME,
    }


# ── Mount Routers ─────────────────────────────────────────────

from app.tasks.api import router as tasks_router
from app.api.voice import router as voice_router

app.include_router(tasks_router)
app.include_router(voice_router)
```

- [ ] **Step 2: Create start.sh**

```bash
#!/bin/bash
# Jarvis Voice Assistant — Start Script
set -e

cd "$(dirname "$0")"

# Create data directory if needed
mkdir -p data models

# Run the server
uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --reload
```

```bash
chmod +x start.sh
```

- [ ] **Step 3: Test application starts**

```bash
cd /root/backend && python -c "
import asyncio
from app.database.connection import engine
from app.database.migrations import create_tables
asyncio.run(create_tables())
print('Database tables created successfully')
"
```

---

### Task 7: Tests

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_tasks.py`
- Create: `backend/tests/test_router.py`
- Create: `backend/tests/test_voice.py`

- [ ] **Step 1: Create tests/__init__.py** (empty)

- [ ] **Step 2: Create tests/conftest.py**

```python
"""Test fixtures and configuration."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.database.connection import Base
from app.database.models import Task
from app.main import app
from app.core.deps import get_db

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_jarvis.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Create tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Override dependency to use test database."""
    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Database session for direct model access."""
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def sample_task(db_session: AsyncSession) -> Task:
    """Create a sample task for testing."""
    task = Task(title="Test Task", description="A test task", priority="high")
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)
    return task
```

- [ ] **Step 3: Create tests/test_tasks.py**

```python
"""Task API tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestTaskAPI:
    """Test suite for task CRUD endpoints."""

    async def test_create_task(self, client: AsyncClient):
        response = await client.post("/api/tasks", json={
            "title": "Learn FastAPI",
            "description": "Study FastAPI documentation",
            "priority": "high",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Learn FastAPI"
        assert data["status"] == "pending"
        assert data["priority"] == "high"
        assert "id" in data

    async def test_create_task_empty_title(self, client: AsyncClient):
        response = await client.post("/api/tasks", json={"title": ""})
        assert response.status_code == 422

    async def test_list_tasks_empty(self, client: AsyncClient):
        response = await client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["tasks"] == []

    async def test_list_tasks_with_data(self, client: AsyncClient):
        # Create two tasks
        await client.post("/api/tasks", json={"title": "Task 1"})
        await client.post("/api/tasks", json={"title": "Task 2"})

        response = await client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["tasks"]) == 2

    async def test_list_tasks_filter_status(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Pending Task"})
        resp = await client.post("/api/tasks", json={"title": "Done Task"})
        task_id = resp.json()["id"]
        await client.patch(f"/api/tasks/{task_id}/complete")

        response = await client.get("/api/tasks?status=completed")
        data = response.json()
        assert data["total"] == 1
        assert data["tasks"][0]["title"] == "Done Task"

    async def test_get_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Get Me"})
        task_id = create_resp.json()["id"]

        response = await client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Get Me"

    async def test_get_task_not_found(self, client: AsyncClient):
        response = await client.get("/api/tasks/999")
        assert response.status_code == 404

    async def test_complete_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Complete Me"})
        task_id = create_resp.json()["id"]

        response = await client.patch(f"/api/tasks/{task_id}/complete")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    async def test_delete_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Delete Me"})
        task_id = create_resp.json()["id"]

        response = await client.delete(f"/api/tasks/{task_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_resp = await client.get(f"/api/tasks/{task_id}")
        assert get_resp.status_code == 404

    async def test_search_tasks(self, client: AsyncClient):
        await client.post("/api/tasks", json={"title": "Learn Python"})
        await client.post("/api/tasks", json={"title": "Learn Rust"})
        await client.post("/api/tasks", json={"title": "Write Tests"})

        response = await client.get("/api/tasks/search/?q=Learn")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    async def test_update_task(self, client: AsyncClient):
        create_resp = await client.post("/api/tasks", json={"title": "Original"})
        task_id = create_resp.json()["id"]

        response = await client.patch(f"/api/tasks/{task_id}", json={
            "title": "Updated",
            "priority": "urgent",
        })
        assert response.status_code == 200
        assert response.json()["title"] == "Updated"
        assert response.json()["priority"] == "urgent"
```

- [ ] **Step 4: Create tests/test_router.py**

```python
"""Intent router tests."""

from __future__ import annotations

from app.voice.router import IntentType, intent_router


class TestIntentRouter:
    """Test suite for the intent router."""

    def test_exact_match(self):
        result = intent_router.route("show my tasks")
        assert result.type == IntentType.TASK_LIST
        assert result.confidence == 1.0

    def test_exact_match_pending(self):
        result = intent_router.route("show my pending tasks")
        assert result.type == IntentType.TASK_LIST

    def test_regex_create_task(self):
        result = intent_router.route("create a task to learn FastAPI")
        assert result.type == IntentType.TASK_CREATE
        assert result.confidence == 0.95
        assert "title" in result.entities

    def test_regex_create_simple(self):
        result = intent_router.route("create task buy groceries")
        assert result.type == IntentType.TASK_CREATE
        assert "groceries" in result.entities.get("title", "")

    def test_regex_complete_task(self):
        result = intent_router.route("complete my task learn FastAPI")
        assert result.type == IntentType.TASK_COMPLETE
        assert result.confidence == 0.95

    def test_regex_complete_simple(self):
        result = intent_router.route("complete learn FastAPI")
        assert result.type == IntentType.TASK_COMPLETE

    def test_regex_delete_task(self):
        result = intent_router.route("delete my task test note")
        assert result.type == IntentType.TASK_DELETE

    def test_regex_search(self):
        result = intent_router.route("search my tasks about FastAPI")
        assert result.type == IntentType.TASK_SEARCH

    def test_regex_search_find(self):
        result = intent_router.route("find FastAPI")
        assert result.type == IntentType.TASK_SEARCH

    def test_fuzzy_match_create(self):
        result = intent_router.route("make a new task")
        # Should fuzzy match to task create
        assert result.is_known()

    def test_unknown_intent(self):
        result = intent_router.route("what is the weather today")
        assert result.type == IntentType.UNKNOWN
        assert not result.is_known()

    def test_case_insensitive(self):
        result = intent_router.route("SHOW MY TASKS")
        assert result.type == IntentType.TASK_LIST

    def test_empty_string(self):
        result = intent_router.route("")
        assert result.type == IntentType.UNKNOWN
```

- [ ] **Step 5: Create tests/test_voice.py**

```python
"""Voice command integration tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestVoiceAPI:
    """Test suite for voice command API."""

    async def test_voice_create_task(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "create a task to learn FastAPI",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_CREATE"
        assert "Task created" in data["spoken_response"]
        assert data["data"] is not None

    async def test_voice_list_tasks(self, client: AsyncClient):
        # Create one task first
        await client.post("/api/voice/command", json={
            "text": "create a task learn Python",
        })

        response = await client.post("/api/voice/command", json={
            "text": "show my tasks",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_LIST"
        assert "Python" in data["spoken_response"]

    async def test_voice_complete_task(self, client: AsyncClient):
        await client.post("/api/voice/command", json={
            "text": "create a task review code",
        })

        response = await client.post("/api/voice/command", json={
            "text": "complete my task review code",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_COMPLETE"
        assert "completed" in data["spoken_response"]

    async def test_voice_search_tasks(self, client: AsyncClient):
        await client.post("/api/voice/command", json={
            "text": "create a task learn WebRTC",
        })

        response = await client.post("/api/voice/command", json={
            "text": "search my tasks about WebRTC",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "TASK_SEARCH"

    async def test_voice_unknown_command(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "what is the meaning of life",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "UNKNOWN"
        assert "didn't understand" in data["spoken_response"]

    async def test_voice_empty_text(self, client: AsyncClient):
        response = await client.post("/api/voice/command", json={
            "text": "",
        })
        assert response.status_code == 400
```

- [ ] **Step 6: Run all tests**

```bash
cd /root/backend && python -m pytest tests/ -v --asyncio-mode=auto
```

---

### Task 8: Docker & Deployment Setup

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/docker-compose.yml`
- Modify: `backend/.gitignore`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# Jarvis Voice Assistant — Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.9"

services:
  jarvis:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
```

- [ ] **Step 3: Update .gitignore**

Append to existing `backend/.gitignore`:

```gitignore
# Jarvis Voice Assistant
*.db
data/
models/vosk-*/
```
