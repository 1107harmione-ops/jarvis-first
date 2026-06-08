# Jarvis Voice Productivity Assistant — Design Document

## Overview

A personal voice-first productivity assistant focused on task management, notes, reminders, voice commands, personal memory, and productivity workflows. Single-user, offline-friendly, low-resource system using FastAPI + SQLite + Redis + Vosk STT + Edge TTS.

## Core Principles

1. **Voice First** — primary interaction is voice
2. **Offline Friendly** — core features work without internet
3. **Fast Response** — sub-second intent routing
4. **Low Resource Usage** — runs on a Raspberry Pi class device
5. **Single User** — no login, no registration, no JWT
6. **Modular Architecture** — each domain is independent
7. **Easy Future Expansion** — plugin system ready
8. **Reliability Over Features** — 95%+ voice command success before adding features

## Tech Stack

- **Backend:** FastAPI (Python 3.12+)
- **Database:** SQLite (V1), PostgreSQL-ready schema
- **Cache:** Redis (temporary state, queue, rate limiting)
- **Queue:** Redis Queue (RQ)
- **STT:** Vosk (offline speech-to-text)
- **TTS:** Edge TTS (primary), Piper TTS (fallback)
- **Search:** PostgreSQL Full Text Search ready design
- **Logging:** Structured JSON logging
- **Config:** .env + Pydantic Settings
- **Deployment:** Render + VPS ready

## Architecture

### Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, lifespan, router includes
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings from .env
│   │   ├── logger.py              # Structured JSON logging
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── deps.py                # FastAPI dependency injection
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py          # SQLite connection management
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   └── migrations.py          # Schema creation / migration
│   │
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── router.py              # Intent router (keyword → regex → fuzzy)
│   │   ├── stt.py                 # Vosk STT wrapper
│   │   ├── tts.py                 # Edge TTS + Piper fallback
│   │   ├── vad.py                 # Voice activity detection
│   │   └── wakeword.py            # OpenWakeWord detector
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── models.py              # Task ORM model
│   │   ├── schemas.py             # Pydantic schemas
│   │   ├── service.py             # Business logic
│   │   └── api.py                 # REST + voice endpoints
│   │
│   ├── notes/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── api.py
│   │
│   ├── reminders/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   ├── api.py
│   │   └── scheduler.py           # RQ scheduler for reminders
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── service.py             # Auto-save preferences, facts
│   │   └── api.py
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   └── service.py             # Cross-entity full text search
│   │
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── registry.py            # Plugin registry
│   │   ├── loader.py              # Dynamic plugin loader
│   │   └── base.py                # Plugin base class
│   │
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── service.py
│   │   └── api.py
│   │
│   ├── backup/
│   │   ├── __init__.py
│   │   ├── exporter.py
│   │   ├── importer.py
│   │   └── service.py
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── tracker.py             # Event tracking
│   │   └── service.py
│   │
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── service.py             # Voice + UI notifications
│   │
│   └── workers/
│       ├── __init__.py
│       ├── reminder_worker.py     # RQ worker for reminders
│       └── tts_worker.py          # Background TTS generation
│
├── tests/
│   ├── conftest.py
│   ├── test_tasks.py
│   ├── test_notes.py
│   ├── test_reminders.py
│   ├── test_memory.py
│   ├── test_router.py
│   ├── test_voice.py
│   └── test_integration.py
│
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── start.sh
```

### Data Flow

```
Voice Input
    ↓
VAD (detect speech start/end)
    ↓
Vosk STT (speech → text)
    ↓
Intent Router (keyword → regex → fuzzy)
    ↓
Domain Service (tasks/notes/reminders/memory)
    ↓
SQLite (persist)
    ↓
Response Generator (text response)
    ↓
Edge TTS (text → speech)
    ↓
Audio Output
```

## Phase 1: Voice Task Manager MVP

### Task Schema

```sql
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',       -- pending, completed, cancelled
    priority TEXT DEFAULT 'medium',      -- low, medium, high, urgent
    due_date TEXT,                        -- ISO 8601 datetime
    tags TEXT DEFAULT '',                 -- comma-separated
    category TEXT DEFAULT 'general',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### Supported Commands

| Command | Example | Intent |
|---------|---------|--------|
| Create | "Create a task to learn FastAPI tomorrow" | TASK_CREATE |
| List | "Show my pending tasks" / "Show my tasks" | TASK_LIST |
| Complete | "Complete my FastAPI task" | TASK_COMPLETE |
| Delete | "Delete my test task" | TASK_DELETE |
| Search | "Search my tasks about FastAPI" | TASK_SEARCH |

### API Endpoints

```
POST   /api/tasks              → Create task
GET    /api/tasks              → List tasks (query: status, category, search)
GET    /api/tasks/{id}         → Get task by ID
PATCH  /api/tasks/{id}         → Update task (e.g., complete)
DELETE /api/tasks/{id}         → Delete task
GET    /api/tasks/search?q=    → Search tasks
```

## Phase 2: Intent Router

### Routing Pipeline

1. **Exact Match** — predefined command patterns match verbatim
2. **Regex Match** — pattern-based intent extraction
3. **Fuzzy Match** — Levenshtein distance for close matches
4. **Clarification** — return ambiguity for user to clarify

### Supported Intents

```
TASK_CREATE, TASK_LIST, TASK_COMPLETE, TASK_DELETE, TASK_SEARCH
NOTE_CREATE, NOTE_SEARCH
REMINDER_CREATE
MEMORY_SAVE, MEMORY_RECALL
```

## Phase 3: Notes System

### Note Schema

```sql
CREATE TABLE notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT DEFAULT '',
    category TEXT DEFAULT 'personal',  -- learning, project, personal, ideas, research
    tags TEXT DEFAULT '',
    priority TEXT DEFAULT 'medium',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### Voice Commands

"Create note about WebRTC"
"Find note about FastAPI"
"Update my WebRTC note"
"Delete my old note"
"Search notes about Python"

## Phase 4: Reminder System

### Reminder Schema

```sql
CREATE TABLE reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    reminder_time TEXT NOT NULL,         -- ISO 8601
    repeat_type TEXT DEFAULT 'once',     -- once, daily, weekly, custom
    repeat_interval INTEGER DEFAULT 0,   -- in seconds for custom
    status TEXT DEFAULT 'pending',       -- pending, triggered, cancelled
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### Flow

```
User: "Remind me tomorrow at 8 PM"
    → Parse time expression
    → Create reminder in SQLite
    → Schedule RQ job
    → At trigger time → voice notification
```

## Phase 5: Memory System

### Memory Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',     -- preference, goal, project, skill, fact
    importance REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now'))
);
```

### Auto-Save Rules

**Always Save:** Preferences, goals, projects, skills, long-term facts
**Never Save:** Greetings, small talk, temporary requests

## Phase 6: Search System

### Full-Text Search Schema (SQLite FTS5)

```sql
CREATE VIRTUAL TABLE tasks_fts USING fts5(
    title, description, tags, content='tasks', content_rowid='id'
);
```

Similar FTS tables for notes, reminders, memories.

### Ranking: Title > Tags > Content > Recency

## Phase 7: Voice System

### STT (Vosk)
- Offline speech recognition
- Model: vosk-model-small-en-us-0.15
- Async processing pipeline

### TTS (Edge TTS → Piper)
- Primary: Edge TTS (online, high quality)
- Fallback: Piper TTS (offline, local)
- Auto-detect network availability

### VAD (WebRTC VAD + Silence Detection)
- Hybrid approach
- Detect speech start/end
- Auto-trigger processing on silence

## Phase 8: Wake Word System

### Primary: OpenWakeWord
- Wake words: "Jarvis", "Assistant"
- Always-on listening mode
- Low CPU usage

### Fallback: Push To Talk
- Keyboard shortcut or button
- Bypasses wake word

### States
Idle → Listening → Processing → Speaking → Error

## Phase 9-18: Future Phases

Detailed designs for plugins, permissions, settings, backup, Redis, logging, analytics, CI/CD, testing, and WebRTC will be added incrementally as each phase is implemented.

## Definition of Success

The assistant must reliably perform using voice only:
1. Create Task
2. List Task
3. Complete Task
4. Delete Task
5. Create Note
6. Search Note
7. Create Reminder
8. Recall Memory

**Target:** 95%+ successful voice command execution.

Do not add new features until this target is achieved.

## Testing Strategy

- **Unit Tests:** Router, memory, tasks, notes
- **Integration Tests:** Voice → Intent → Service → Database
- **Test Framework:** pytest + pytest-asyncio
- **Coverage Target:** 85%+ for core modules

## Deployment

- Docker Compose for local development
- Render.com for production
- SQLite + Redis in production
- Optional PostgreSQL migration path
