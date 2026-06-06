# JARVIS Memory Architecture — MongoDB Design Spec

**Date:** 2026-06-06
**Status:** Draft
**Author:** AI Architecture Team

---

## 1. Overview

### 1.1 Purpose

Design and implement a production-grade memory architecture for **JARVIS**, an AI assistant, using **MongoDB Atlas** with the **Motor** async Python driver. The system replaces the existing SQLite+ChromaDB memory layer with a scalable, semantic-aware, long-term memory architecture that:

- Remembers users across months and years
- Learns user preferences and behavior patterns
- Automatically retrieves relevant memories during conversations
- Supports multiple memory types (STM, LTM, episodic, semantic, etc.)
- Uses vector embeddings for semantic search via MongoDB Atlas `$vectorSearch`
- Provides a structured context builder for LLM injection

### 1.2 Scope

This spec covers the standalone `jarvis-memory` Python package — a drop-in memory backend for any AI assistant. It includes:

- MongoDB schema design (9 collections)
- Index architecture (BSON + Atlas Search)
- Pydantic data models
- Async CRUD repositories
- Memory scoring and ranking system
- Vector embedding + semantic search pipeline
- Context builder for LLM prompt enrichment
- Memory lifecycle (STM → LTM consolidation, summarization, forgetting)
- REST API layer (optional, FastAPI-based)
- Production deployment guide

### 1.3 Out of Scope

- Integration with the existing planed_jv JARVIS codebase (done as a follow-up)
- UI dashboards or memory visualization
- Multi-modal memory (image/audio embeddings — future phase)
- Real-time memory replication across regions

---

## 2. Package Structure

```
jarvis-memory/
├── jarvis_memory/
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Configuration (pydantic-settings)
│   ├── database.py                 # Motor client singleton + lifecycle
│   ├── models/                     # Pydantic v2 models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── conversation.py
│   │   ├── message.py
│   │   ├── memory.py               # All memory types (single polymorphic model)
│   │   ├── task.py
│   │   ├── knowledge.py
│   │   ├── agent_log.py
│   │   ├── analytics.py
│   │   └── settings.py
│   ├── repositories/               # Async data access layer
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseRepository[T] (generic CRUD)
│   │   ├── user_repo.py
│   │   ├── conversation_repo.py
│   │   ├── message_repo.py
│   │   ├── memory_repo.py
│   │   ├── task_repo.py
│   │   └── knowledge_repo.py
│   ├── services/                   # Business logic
│   │   ├── __init__.py
│   │   ├── memory_service.py       # CRUD + lifecycle + consolidation
│   │   ├── retrieval_service.py    # Multi-strategy retrieval + fusion
│   │   ├── context_builder.py      # LLM context assembly
│   │   ├── embedding_service.py    # Embedding generation (model-agnostic)
│   │   ├── scoring_service.py      # Importance + ranking scoring
│   │   ├── consolidation_service.py# STM→LTM, summarization, forgetting
│   │   └── analytics_service.py    # Usage analytics
│   ├── api/                        # FastAPI REST layer (optional)
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI application
│   │   ├── routes.py               # Route definitions
│   │   └── schemas.py              # Pydantic request/response schemas
│   └── cli.py                      # CLI (typer) for management
├── tests/
│   ├── conftest.py                 # Async fixtures + mongomock
│   ├── test_models/
│   ├── test_repositories/
│   ├── test_services/
│   └── test_api/
├── scripts/
│   ├── create_indexes.py           # Atlas index creation script
│   └── seed_data.py                # Development seed data
├── deploy/
│   ├── docker-compose.yml          # For local MongoDB (no Atlas)
│   ├── Dockerfile
│   └── mongo_indexes.json          # Atlas search index definitions
├── pyproject.toml                  # Dependencies, build config
├── ARCHITECTURE.md                 # Implementation guide
└── README.md
```

---

## 3. Data Models

### 3.1 User Model (`users` collection)

```python
class UserProfile(BaseModel):
    name: str | None = None
    timezone: str = "UTC"
    location: str | None = None
    bio: str | None = None

class UserPreferences(BaseModel):
    language: str = "en"
    voice_speed: float = 1.0
    voice_pitch: str = "neutral"
    response_style: str = "concise"  # concise | detailed | humorous
    theme: str = "dark"
    notification_enabled: bool = True

class UserVoiceSettings(BaseModel):
    voice_id: str = "default"
    wake_word: str = "jarvis"
    stt_language: str = "en-US"
    tts_speed: float = 1.0

class UserDocument(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str                                  # Unique external ID (UUID)
    username: str
    email: str
    password_hash: str | None = None              # Optional auth
    profile: UserProfile = Field(default_factory=UserProfile)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    voice_settings: UserVoiceSettings = Field(default_factory=UserVoiceSettings)
    personal_data: dict[str, Any] = Field(default_factory=dict)  # Extensible
    tags: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### 3.2 Conversation Model (`conversations` collection)

```python
class ConversationDocument(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    conversation_id: str
    user_id: str
    session_id: str = "default"
    title: str = ""
    summary: str = ""                              # Auto-generated summary
    message_count: int = 0
    tokens_used: int = 0
    tags: list[str] = Field(default_factory=list)
    is_archived: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### 3.3 Message Model (`messages` collection)

```python
class MessageDocument(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    message_id: str
    conversation_id: str
    user_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    content_type: str = "text"                     # text | code | image | tool_call
    intent: str | None = None                      # Classified intent
    embedding: list[float] | None = None            # 384-dim vector
    importance_score: float = 0.0
    tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 3.4 Memory Model (`memories` collection) — Central Polymorphic Model

```python
class MemoryDocument(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    memory_id: str
    user_id: str
    memory_type: Literal[
        "short_term", "long_term", "semantic",
        "episodic", "user_preference", "task", "knowledge"
    ]
    importance_score: float = 0.0                  # [0, 1] computed
    content: str                                    # The memory text
    embedding: list[float] | None = None            # Vector for semantic search
    summary: str | None = None                      # Auto-generated summary
    tags: list[str] = Field(default_factory=list)
    source: str | None = None                       # conversation | user_input | system | knowledge_base
    source_id: str | None = None                    # ID of source conversation/message
    context: dict[str, Any] = Field(default_factory=dict)  # Extensible context
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Lifecycle
    access_count: int = 0
    last_accessed: datetime | None = None
    decay_rate: float = 0.1                        # For computing decay
    consolidated: bool = False                      # True if STM→LTM promoted
    expires_at: datetime | None = None              # TTL for STM (24h)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

**Memory type semantics:**

| Type | TTL | Consolidation | Description |
|------|-----|---------------|-------------|
| `short_term` | 24h | Auto-promoted to LTM if high importance | Recent conversation context |
| `long_term` | — | Manually or via consolidation | Enduring personal facts |
| `semantic` | — | Summarization pipeline | General concepts, learned knowledge |
| `episodic` | — | Direct save | Specific events with timestamp |
| `user_preference` | — | Learned from behavior | "User prefers dark mode" |
| `task` | — | Created/done lifecycle | "Remind me to buy milk" |
| `knowledge` | — | External data | Documents, research, articles |

### 3.5 Task Model (`tasks` collection)

```python
class TaskDocument(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    task_id: str
    user_id: str
    title: str
    description: str = ""
    status: Literal["pending", "in_progress", "completed", "cancelled", "failed"]
    priority: int = 0                               # 0 (low) - 5 (critical)
    task_type: str = "one_off"                      # one_off | recurring | recurring_template
    recurrence_rule: str | None = None               # cron expression for recurring
    due_at: datetime | None = None
    completed_at: datetime | None = None
    depends_on: list[str] = Field(default_factory=list)  # Task IDs for dependencies
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### 3.6 Knowledge Model (`knowledge` collection)

```python
class KnowledgeDocument(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    knowledge_id: str
    user_id: str | None = None                       # None = global knowledge
    title: str
    content: str
    summary: str | None = None
    url: str | None = None
    source_type: Literal["document", "article", "note", "research", "web", "code", "other"]
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    chunk_index: int = 0                             # For long documents
    chunk_total: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### 3.7 Remaining Models

**AgentLog** (`agent_logs`) — stores system actions, LLM calls, errors. TTL-indexed (30d).
**Analytics** (`analytics`) — aggregated daily/weekly usage metrics per user.
**Settings** (`settings`) — global and per-user configuration key-value store.

---

## 4. Index Design

### 4.1 BSON Indexes

#### `users`
```
{ "user_id": 1 }              UNIQUE — primary lookup
{ "email": 1 }                UNIQUE — auth lookup
{ "username": 1 }             — display lookup
{ "tags": 1 }                 — filtering
{ "is_active": 1, "created_at": -1 }  — active user queries
```

#### `conversations`
```
{ "conversation_id": 1 }      UNIQUE
{ "user_id": 1, "updated_at": -1 }  — user's recent conversations
{ "user_id": 1, "created_at": -1 }  — history
{ "user_id": 1, "tags": 1 }        — tag filtering
{ "session_id": 1, "updated_at": -1 }  — session-based queries
```

#### `messages`
```
{ "message_id": 1 }           UNIQUE
{ "conversation_id": 1, "timestamp": 1 }  — conversation messages in order
{ "user_id": 1, "timestamp": -1 }        — user message history
{ "intent": 1, "timestamp": -1 }         — intent analytics
{ "user_id": 1, "importance_score": -1 } — important messages
```

#### `memories`
```
{ "memory_id": 1 }            UNIQUE
{ "user_id": 1, "memory_type": 1, "importance_score": -1 }
    — primary query pattern: user's memories by type, ranked
{ "user_id": 1, "last_accessed": -1 }
    — recency-based retrieval
{ "user_id": 1, "memory_type": 1, "created_at": -1 }
    — chronological memory browsing
{ "user_id": 1, "access_count": -1 }
    — frequency-based retrieval
{ "user_id": 1, "consolidated": 1, "memory_type": 1 }
    — consolidation queries
{ "tags": 1 }                          — cross-user tag search
{ "expires_at": 1 }                    — TTL monitor (STM cleanup)
{ "embedding": "2dsphere" }            — HNSW index placeholder
    (Note: Atlas $vectorSearch uses a separate Search index, not a 2dsphere index)
```

**Compound index coverage:** The `{user_id, memory_type, importance_score}` compound index supports the most critical query — "get the most important memories of type X for user Y". Adding `last_accessed` and `access_count` as secondary sort keys enables fused ranking queries.

#### `tasks`
```
{ "task_id": 1 }               UNIQUE
{ "user_id": 1, "status": 1, "due_at": 1 }
    — user's pending tasks sorted by due date
{ "user_id": 1, "status": 1, "priority": -1 }
    — high-priority tasks
{ "user_id": 1, "task_type": 1, "status": 1 }
    — recurring task management
{ "user_id": 1, "completed_at": -1 }
    — completed task history
```

#### `knowledge`
```
{ "knowledge_id": 1 }          UNIQUE
{ "user_id": 1, "source_type": 1, "created_at": -1 }
    — user's knowledge by type
{ "tags": 1 }                         — tag search
{ "title": "text", "content": "text" } — Atlas Search text index
{ "user_id": 1, "chunk_index": 1 }
    — document chunk ordering
```

#### `agent_logs`
```
{ "timestamp": 1 }              — default descending sort
{ "user_id": 1, "timestamp": -1 }
{ "level": 1, "timestamp": -1 }
{ "action": 1, "timestamp": -1 }
TTL index on "timestamp" (30 days)
```

#### `analytics`
```
{ "user_id": 1, "period": 1, "date": 1 }  UNIQUE compound
{ "event_type": 1, "date": 1 }
```

#### `settings`
```
{ "scope": 1, "scope_id": 1, "key": 1 }   UNIQUE compound
```

### 4.2 Atlas Search Index (Vector Search)

Create a dedicated **Atlas Search Index** on the `memories` collection:

```json
{
  "name": "memory_vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 384,
        "similarity": "cosine",
        "quantization": "scalar"
      },
      {
        "type": "filter",
        "path": "memory_type"
      },
      {
        "type": "filter",
        "path": "user_id"
      },
      {
        "type": "filter",
        "path": "tags"
      }
    ]
  }
}
```

Second index on `knowledge` for document search:

```json
{
  "name": "knowledge_vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 384,
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "user_id"
      },
      {
        "type": "filter",
        "path": "source_type"
      }
    ]
  }
}
```

> **Note:** Atlas `$vectorSearch` requires an M10+ cluster. The embedding dimension (384) matches `all-MiniLM-L6-v2`.

---

## 5. Scoring & Ranking System

### 5.1 Importance Scoring Formula

Each memory receives a composite score `S ∈ [0, 1]`:

```
S = w₁ · R + w₂ · I + w₃ · F + w₄ · P + w₅ · C
```

Where:

| Factor | Symbol | Weight (default) | Description |
|--------|--------|-----------------|-------------|
| Recency | R | 0.25 | `exp(-λ · Δt)` — exponential decay since last access |
| Importance | I | 0.30 | Explicit importance set by system or user |
| Frequency | F | 0.15 | `min(access_count / max_count, 1.0)` — normalized access frequency |
| Preference | P | 0.20 | Boost if memory matches user's known preferences |
| Relevance | C | 0.10 | Semantic similarity to current query (during retrieval) |

**Decay function:** `R = exp(-λ · Δt_hours / 24)` where λ = decay_rate (default 0.1). A memory decays to ~37% of its recency after 10 days without access.

### 5.2 Scoring Service

The `ScoringService`:

- **Computes `I`** during memory creation by analyzing content keywords, sentiment, and user feedback
- **Updates `R`** and `F` on each memory access (read)
- **Recalculates `S`** periodically for consolidation decisions
- **Normalizes** all factors to [0, 1] before weighted summation

### 5.3 Ranking During Retrieval

When a query comes in, memories are ranked by:

1. **Fused score** = `0.6 × vector_similarity + 0.4 × importance_score`
2. **Deduplication** — merge highly similar memories (cosine > 0.92)
3. **Diversity** — ensure at least one memory from each relevant type
4. **Top-K** — return top N (default 10, configurable)

---

## 6. Retrieval System

### 6.1 Retrieval Flow

```
User Message
    │
    ▼
1. Embed query (384-dim vector)
    │
    ├─── 2a. Vector Search (Atlas $vectorSearch)
    │      ├── memories (k=20, pre-filter by user_id)
    │      └── knowledge (k=10, pre-filter if user scoped)
    │
    ├─── 2b. Keyword Search
    │      ├── memories ($text search on content + tags)
    │      └── knowledge ($text search on title + content)
    │
    ├─── 2c. Structured Queries
    │      ├── Active tasks (user_id + status=pending)
    │      ├── User preferences (memory_type=user_preference)
    │      └── Recent STM (user_id + memory_type=short_term, last 24h)
    │
    ▼
3. Fusion & Ranking
    ├── Merge results from all sources
    ├── Deduplicate (by memory_id, cosine threshold 0.92)
    ├── Apply fused ranking score
    └── Return top K
    │
    ▼
4. Context Builder assembles structured payload (see §7)
    │
    ▼
5. Send to LLM
```

### 6.2 Recency-Fused Vector Search

```python
async def hybrid_search(
    user_id: str,
    query: str,
    query_embedding: list[float],
    memory_types: list[str] | None = None,
    top_k: int = 10,
) -> list[ScoredMemory]:
    """Hybrid search: vector + keyword with recency boost."""
    
    # 1. Vector search via Atlas
    vector_results = await self._vector_search(
        query_embedding, user_id, memory_types, k=top_k * 2
    )
    
    # 2. Text search fallback
    text_results = await self._text_search(
        query, user_id, memory_types, k=top_k
    ) if not vector_results else []
    
    # 3. Fuse & deduplicate
    fused = self._fuse_results(vector_results + text_results)
    
    # 4. Apply recency boost
    for mem in fused:
        recency = self.scoring_service.compute_recency(mem.last_accessed)
        mem.score = 0.6 * mem.score + 0.2 * mem.importance_score + 0.2 * recency
    
    return sorted(fused, key=lambda x: x.score, reverse=True)[:top_k]
```

---

## 7. Context Builder

### 7.1 Output Structure

The context builder produces a serializable dict that gets injected into the LLM prompt:

```json
{
  "user_context": {
    "user_id": "uuid-xxx",
    "username": "tony",
    "profile": { "name": "Tony", "timezone": "America/New_York" },
    "preferences": {
      "language": "en",
      "response_style": "concise",
      "voice_speed": 1.0
    }
  },
  "conversation_context": {
    "current_conversation_id": "conv-xxx",
    "recent_messages": [
      {"role": "user", "content": "What's my schedule today?", "timestamp": "..."},
      {"role": "assistant", "content": "You have a meeting at 10am...", "timestamp": "..."}
    ],
    "conversation_summary": "User asking about daily schedule"
  },
  "memory_context": {
    "short_term": [
      {"content": "...", "importance": 0.8, "tags": ["recent"]}
    ],
    "long_term": [
      {"content": "User prefers morning meetings", "importance": 0.7}
    ],
    "episodic": [
      {"content": "User had a great meeting with Stark Industries on June 1", "importance": 0.9}
    ],
    "semantic": [
      {"content": "Stark Industries is a technology company...", "importance": 0.6}
    ],
    "user_preferences": [
      {"content": "User prefers dark mode and concise responses", "importance": 0.85}
    ]
  },
  "task_context": {
    "pending": [
      {"title": "Review Q3 report", "priority": 4, "due_at": "..."}
    ],
    "completed_recently": [
      {"title": "Email CFO", "completed_at": "..."}
    ]
  },
  "knowledge_context": {
    "relevant_documents": [
      {"title": "Project Blueprint v3", "summary": "...", "relevance": 0.92}
    ]
  }
}
```

### 7.2 Context Assembly Logic

```python
class ContextBuilder:
    def __init__(self, retrieval_service, memory_service, task_repo, knowledge_repo):
        ...

    async def build_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        max_tokens: int = 3000,
    ) -> ContextPayload:
        """Assemble full context payload for LLM injection."""
        
        # 1. Fetch user profile + preferences (always included, negligible size)
        user = await self._get_user_context(user_id)
        
        # 2. Fetch conversation history (last N messages)
        conversation = await self._get_conversation_context(conversation_id)
        
        # 3. Retrieve relevant memories (hybrid search)
        memories = await self.retrieval_service.hybrid_search(
            user_id, query, query_embedding, top_k=15
        )
        memory_groups = self._group_memories_by_type(memories)
        
        # 4. Fetch active tasks
        tasks = await self._get_task_context(user_id)
        
        # 5. Fetch knowledge context
        knowledge = await self.retrieval_service.search_knowledge(
            user_id, query, query_embedding, top_k=5
        )
        
        # 6. Assemble and trim to max_tokens
        payload = ContextPayload(
            user_context=user,
            conversation_context=conversation,
            memory_context=memory_groups,
            task_context=tasks,
            knowledge_context=knowledge,
        )
        
        # 7. Trim if above token budget
        return self._trim_to_budget(payload, max_tokens)
```

---

## 8. Memory Lifecycle

### 8.1 STM → LTM Consolidation

Run as a background task (every 5 minutes during active use, hourly otherwise):

1. **Query** all STM entries for user where `created_at > 1h` and `importance_score > 0.6`
2. **Deduplicate** — if similar content already exists as LTM, update its access_count and last_accessed instead of creating duplicate
3. **Summarize** — group highly related STM entries and create a consolidated LTM summary
4. **Promote** — create LTM entry with `source_id` referencing STM entries, mark STM as `consolidated = True`
5. **Cleanup** — delete STM entries past 24h TTL (handled by MongoDB TTL index)

### 8.2 Semantic Memory Extraction

Run on conversation completion or idle:

1. Extract factual statements from conversation
2. Deduplicate against existing semantic memories
3. Score by confidence
4. Store with `memory_type = "semantic"`

### 8.3 Forgetting / Archival

- **STM**: Automatic TTL deletion at 24h
- **LTM**: Archived when `importance_score < 0.1` for 90+ days without access
- **Agent logs**: TTL deletion at 30 days

---

## 9. CRUD API Layer

### 9.1 Repository Pattern

Base repository provides generic async CRUD:

```python
class BaseRepository(Generic[DocumentT]):
    def __init__(self, collection: Collection, model_class: type[DocumentT]):
        ...

    async def create(self, document: DocumentT) -> DocumentT
    async def get(self, doc_id: str) -> DocumentT | None
    async def update(self, doc_id: str, updates: dict) -> DocumentT | None
    async def delete(self, doc_id: str) -> bool
    async def find(self, filter: dict, sort: list, limit: int, skip: int) -> list[DocumentT]
    async def count(self, filter: dict) -> int
    async def aggregate(self, pipeline: list[dict]) -> list[dict]
```

### 9.2 REST API Endpoints (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/users` | Create user |
| GET | `/api/v1/users/{user_id}` | Get user profile + preferences |
| PUT | `/api/v1/users/{user_id}` | Update user |
| DELETE | `/api/v1/users/{user_id}` | Delete user (cascade) |
| POST | `/api/v1/conversations` | Create conversation |
| GET | `/api/v1/conversations/{conv_id}` | Get conversation with messages |
| POST | `/api/v1/conversations/{conv_id}/messages` | Add message |
| POST | `/api/v1/memories/search` | Hybrid search memories |
| POST | `/api/v1/memories` | Create memory |
| GET | `/api/v1/memories/{memory_id}` | Get memory |
| PUT | `/api/v1/memories/{memory_id}` | Update memory |
| DELETE | `/api/v1/memories/{memory_id}` | Delete memory |
| POST | `/api/v1/context` | Build context payload for query |
| POST | `/api/v1/tasks` | Create task |
| GET | `/api/v1/tasks?status=pending` | List tasks |
| PUT | `/api/v1/tasks/{task_id}` | Update task |
| POST | `/api/v1/knowledge` | Add knowledge document |
| GET | `/api/v1/knowledge/search` | Search knowledge |
| POST | `/api/v1/analytics/events` | Track analytics event |
| GET | `/api/v1/settings` | Get settings |
| PUT | `/api/v1/settings` | Update settings |

---

## 10. Embedding Service

### 10.1 Model-Agnostic Interface

```python
class EmbeddingService:
    def __init__(self, provider: str = "sentence_transformers", model_name: str = "all-MiniLM-L6-v2"):
        ...

    async def embed(self, text: str) -> list[float]
    async def embed_batch(self, texts: list[str]) -> list[list[float]]
    async def similarity(self, a: list[float], b: list[float]) -> float  # cosine
    
    @property
    def dimensions(self) -> int  # 384 for all-MiniLM-L6-v2
```

### 10.2 Supported Providers

1. **sentence-transformers** (local, default) — `all-MiniLM-L6-v2`, 384-dim, runs on CPU
2. **OpenAI** — `text-embedding-3-small`, 1536-dim (requires API key)
3. **HTTP** — Custom embedding service endpoint (for isolated deployment)

---

## 11. Production Deployment Guide

### 11.1 Prerequisites

- MongoDB Atlas cluster M10+ (required for `$vectorSearch`)
- Python 3.10+
- 2 GB RAM minimum for embedding model

### 11.2 Environment Variables

```bash
# MongoDB
MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/jarvis?retryWrites=true&w=majority"
MONGODB_DB_NAME="jarvis"

# Embedding
EMBEDDING_PROVIDER="sentence_transformers"
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Optional: OpenAI for embeddings
# OPENAI_API_KEY="sk-..."

# Memory Settings
MEMORY_STM_TTL_HOURS=24
MEMORY_CONSOLIDATION_INTERVAL_MINUTES=30
MEMORY_DECAY_RATE=0.1
MEMORY_DEFAULT_TOP_K=10
MEMORY_MAX_CONTEXT_TOKENS=3000

# Server
API_HOST="0.0.0.0"
API_PORT=8000
LOG_LEVEL="INFO"
```

### 11.3 Docker Deployment

```yaml
# docker-compose.yml
version: "3.9"
services:
  jarvis-memory:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - EMBEDDING_PROVIDER=sentence_transformers
    volumes:
      - embedding_cache:/root/.cache
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

volumes:
  embedding_cache:
```

### 11.4 Atlas Index Setup

Run `scripts/create_indexes.py` after deploying the Atlas cluster:

```bash
python scripts/create_indexes.py --uri "mongodb+srv://..."
```

This script creates:
1. All BSON indexes (unique, compound, TTL)
2. Atlas Search index for vector search on `memories`
3. Atlas Search index for vector search on `knowledge`

### 11.5 Performance Targets

| Operation | Target P99 | Strategy |
|-----------|-----------|----------|
| Memory write | < 50ms | Direct insert, no read |
| Memory read (by ID) | < 20ms | Indexed `_id` lookup |
| Memory search (vector) | < 200ms | Atlas $vectorSearch with HNSW |
| Memory search (hybrid) | < 300ms | Parallel vector + keyword fusion |
| Context build | < 500ms | Parallelized 5-way fetch |
| User profile load | < 20ms | Indexed `user_id` lookup |

### 11.6 Monitoring

- Atlas cluster metrics: CPU, RAM, disk IOPS, index sizes
- Application metrics: endpoint latency, memory count, vector search latency
- Log aggregation: agent_logs collection with structured JSON

---

## 12. Implementation Phases

| Phase | Components | Estimated Effort |
|-------|-----------|-----------------|
| **1. Foundation** | config, database, models, base repo | High |
| **2. Repositories** | All 7 repositories with tests | High |
| **3. Services** | memory, scoring, embedding, retrieval | High |
| **4. Context Builder** | Assembly, grouping, token trimming | Medium |
| **5. Consolidation** | STM→LTM, forgetting, summarization | Medium |
| **6. API Layer** | FastAPI routes, schemas, validation | Medium |
| **7. CLI + Scripts** | Index creation, seeding, management | Low |
| **8. Deployment** | Docker, docs, deployment guide | Low |

---

## 13. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single `memories` collection with `memory_type` discriminator | Enables cross-type queries, simplifies vector index; queries use `memory_type` as filter |
| Polymorphic Pydantic model with optional fields | Keeps schema flexible; memory types share 80% of fields; type-specific data goes in `metadata` |
| Atlas native `$vectorSearch` | Production-grade vector search with HNSW indexing; no extra infra; 384-dim on M10 is cost-effective |
| Repository pattern separating data from business logic | Testable (mock repos), swappable backends, clean boundaries |
| Fused ranking (vector + importance + recency) | Better than pure vector similarity; captures what's important to the user, not just semantically similar |
| Context builder with token budget | Prevents LLM context overflow; prioritizes highest-value content |
| STM TTL + consolidation | Automatic forgetting without manual cleanup; important STM gets promoted to LTM |
