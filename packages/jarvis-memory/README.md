# jarvis-memory

MongoDB-backed memory architecture for AI assistants.

Built with **Motor** (async MongoDB driver), **FastAPI** (REST layer), and
**sentence-transformers** (embeddings).

## Features

- **9 collections**: users, conversations, messages, memories, tasks, knowledge, agent_logs, analytics, settings
- **Polymorphic memory model**: short_term, long_term, semantic, episodic, user_preference, task, knowledge
- **Hybrid search**: Vector search (Atlas `$vectorSearch`) + keyword search with recency-fused ranking
- **Embedding service**: `all-MiniLM-L6-v2` (384-dim) with model-agnostic interface
- **Scoring**: Composite importance formula with recency, frequency, importance, preference, relevance factors
- **Context builder**: Structured LLM context payload with token budget trimming
- **Consolidation**: STM → LTM promotion with deduplication and summarization
- **REST API**: FastAPI with full CRUD endpoints for all models
- **Async throughout**: Motor async driver, asyncio-based services

## Quick Start

```bash
# Install
pip install jarvis-memory

# Set environment variables
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="jarvis"

# Create indexes
python scripts/create_indexes.py

# Run API server
uvicorn jarvis_memory.api.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
jarvis-memory/
├── jarvis_memory/            # Main package
│   ├── config.py             # Configuration (pydantic-settings)
│   ├── database.py           # Motor client singleton
│   ├── models/               # Pydantic v2 data models
│   ├── repositories/         # Async CRUD (repository pattern)
│   ├── services/             # Business logic layer
│   └── api/                  # FastAPI REST endpoints
├── tests/                    # pytest test suite
├── scripts/                  # Index creation, seed data
├── deploy/                   # Docker, Atlas index definitions
└── pyproject.toml            # Build configuration
```

## Configuration

All settings via environment variables (see `jarvis_memory/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB_NAME` | `jarvis` | Database name |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | Embedding provider |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name |
| `MEMORY_STM_TTL_HOURS` | `24` | STM TTL in hours |
| `MEMORY_DECAY_RATE` | `0.1` | Exponential decay rate |
| `MEMORY_DEFAULT_TOP_K` | `10` | Top-K retrieval default |
| `MEMORY_MAX_CONTEXT_TOKENS` | `3000` | Context token budget |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/users` | Create user |
| GET | `/api/v1/users/{user_id}` | Get user |
| PUT | `/api/v1/users/{user_id}` | Update user |
| POST | `/api/v1/conversations` | Create conversation |
| GET | `/api/v1/conversations/{conv_id}` | Get conversation |
| POST | `/api/v1/conversations/{conv_id}/messages` | Add message |
| GET | `/api/v1/conversations/{conv_id}/messages` | Get messages |
| POST | `/api/v1/memories` | Create memory |
| GET | `/api/v1/memories/{memory_id}` | Get memory |
| PUT | `/api/v1/memories/{memory_id}` | Update memory |
| DELETE | `/api/v1/memories/{memory_id}` | Delete memory |
| POST | `/api/v1/memories/search` | Hybrid search |
| POST | `/api/v1/context` | Build context payload |
| GET | `/api/v1/tasks` | List tasks |
| POST | `/api/v1/tasks` | Create task |
| PUT | `/api/v1/tasks/{task_id}` | Update task |
| POST | `/api/v1/knowledge` | Add knowledge |
| GET | `/api/v1/knowledge/search` | Search knowledge |
| GET | `/health` | Health check |

## License

MIT
