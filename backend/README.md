# Jarvis Voice Productivity Assistant

An intelligent voice-controlled productivity assistant built with **FastAPI**, **SQLite**, **Redis**, and **RQ**. Manage tasks, notes, reminders, and memory — all through natural voice commands.

## Features

### 🗂️ Task Management
- Create, list, complete, update, delete, and search tasks
- Prioritize with levels: low, medium, high, urgent
- Filter by status (pending/completed) and priority

### 📝 Notes
- Rich note-taking with titles, content, categories, and tags
- Categories: learning, project, personal, ideas, research
- Full-text search across titles and content

### ⏰ Reminders (Redis + RQ)
- Schedule one-time or recurring reminders (daily, weekly)
- Background worker fires reminders automatically
- Built-in async checker runs every 60s as fallback
- Repeat reminders reschedule themselves

### 🧠 Memory
- Store facts about the user with automatic importance scoring
- Smart skip patterns prevent saving greetings/garbage
- Recall memories by keyword, forget by query

### 🔍 Unified Search (FTS5)
- Full-text search across tasks, notes, and memory simultaneously
- Powered by SQLite FTS5 with Porter stemmer
- Auto-synced via database triggers

### 🎤 Voice Commands
- Natural language intent routing: exact → regex → fuzzy matching
- Supported intents: task/note CRUD, reminders, memory, global search
- Speech-to-text via **Vosk** (offline)
- Text-to-speech via **Edge TTS** (natural voices)

## Quick Start

### Prerequisites
- Python 3.12+
- Redis (optional, for RQ scheduler)

### Local Development

```bash
# Clone and enter the project
git clone https://github.com/1107harmione-ops/jarvis-first.git
cd jarvis-first/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start the server
./start.sh
```

The API will be available at **http://localhost:8000**.

### Docker (with Redis)

```bash
docker compose up --build
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `Jarvis` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `DEBUG` | `true` | Enable debug mode |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DATABASE_URL` | `sqlite+aiosqlite:///./jarvis.db` | Database connection string |
| `MONGODB_URL` | *(empty)* | MongoDB URL (optional, for scalable storage) |
| `MONGODB_DB_NAME` | `jarvis` | MongoDB database name |
| `GROQ_API_KEY` | *(empty)* | Groq API key for LLM inference ([get free key](https://console.groq.com)) |
| `GROQ_MODEL` | `mixtral-8x7b-32768` | Groq model name |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |
| `GROQ_MAX_TOKENS` | `4096` | Max tokens for Groq responses |
| `GROQ_TEMPERATURE` | `0.7` | LLM temperature (0.0–1.0) |
| `OPENCODE_ZEN_API_KEY` | *(empty)* | OpenCode Zen API key for AI code assistance |
| `OPENCODE_ZEN_MODEL` | `gpt-4o-mini` | OpenCode Zen model name |
| `OPENCODE_ZEN_BASE_URL` | *(empty)* | OpenCode Zen base URL (optional) |
| `OPENCODE_ZEN_MAX_TOKENS` | `2048` | Max tokens for code responses |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database index |
| `REDIS_PASSWORD` | *(empty)* | Redis password (optional) |
| `REDIS_URL` | *(computed)* | Full Redis URL (overrides host/port/db) |
| `RQ_QUEUE_NAME` | `jarvis` | RQ queue name |
| `RQ_DEFAULT_TIMEOUT` | `300` | Default RQ job timeout (seconds) |
| `REMINDER_CHECK_INTERVAL` | `60` | Background checker interval (seconds) |
| `VOSK_MODEL_PATH` | `./models/vosk-model-small-en-us-0.15` | Path to Vosk STT model |
| `VOSK_SAMPLE_RATE` | `16000` | Audio sample rate for STT |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge TTS voice name |
| `VAD_MODE` | `1` | VAD aggressiveness (0–3) |
| `VAD_FRAME_MS` | `30` | VAD frame size in milliseconds |
| `VAD_ENERGY_THRESHOLD` | `300.0` | VAD energy threshold |
| `WAKE_WORD_ENABLED` | `false` | Enable wake word detection |
| `WAKE_WORD_SENSITIVITY` | `0.5` | Wake word sensitivity |
| `WAKE_WORD_KEYWORDS` | `jarvis,hey jarvis` | Comma-separated wake words |
| `WAKE_WORD_ENERGY_THRESHOLD` | `500.0` | Wake word energy threshold |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json or console) |
| `DATA_DIR` | `./data` | Data storage directory |

## API Endpoints

### Tasks (`/api/tasks`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/tasks` | Create a task |
| GET | `/api/tasks` | List tasks (filter by `status`, `priority`, `category`) |
| GET | `/api/tasks/search` | Search tasks by query (`?q=...`) |
| GET | `/api/tasks/{id}` | Get a task |
| PATCH | `/api/tasks/{id}` | Update a task |
| DELETE | `/api/tasks/{id}` | Delete a task |
| PATCH | `/api/tasks/{id}/complete` | Mark task as completed |

### Notes (`/api/notes`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/notes` | Create a note |
| GET | `/api/notes` | List notes (filter by `category`, `priority`) |
| GET | `/api/notes/search` | Search notes by query (`?q=...`) |
| GET | `/api/notes/{id}` | Get a note |
| PATCH | `/api/notes/{id}` | Update a note |
| DELETE | `/api/notes/{id}` | Delete a note |

### Reminders (`/api/reminders`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/reminders` | Create a reminder |
| GET | `/api/reminders` | List reminders (filter by `status`) |
| GET | `/api/reminders/{id}` | Get a reminder |
| PATCH | `/api/reminders/{id}` | Update a reminder |
| DELETE | `/api/reminders/{id}` | Delete a reminder |
| POST | `/api/reminders/check-due` | Check and fire due reminders |

### Memory (`/api/memory`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory` | Store a memory/fact |
| GET | `/api/memory` | List memories (filter by `category`) |
| GET | `/api/memory/search` | Search memories by query (`?q=...`) |
| DELETE | `/api/memory/{id}` | Forget a memory |

### Voice (`/api/voice`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/voice/command` | Process a voice command (`{"text": "..."}`) |

Example voice commands:
```
"create a task to learn FastAPI"
"show my tasks"
"complete my task review code"
"create a note about project ideas"
"search my notes about Python"
"remind me to call doctor tomorrow"
"remember that I love Python"
"what do you know about Python"
"search for python"
```

### Search (`/api/search`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/search` | Global FTS5 search (`?q=...&type=task`) |

## Project Structure

```
backend/
├── app/
│   ├── api/            # REST API route handlers
│   ├── core/           # Config, logging, exceptions, dependencies
│   ├── database/       # SQLAlchemy connection, models, migrations, FTS
│   ├── tasks/          # Task service + schemas
│   ├── notes/          # Note service + schemas
│   ├── reminders/      # Reminder service + Redis/RQ worker
│   ├── memory/         # Memory service + schemas
│   ├── search/         # FTS5 search service
│   └── voice/          # Intent router, STT, TTS, VAD, wakeword
├── tests/              # 64 pytest tests
├── Dockerfile          # Container build
├── docker-compose.yml  # App + Redis + Worker services
├── Makefile            # Common commands
├── requirements.txt    # Python dependencies
├── start.sh            # Development start script
└── .env.example        # Environment template
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v --asyncio-mode=auto

# Run specific test file
python -m pytest tests/test_notes.py -v --asyncio-mode=auto
```

## Developer Commands

```bash
make install      # Install dependencies
make run          # Start development server
make lint         # Check code style (ruff)
make clean        # Remove cache files and test DB
```
