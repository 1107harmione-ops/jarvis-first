# J.A.R.V.I.S. — AI Assistant Platform

Production-grade AI assistant system with MongoDB memory architecture and real-time voice capabilities. Multi-agent system (coding, research, vision, memory, planner), WebSocket real-time communication, and Android voice interface.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Render (Docker)                     │
│  ┌────────────────────────────────────────────────┐  │
│  │           FastAPI Backend (8000)                │  │
│  │  ┌───────┐ ┌────────┐ ┌──────┐ ┌──────────┐  │  │
│  │  │ Auth  │ │ Chat   │ │Voice │ │ Research │  │  │
│  │  │  API  │ │  API   │ │  WS  │ │    API   │  │  │
│  │  └───┬───┘ └───┬────┘ └──┬───┘ └────┬─────┘  │  │
│  │      └──────────┼─────────┼──────────┘         │  │
│  │                 ▼         ▼                     │  │
│  │  ┌────────────────────────────────────────┐     │  │
│  │  │        Agent System (LangGraph)         │     │  │
│  │  │  Coding │ Research │ Vision │ Planner   │     │  │
│  │  └─────────────────────┬──────────────────┘     │  │
│  │                        ▼                        │  │
│  │  ┌────────────────────────────────────────┐     │  │
│  │  │     LLM Router (DeepSeek/OpenAI)       │     │  │
│  │  └────────────────────────────────────────┘     │  │
│  └──────────────────────┬─────────────────────────┘  │
│                         │                            │
│                         ▼                            │
│              ┌──────────────────┐                    │
│              │  MongoDB Atlas   │                    │
│              │   (Free M0)      │                    │
│              └──────────────────┘                    │
└──────────────────────────────────────────────────────┘
         │
         │ WebSocket (wss://)
         ▼
┌──────────────────┐
│  Android App     │
│  (Kotlin/OkHttp) │
└──────────────────┘
```

---

## Directory Structure

```
├── render.yaml                # Render Blueprint deployment config
├── backend/                   # ★ Main FastAPI backend
│   ├── main.py                # Entry point — uvicorn backend.main:app
│   ├── Dockerfile             # Multi-stage production Docker image
│   ├── start.sh               # Render/Docker entrypoint with dynamic port
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment variable template
│   ├── config/
│   │   └── settings.py        # Pydantic Settings (env-driven config)
│   ├── api/                   # REST API routes (auth, chat, voice, etc.)
│   ├── agents_v2/             # LangGraph multi-agent system
│   ├── agents/                # V1 agent system
│   ├── database/              # MongoDB models, schemas, connection
│   ├── llm/                   # LLM clients (DeepSeek, OpenAI, Minimax)
│   ├── memory/                # STM/LTM/vector memory system
│   ├── services/              # Business logic (research, voice, tasks)
│   ├── websocket/             # WebSocket handlers (chat, voice)
│   ├── utils/                 # Auth, logging, security utilities
│   └── tests/                 # Test suite (pytest)
├── android/
│   └── jarvis-android/        # Android voice app (Kotlin)
└── packages/
    ├── jarvis-memory/         # Standalone memory package (optional)
    └── jarvis-voice/          # Standalone voice package (optional)
```

---

## Deploy to Render

### Prerequisites

1. **Render account** — [Sign up](https://render.com) (free tier works)
2. **MongoDB Atlas** — [Free M0 cluster](https://www.mongodb.com/atlas)
3. **LLM API key** — At least one: [DeepSeek](https://platform.deepseek.com), [OpenAI](https://platform.openai.com), or [Minimax](https://www.minimax.io)
4. **GitHub repository** — Push this repo to GitHub (Render imports from GitHub)

### Step 1: MongoDB Atlas Setup

1. Go to [MongoDB Atlas](https://cloud.mongodb.com) → Create a **free M0 cluster**
2. Under **Security** → **Database Access**, create a database user (username + password)
3. Under **Security** → **Network Access**, add `0.0.0.0/0` (allow all — Render has dynamic IPs)
4. Click **Connect** → **Drivers** → copy your connection string:
   ```
   mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/jarvis?retryWrites=true&w=majority
   ```

### Step 2: Deploy via Render Blueprint (One-Click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/1107harmione-ops/jarvis-first)

**Or follow these manual steps:**

1. Push this repository to GitHub:
   ```bash
   git remote add origin https://github.com/1107harmione-ops/jarvis-first.git
   git push -u origin master
   ```

2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**

3. Connect your GitHub repo — Render auto-detects `render.yaml`

4. **Set sensitive environment variables** in the Render Dashboard after deployment:

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `MONGODB_URI` | ✅ | MongoDB Atlas connection string |
   | `SECRET_KEY` | ✅ | Random 64-char hex string |
   | `JWT_SECRET_KEY` | ✅ | Random 64-char hex string |
   | `DEEPSEEK_API_KEY` | ✅* | DeepSeek API key (recommended LLM) |
   | `CODEX_API_KEY` | | OpenAI API key |
   | `MINIMAX_API_KEY` | | Minimax API key |
   | `MIMO_API_KEY` | | Mimo API key |

   *At least one LLM API key is required.

5. Click **Apply** — Render builds and deploys automatically

6. Your backend is live at: **`https://<app-name>.onrender.com`**

### Step 3: Verify Deployment

```bash
# Health check
curl https://<your-app>.onrender.com/api/admin/health

# API root
curl https://<your-app>.onrender.com/

# OpenAPI docs (if ENVIRONMENT != production)
curl https://<your-app>.onrender.com/docs
```

### Step 4: Connect Android App

Update your Android app's WebSocket URL in `android/jarvis-android/app/src/main/java/com/jarvis/voice/model/VoiceConfig.kt`:

```kotlin
val serverUrl: String = "wss://<your-app>.onrender.com/ws/voice"
```

Then rebuild the APK.

---

## Environment Variables

| Category | Variable | Default | Description |
|----------|----------|---------|-------------|
| **Core** | `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| | `DEBUG` | `false` | Enable debug mode |
| | `SECRET_KEY` | *(required)* | Crypto signing key |
| **Server** | `PORT` | `8000` | HTTP port (Render sets this automatically) |
| | `WORKERS` | `2` | Uvicorn worker processes |
| | `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| | `RATE_LIMIT_PER_MINUTE` | `60` | Max API requests per IP per minute |
| **MongoDB** | `MONGODB_URI` | *(required)* | Atlas connection string (`mongodb+srv://...`) |
| | `MONGODB_DATABASE` | `jarvis` | Database name |
| **JWT** | `JWT_SECRET_KEY` | *(required)* | JWT signing key |
| | `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token TTL |
| | `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token TTL |
| **LLM** | `DEEPSEEK_API_KEY` | | DeepSeek API key |
| | `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model |
| | `CODEX_API_KEY` | | OpenAI API key |
| | `CODEX_MODEL` | `gpt-4o` | OpenAI model |
| | `MINIMAX_API_KEY` | | Minimax API key |
| | `MINIMAX_MODEL` | `minimax-m2.1` | Minimax model |
| **Memory** | `MEMORY_STM_TTL_HOURS` | `24` | Short-term memory expiry |
| | `MEMORY_LTM_IMPORTANCE_THRESHOLD` | `0.6` | Min importance for long-term storage |
| | `MEMORY_VECTOR_DIMENSION` | `384` | Embedding dimension |
| **Voice** | `PIPER_ENABLED` | `false` | Disable Piper TTS on Render (no binary) |
| | `VOICE_SESSION_TIMEOUT_SECONDS` | `300` | Voice session idle timeout |
| | `VOICE_WEBSOCKET_TIMEOUT` | `600` | WebSocket idle timeout |
| **Logging** | `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

For the full list, see [`backend/.env.example`](backend/.env.example).

---

## Local Development

### Backend

```bash
cd backend

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio  # Dev dependencies

# Configure environment
cp .env.example .env
# Edit .env — set MONGODB_URI, API keys, etc.

# Start development server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v
```

### With Docker

```bash
cd backend
docker build -t jarvis-backend .
docker run -p 8000:8000 --env-file .env jarvis-backend
```

### With Docker Compose (Full Stack)

```bash
cd backend
docker compose up -d
# Starts: API (8000), MongoDB (27017), Redis (6379), Mongo-Express (8081)
```

### Android App

Open `android/jarvis-android/` in Android Studio, update the WebSocket URL in `VoiceConfig.kt`, and build.

---

## API Overview

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| `POST` | `/api/auth/register` | Create account | No |
| `POST` | `/api/auth/login` | Login → JWT tokens | No |
| `POST` | `/api/auth/refresh` | Refresh access token | Refresh |
| `POST` | `/api/chat/send` | Send message | JWT |
| `GET` | `/api/chat/history` | Chat history | JWT |
| `POST` | `/api/voice/command` | Voice command | JWT |
| `POST` | `/api/memory/store` | Store memory | JWT |
| `GET` | `/api/memory/recall` | Recall memories | JWT |
| `POST` | `/api/tasks/create` | Create task | JWT |
| `POST` | `/api/agents/route` | Route to agent | JWT |
| `POST` | `/api/v2/research/search` | Quick research | JWT |
| `POST` | `/api/v2/research/deep` | Deep research | JWT |
| `GET` | `/api/admin/health` | Health check | No |
| `WS` | `/ws/chat?token=<jwt>` | Real-time chat | JWT |
| `WS` | `/ws/voice?token=<jwt>` | Voice conversation | JWT |

Full API docs at `/docs` (development only).

---

## License

MIT
