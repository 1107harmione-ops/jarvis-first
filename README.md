# J.A.R.V.I.S. — AI Assistant Platform

Production-grade AI assistant system with MongoDB memory architecture and real-time voice capabilities.

## Repository Structure

```
├── docs/superpowers/specs/
│   ├── 2026-06-06-jarvis-memory-architecture-design.md   # Memory architecture spec
│   └── 2026-06-06-jarvis-voice-system-design.md          # Voice system spec
├── packages/
│   ├── jarvis-memory/     # MongoDB memory backend (Motor + FastAPI)
│   └── jarvis-voice/      # Voice server (Whisper STT + Piper TTS)
├── android/
│   └── jarvis-android/    # Android voice app (Kotlin)
└── README.md
```

## Packages

### jarvis-memory
MongoDB-based memory architecture with vector search, memory scoring, and context builder.

- **Models**: 9 collections (users, conversations, messages, memories, tasks, knowledge, agent_logs, analytics, settings)
- **Vector Search**: MongoDB Atlas `$vectorSearch` with 384-dim embeddings (all-MiniLM-L6-v2)
- **Memory Types**: Short-term, long-term, semantic, episodic, user preference, task, knowledge
- **Scoring**: Composite formula — recency (25%) + importance (30%) + frequency (15%) + preference (20%) + relevance (10%)
- **Context Builder**: Structured LLM context with token budget trimming
- **Consolidation**: STM → LTM promotion with deduplication

### jarvis-voice
Real-time voice system with streaming STT/TTS and wake word detection.

- **Streaming STT**: faster-whisper with partial results every 300ms
- **Streaming TTS**: Piper subprocess with `--output-raw` and interrupt support
- **Wake Word**: OpenWakeWord neural detection + energy fallback
- **State Machine**: 9-state lifecycle (IDLE → WAKE_PENDING → LISTENING → STT → THINKING → SPEAKING → INTERRUPTED)
- **Multilingual**: English + Hindi support

### jarvis-android
Android foreground service for always-listening voice interaction.

- **Streaming PCM**: AudioRecord → WebSocket → AudioTrack full-duplex pipeline
- **Interrupt Handling**: ~200ms voice energy detection during TTS playback
- **Bluetooth SCO**: Automated headset routing
- **Audio Focus**: Proper Android audio focus lifecycle
- **Offline Fallback**: Graceful degradation when server is unreachable

## Quick Start

### Memory Backend
```bash
cd packages/jarvis-memory
pip install -e .
python scripts/create_indexes.py --uri "mongodb+srv://..."
uvicorn jarvis_memory.api.app:app --port 8000
```

### Voice Server
```bash
cd packages/jarvis-voice
pip install -e .
# Download models first
uvicorn jarvis_voice.server:app --port 8002
```

### Android App
Open `android/jarvis-android/` in Android Studio and build.
