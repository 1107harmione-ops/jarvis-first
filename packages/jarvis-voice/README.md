# JARVIS Voice

Production-grade voice system backend for the JARVIS AI assistant. Built with **FastAPI** WebSocket streaming,
**faster-whisper** STT, **Piper** TTS, and **OpenWakeWord** detection.

## Features

- **Full-duplex voice** — Bidirectional PCM16 audio over a single WebSocket connection
- **Streaming STT** — Partial results every 300ms via faster-whisper (int8 quantized)
- **Streaming TTS** — Low-latency Piper subprocess with `--output-raw` piping
- **Wake word** — OpenWakeWord neural detection with energy-based fallback
- **Interrupt handling** — User voice activity during TTS triggers immediate interrupt
- **State machine** — Full session lifecycle (IDLE → LISTENING → STT → THINKING → SPEAKING)
- **Multilingual** — English and Hindi supported out of the box
- **Voice memory** — Command/session logging via local file store

## Quick Start

```bash
# Install
pip install -e .

# Run the server
uvicorn jarvis_voice.server:app --host 0.0.0.0 --port 8002
```

## WebSocket Protocol

Connect to `ws://<host>:8002/ws/voice`

### Client → Server

| Type | Payload | Description |
|------|---------|-------------|
| Binary | PCM16 mono audio | Raw audio chunks at 16kHz |
| `audio_start` | `{"type":"audio_start","format":"pcm16","sample_rate":16000}` | Signal start of utterance |
| `audio_end` | `{"type":"audio_end"}` | Signal end of utterance |
| `interrupt` | `{"type":"interrupt"}` | User requests immediate stop |
| `config` | `{"type":"config","language":"hi","voice_speed":1.2}` | Update session config |

### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| Binary | PCM16 mono audio | Raw TTS audio chunks at 22050Hz |
| `state_change` | `{"type":"state_change","state":"listening"}` | State transition |
| `partial` | `{"type":"partial","text":"hello...","confidence":0.85}` | Partial STT result |
| `transcript` | `{"type":"transcript","text":"hello world","confidence":0.92,"language":"en"}` | Final transcription |
| `tts_start` | `{"type":"tts_start"}` | TTS playback begins |
| `tts_end` | `{"type":"tts_end"}` | TTS playback ends |
| `thinking` | `{"type":"thinking"}` | LLM is generating response |
| `error` | `{"type":"error","message":"..."}` | Error notification |

## Configuration

Configuration uses `pydantic-settings` and reads from environment variables (prefixed with `JARVIS_`) or `.env` file.

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_SAMPLE_RATE` | 16000 | Audio sample rate |
| `JARVIS_WHISPER_MODEL` | base | Whisper model size |
| `JARVIS_WHISPER_DEVICE` | cpu | Device for inference |
| `JARVIS_PIPER_EXECUTABLE` | piper | Piper binary path |
| `JARVIS_WAKE_WORD` | hey jarvis | Wake word phrase |
| `JARVIS_SILENCE_TIMEOUT_SEC` | 1.5 | Silence triggers STT |
| `JARVIS_DEFAULT_LANGUAGE` | en | Default language |
| `JARVIS_SUPPORTED_LANGUAGES` | en,hi | Comma-separated |

## Project Structure

```
jarvis-voice/
├── jarvis_voice/
│   ├── config.py              # VoiceConfig (pydantic-settings)
│   ├── models.py              # Pydantic models + message types
│   ├── server.py              # FastAPI app + WebSocket endpoint
│   ├── session/               # Voice session state machine
│   ├── stt/                   # Speech-to-text (faster-whisper)
│   ├── tts/                   # Text-to-speech (Piper)
│   ├── wakeword/              # Wake word detection
│   ├── pipeline/              # Audio processing (VAD, noise gate)
│   └── memory/                # Voice history logging
├── tests/
├── deploy/
└── pyproject.toml
```

## Development

```bash
pip install -e ".[dev]"
pytest --asyncio-mode=auto -v
```

## License

MIT
