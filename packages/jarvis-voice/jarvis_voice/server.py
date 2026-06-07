"""FastAPI application for the JARVIS Voice system.

Provides:
- GET  /health     — Health check (STT/TTS/wake word status)
- WS   /ws/voice   — Bidirectional voice WebSocket
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from jarvis_voice.config import VoiceConfig
from jarvis_voice.memory.voice_memory import VoiceMemory
from jarvis_voice.session.manager import VoiceSessionManager
from jarvis_voice.stt.whisper_stt import WhisperSTT
from jarvis_voice.tts.piper_tts import PiperTTS
from jarvis_voice.wakeword.energy_detector import EnergyWakeWordDetector
from jarvis_voice.wakeword.openwakeword_detector import OpenWakeWordDetector

logger = logging.getLogger("jarvis_voice.server")


# ── Global state (set during lifespan) ──────────────────────────────

config: VoiceConfig | None = None
session_manager: VoiceSessionManager | None = None
stt_provider: WhisperSTT | None = None
tts_provider: PiperTTS | None = None
wakeword_detector: OpenWakeWordDetector | None = None
voice_memory: VoiceMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down voice components."""
    global config, session_manager, stt_provider, tts_provider, wakeword_detector, voice_memory

    logger.info("Initialising JARVIS Voice System...")

    # Load configuration from environment / .env
    config = VoiceConfig()

    # Initialise providers
    stt_provider = WhisperSTT(config)
    tts_provider = PiperTTS(config)

    # Wake word: try OpenWakeWord, fall back to energy detector
    if config.wake_word_provider == "openwakeword":
        wakeword_detector = OpenWakeWordDetector(
            wake_word=config.wake_word,
            sensitivity=config.wake_word_sensitivity,
        )
    else:
        wakeword_detector = EnergyWakeWordDetector(
            threshold=config.interrupt_energy_threshold,
            cooldown=config.wake_word_cooldown,
        )

    voice_memory = VoiceMemory(storage_dir="voice_data")
    session_manager = VoiceSessionManager(
        config=config,
        stt=stt_provider,
        tts=tts_provider,
        wakeword=wakeword_detector,
    )

    logger.info(
        "JARVIS Voice System ready (STT=%s, TTS=%s, wake=%s, port=%d)",
        config.whisper_model,
        config.tts_provider,
        config.wake_word,
        config.port,
    )

    yield

    # Cleanup
    logger.info("Shutting down JARVIS Voice System...")

    # Destroy all active sessions
    if session_manager:
        for session in list(session_manager.active_sessions):
            await session_manager.destroy_session(session)


app = FastAPI(
    title="JARVIS Voice System",
    version="0.1.0",
    lifespan=lifespan,
)


# ── REST endpoints ───────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check returning status of all voice components."""
    global config, session_manager, stt_provider, tts_provider, wakeword_detector

    stt_loaded = stt_provider is not None and stt_provider.model is not None
    tts_loaded = tts_provider is not None
    ww_loaded = wakeword_detector is not None
    active_sessions = session_manager.active_count if session_manager else 0

    return {
        "status": "online",
        "stt_model": config.whisper_model if config else "unknown",
        "stt_loaded": stt_loaded,
        "tts_voices": list(tts_provider.voices.keys()) if tts_provider else [],
        "wake_word": config.wake_word if config else "unknown",
        "wake_word_provider": config.wake_word_provider if config else "unknown",
        "languages": config.supported_languages if config else [],
        "active_sessions": active_sessions,
        "version": "0.1.0",
    }


# ── WebSocket endpoint ───────────────────────────────────────────────

@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    """Main bidirectional voice WebSocket.

    Accepts a connection, then loops on:
    - Binary messages: raw PCM16 audio chunks → routed to session manager
    - Text messages: JSON control messages → routed to session manager

    On disconnect, the session is cleaned up.
    """
    global session_manager

    if session_manager is None:
        await websocket.close(code=1011, reason="Server not initialised")
        return

    await websocket.accept()

    # Create a new voice session
    session = await session_manager.create_session(
        websocket=websocket,
        user_id="default",
    )

    logger.info(
        "WebSocket connected: session=%s from %s",
        session.session_id, websocket.client,
    )

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                # Binary PCM16 audio chunk
                audio_chunk = message["bytes"]
                if audio_chunk:
                    await session_manager.handle_audio(session, audio_chunk)

            elif "text" in message:
                # JSON control message
                text = message["text"]
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid JSON: %s", exc)
                    continue

                await session_manager.handle_control(session, data)

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: session=%s", session.session_id,
        )
    except Exception as exc:
        logger.exception(
            "WebSocket error (session=%s): %s", session.session_id, exc,
        )
    finally:
        await session_manager.destroy_session(session)


# ── Main entry point ─────────────────────────────────────────────────

def main() -> None:
    """Run the server via uvicorn."""
    import uvicorn

    cfg = VoiceConfig()
    uvicorn.run(
        "jarvis_voice.server:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
        ws_max_size=2 ** 21,  # 2 MB max WebSocket message
    )


if __name__ == "__main__":
    main()
