"""
Voice API — voice session management and audio processing endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.database.models import VoiceSessionCreate, VoiceSessionResponse
from backend.services.voice_service import voice_service
from backend.utils.auth import get_current_user
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/voice", tags=["Voice"])


@router.post("/sessions")
async def create_session(
    body: VoiceSessionCreate,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a new voice session."""
    session = voice_service.create_session(
        user_id=user["id"],
        language=body.language,
    )
    return {
        "success": True,
        "data": VoiceSessionResponse(
            session_id=session.session_id,
            state=session.state.value,
            language=session.language,
            started_at=session.created_at.isoformat(),
            expires_at=session.expires_at.isoformat(),
        ),
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get voice session status."""
    session = voice_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {
        "success": True,
        "data": session.to_dict(),
    }


@router.delete("/sessions/{session_id}")
async def end_session(
    session_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """End a voice session."""
    if voice_service.end_session(session_id):
        return {"success": True, "message": "Session ended"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/sessions/{session_id}/audio")
async def process_audio(
    session_id: str,
    file: UploadFile = File(...),
    language: str = Form("en"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Process audio from a voice session (STT → respond → TTS).

    Accepts audio files in webm, wav, mp3, ogg formats.
    Returns transcript, response text, and synthesized audio.
    """
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio data")

    result = await voice_service.process_audio(
        session_id=session_id,
        audio_data=audio_data,
        language=language,
    )
    return {
        "success": True,
        "data": result,
    }


@router.post("/sessions/{session_id}/interrupt")
async def interrupt_session(
    session_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Interrupt the current voice session (stop TTS, re-listen)."""
    if voice_service.handle_interrupt(session_id):
        return {"success": True, "message": "Interrupted"}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("en"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Transcribe audio to text without creating a session."""
    audio_data = await file.read()
    text = await voice_service.transcribe(audio_data, language=language)
    return {"success": True, "data": {"text": text, "language": language}}


@router.post("/synthesize")
async def synthesize_speech(
    text: str = Form(..., min_length=1, max_length=2000),
    voice: str = Form("alloy"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Synthesize text to speech audio."""
    audio_bytes = await voice_service.synthesize(text, voice=voice)
    from fastapi.responses import Response
    return Response(
        content=audio_bytes,
        media_type="audio/opus",
        headers={"Content-Disposition": "inline; filename=speech.opus"},
    )


@router.get("/history")
async def get_voice_history(
    limit: int = 20,
    offset: int = 0,
    language: str | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get voice interaction history with pagination."""
    from backend.memory.voice_memory import voice_memory_service

    items = await voice_memory_service.get_history(
        user_id=user["id"],
        limit=limit,
        offset=offset,
        language=language,
    )
    total = await voice_memory_service.get_history_count(
        user_id=user["id"],
        language=language,
    )
    return {
        "success": True,
        "data": {"items": items, "total": total},
    }


@router.get("/config")
async def get_voice_config(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get user's voice preferences."""
    from backend.memory.voice_memory import voice_memory_service

    prefs = await voice_memory_service.get_preferences(user["id"])
    return {"success": True, "data": prefs}


@router.put("/config")
async def update_voice_config(
    config: dict[str, Any],
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Update user's voice preferences."""
    from backend.memory.voice_memory import voice_memory_service

    await voice_memory_service.store_preferences(user["id"], config)
    return {"success": True, "message": "Preferences updated"}


@router.get("/metrics")
async def get_voice_metrics(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get aggregate voice usage statistics."""
    from backend.memory.voice_memory import voice_memory_service
    from backend.services.offline_handler import offline_handler

    stats = await voice_memory_service.get_voice_stats(user["id"])
    stats["offline_status"] = offline_handler.status_summary
    stats["is_online"] = offline_handler.is_fully_online
    return {"success": True, "data": stats}


@router.get("/commands/frequent")
async def get_frequent_commands(
    limit: int = 20,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get user's most frequently used voice commands."""
    from backend.memory.voice_memory import voice_memory_service

    commands = await voice_memory_service.get_frequent_commands(
        user_id=user["id"],
        limit=limit,
    )
    return {"success": True, "data": commands}


@router.post("/offline/queue")
async def queue_offline_command(
    transcript: str,
    language: str = "en",
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Queue a voice command for processing when back online."""
    from backend.services.offline_handler import offline_handler

    queue_id = await offline_handler.queue_command(
        user_id=user["id"],
        transcript=transcript,
        language=language,
    )
    # Provide immediate offline response
    response = await offline_handler.get_offline_response(transcript, language)
    return {
        "success": True,
        "data": {
            "queue_id": queue_id,
            "response": response,
            "queued": True,
        },
    }


@router.get("/offline/queue")
async def get_offline_queue(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get queued offline commands."""
    from backend.services.offline_handler import offline_handler

    commands = await offline_handler.get_queued_commands(user["id"])
    return {"success": True, "data": commands}


@router.post("/offline/process")
async def process_offline_queue(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Process queued offline commands (triggered when back online)."""
    from backend.services.offline_handler import offline_handler

    processed = await offline_handler.process_queued_commands(user["id"])
    return {
        "success": True,
        "data": {"processed": processed},
    }


@router.get("/voices")
async def get_available_voices(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get available Piper TTS voices."""
    from backend.services.piper_tts import piper_service

    voices = await piper_service.get_available_voices()
    return {"success": True, "data": voices}


@router.post("/synthesize/stream")
async def synthesize_speech_stream(
    text: str = Form(..., min_length=1, max_length=2000),
    language: str = Form("en"),
    voice: str | None = Form(None),
    speed: float = Form(1.0),
    user: dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Synthesize text to speech with streaming audio response."""
    from backend.services.piper_tts import piper_service
    from fastapi.responses import StreamingResponse

    async def audio_stream():
        async for chunk in piper_service.stream_synthesize(
            text=text,
            language=language,
            voice=voice,
            speed=speed,
        ):
            yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/L16;rate=22050;channels=1",
        headers={
            "Content-Disposition": "inline; filename=speech.raw",
            "X-Audio-Sample-Rate": "22050",
            "X-Audio-Channels": "1",
            "X-Audio-Format": "pcm16",
        },
    )
