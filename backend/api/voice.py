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
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get voice interaction history."""
    from backend.database.mongodb import mongodb
    from backend.database.schemas import serialize_doc

    cursor = mongodb.agent_logs.find(
        {"user_id": user["id"], "agent_name": "voice_service", "action": "voice_interaction"},
        sort=[("created_at", -1)],
        limit=limit,
    )
    logs = await cursor.to_list(length=limit)
    history = []
    for log in logs:
        s = serialize_doc(log)
        history.append({
            "id": s.get("id"),
            "transcript": s.get("input_summary", ""),
            "response": s.get("output_summary", ""),
            "language": s.get("metadata", {}).get("language", "en"),
            "duration_ms": s.get("duration_ms", 0),
            "created_at": s.get("created_at", ""),
        })
    return {"success": True, "data": {"items": history, "total": len(history)}}
