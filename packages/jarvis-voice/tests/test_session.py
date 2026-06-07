"""Tests for the voice session state machine."""

from __future__ import annotations

import asyncio

import pytest

from jarvis_voice.session.manager import VoiceSessionManager
from jarvis_voice.session.state import VoiceState
from tests.conftest import make_audio_chunk


class TestVoiceSessionManager:
    """Test suite for VoiceSessionManager state machine."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, mock_websocket):
        """Test session creation starts in IDLE state."""
        session = await session_manager.create_session(mock_websocket)
        assert session.state == VoiceState.IDLE
        assert session.session_id is not None
        assert session.user_id == "default"
        assert session.websocket == mock_websocket

    @pytest.mark.asyncio
    async def test_destroy_session(self, session_manager, active_session):
        """Test destroying a session cleans up and sends OFFLINE state."""
        await session_manager.destroy_session(active_session)
        assert active_session.state == VoiceState.OFFLINE
        assert active_session.session_id not in session_manager._sessions

    @pytest.mark.asyncio
    async def test_idle_to_listening_on_wake(self, session_manager, mock_websocket, mock_detecting_wakeword):
        """Test IDLE → LISTENING when wake word is detected."""
        manager = VoiceSessionManager(
            config=session_manager.config,
            stt=session_manager.stt,
            tts=session_manager.tts,
            wakeword=mock_detecting_wakeword,
        )
        session = await manager.create_session(mock_websocket)
        assert session.state == VoiceState.IDLE

        # Send audio chunk that triggers wake word
        chunk = make_audio_chunk(amplitude=0.5)
        await manager.handle_audio(session, chunk)

        assert session.state == VoiceState.LISTENING

    @pytest.mark.asyncio
    async def test_idle_stays_idle_without_wake(self, session_manager, active_session):
        """Test IDLE remains IDLE when no wake word is detected."""
        chunk = make_audio_chunk(amplitude=0.0)  # Silence
        await session_manager.handle_audio(active_session, chunk)
        assert active_session.state == VoiceState.IDLE

    @pytest.mark.asyncio
    async def test_listening_accumulates_audio(self, session_manager, active_session):
        """Test that audio chunks are buffered in LISTENING state."""
        active_session.state = VoiceState.LISTENING
        chunk = make_audio_chunk(amplitude=0.5)
        await session_manager.handle_audio(active_session, chunk)
        assert active_session.audio_buffer.qsize() > 0

    @pytest.mark.asyncio
    async def test_interrupt_during_speaking(self, session_manager, active_session):
        """Test SPEAKING → INTERRUPTED when user speaks."""
        active_session.state = VoiceState.SPEAKING
        active_session.interrupt_event.clear()

        # Send a high-energy audio chunk
        loud_chunk = make_audio_chunk(amplitude=0.8)
        await session_manager.handle_audio(active_session, loud_chunk)

        # Should have transitioned through INTERRUPTED to LISTENING
        # _handle_interrupt transitions INTERRUPTED → LISTENING immediately
        assert active_session.state == VoiceState.LISTENING
        assert active_session.total_interrupts == 1

    @pytest.mark.asyncio
    async def test_no_interrupt_during_speaking_with_silence(self, session_manager, active_session):
        """Test SPEAKING stays SPEAKING with silence."""
        active_session.state = VoiceState.SPEAKING
        active_session.interrupt_event.clear()

        silent_chunk = make_audio_chunk(amplitude=0.0)
        await session_manager.handle_audio(active_session, silent_chunk)

        assert active_session.state == VoiceState.SPEAKING
        assert active_session.total_interrupts == 0

    @pytest.mark.asyncio
    async def test_audio_start_control(self, session_manager, active_session):
        """Test audio_start control transitions to LISTENING."""
        await session_manager.handle_control(active_session, {"type": "audio_start"})
        assert active_session.state == VoiceState.LISTENING

    @pytest.mark.asyncio
    async def test_audio_end_control_triggers_stt(self, session_manager, active_session):
        """Test audio_end control triggers STT from LISTENING."""
        active_session.state = VoiceState.LISTENING
        # Put some audio in the buffer
        chunk = make_audio_chunk(amplitude=0.5)
        await active_session.audio_buffer.put(chunk)

        await session_manager.handle_control(active_session, {"type": "audio_end"})

        # Should transition to STT_PROCESSING and create STT task
        assert active_session.state in (VoiceState.STT_PROCESSING, VoiceState.THINKING)
        # Wait for STT to complete
        if active_session.stt_task:
            await asyncio.wait_for(active_session.stt_task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_interrupt_control(self, session_manager, active_session):
        """Test interrupt control message."""
        active_session.state = VoiceState.SPEAKING
        await session_manager.handle_control(active_session, {"type": "interrupt"})
        assert active_session.state == VoiceState.LISTENING

    @pytest.mark.asyncio
    async def test_config_control_language(self, session_manager, active_session):
        """Test config control updates session language."""
        await session_manager.handle_control(active_session, {
            "type": "config",
            "language": "hi",
        })
        assert active_session.language == "hi"

    @pytest.mark.asyncio
    async def test_config_control_unsupported_language(self, session_manager, active_session):
        """Test unsupported language is rejected."""
        await session_manager.handle_control(active_session, {
            "type": "config",
            "language": "fr",
        })
        # Should remain at default
        assert active_session.language == "en"

    @pytest.mark.asyncio
    async def test_silence_timeout_triggers_stt(self, session_manager, active_session):
        """Test that prolonged silence in LISTENING triggers STT."""
        active_session.state = VoiceState.LISTENING
        active_session.last_activity = 0  # Force old timestamp
        active_session.audio_buffer = asyncio.Queue()
        # Put some audio in the buffer so there's something to transcribe
        chunk = make_audio_chunk(amplitude=0.5, duration_ms=500)
        await active_session.audio_buffer.put(chunk)

        await session_manager._check_timeouts()
        assert active_session.state in (VoiceState.STT_PROCESSING, VoiceState.THINKING, VoiceState.IDLE)

    @pytest.mark.asyncio
    async def test_state_transition_sends_event(self, session_manager, mock_websocket, mock_detecting_wakeword):
        """Test that state transitions send WebSocket events."""
        manager = VoiceSessionManager(
            config=session_manager.config,
            stt=session_manager.stt,
            tts=session_manager.tts,
            wakeword=mock_detecting_wakeword,
        )
        session = await manager.create_session(mock_websocket)

        await manager._transition(session, VoiceState.LISTENING)
        assert any("state_change" in t for t in mock_websocket.sent_texts)

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, session_manager):
        """Test multiple sessions can coexist."""
        ws1 = mock_websocket = None
        from tests.conftest import MockWebSocket
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        session1 = await session_manager.create_session(ws1, user_id="user1")
        session2 = await session_manager.create_session(ws2, user_id="user2")

        assert session_manager.active_count == 2
        assert session1.session_id != session2.session_id

        await session_manager.destroy_session(session1)
        assert session_manager.active_count == 1

        await session_manager.destroy_session(session2)
        assert session_manager.active_count == 0

    @pytest.mark.asyncio
    async def test_tts_pipeline_integration(self, session_manager, active_session):
        """Test full STT → THINKING → SPEAKING pipeline via _run_stt."""
        active_session.state = VoiceState.STT_PROCESSING
        # Put audio in the buffer
        chunk = make_audio_chunk(amplitude=0.5, duration_ms=500)
        await active_session.audio_buffer.put(chunk)

        async def partial_cb(text, conf):
            pass

        # Run STT
        await session_manager._run_stt(active_session, partial_cb)

        # Wait for TTS task to complete
        if active_session.tts_task and not active_session.tts_task.done():
            await asyncio.wait_for(active_session.tts_task, timeout=5.0)

        # Should have completed STT → THINKING → SPEAKING → IDLE cycle
        assert active_session.final_text != ""
        # TTS should have produced chunks (check via the mock TTS provider)
        assert session_manager.tts.call_count > 0
        assert active_session.total_commands == 1

    @pytest.mark.asyncio
    async def test_websocket_event_on_state_change(self, session_manager, mock_websocket, mock_detecting_wakeword):
        """Test that WebSocket receives state change events."""
        manager = VoiceSessionManager(
            config=session_manager.config,
            stt=session_manager.stt,
            tts=session_manager.tts,
            wakeword=mock_detecting_wakeword,
        )
        session = await manager.create_session(mock_websocket)

        # Trigger a few transitions
        await manager._transition(session, VoiceState.LISTENING)
        await manager._transition(session, VoiceState.STT_PROCESSING)
        await manager._transition(session, VoiceState.THINKING)
        await manager._transition(session, VoiceState.SPEAKING)

        # Find all state_change events
        state_events = [
            t for t in mock_websocket.sent_texts
            if '"state_change"' in t
        ]
        assert len(state_events) >= 3
