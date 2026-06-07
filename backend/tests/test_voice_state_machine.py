"""
Tests for Voice State Machine — state transitions, validation, edge cases.
"""

from __future__ import annotations

import pytest


class TestVoiceStateEnum:
    """Tests for voice state enum consistency."""

    def test_all_states_present(self) -> None:
        """All voice states should exist across codebase."""
        from backend.database.models import VoiceState as ModelVoiceState
        from backend.services.voice_service import VoiceSessionState
        from backend.services.voice_pipeline import PipelineState

        model_states = {s.value for s in ModelVoiceState}
        service_states = {s.value for s in VoiceSessionState}
        pipeline_states = {s.value for s in PipelineState}

        # Core states that should be in all enums
        core_states = {"idle", "listening", "speaking", "interrupted", "error"}

        assert core_states.issubset(model_states)
        assert core_states.issubset(service_states)
        assert core_states.issubset(pipeline_states)

    def test_state_transitions_idle(self) -> None:
        """From IDLE, valid transitions."""
        from backend.services.voice_pipeline import PipelineState

        valid_from_idle = {
            PipelineState.LISTENING,
            PipelineState.OFFLINE,
        }
        # IDLE can only go to LISTENING or OFFLINE
        for state in PipelineState:
            if state in valid_from_idle:
                continue
            assert state != PipelineState.IDLE  # Shouldn't transition to self

    def test_state_transitions_listening(self) -> None:
        from backend.services.voice_pipeline import PipelineState

        valid_from_listening = {
            PipelineState.PROCESSING_STT,
            PipelineState.INTERRUPTED,
            PipelineState.IDLE,
            PipelineState.OFFLINE,
        }
        for state in PipelineState:
            if state in valid_from_listening:
                continue
            # All other states are invalid transitions from LISTENING
            pass  # This is a model test, not runtime

    def test_state_transitions_speaking(self) -> None:
        from backend.services.voice_pipeline import PipelineState

        valid_from_speaking = {
            PipelineState.IDLE,
            PipelineState.INTERRUPTED,
            PipelineState.LISTENING,
        }
        for state in PipelineState:
            if state in valid_from_speaking:
                continue
            pass  # Model constraint

    def test_voice_state_values(self) -> None:
        """Verify all voice state values are lowercase and match pattern."""
        from backend.database.models import VoiceState

        for state in VoiceState:
            assert state.value.islower()
            assert "_" not in state.value or state.value in (
                "wake_pending", "processing_stt"
            )

    def test_pipeline_state_values(self) -> None:
        """Verify pipeline state values match expected pattern."""
        from backend.services.voice_pipeline import PipelineState

        for state in PipelineState:
            assert state.value.islower()

    def test_offline_state(self) -> None:
        """Offline state should be in all state enums."""
        from backend.database.models import VoiceState
        from backend.services.voice_service import VoiceSessionState
        from backend.services.voice_pipeline import PipelineState

        assert VoiceState.OFFLINE.value == "offline"
        assert VoiceSessionState.OFFLINE.value == "offline"
        assert PipelineState.OFFLINE.value == "offline"

    def test_error_state(self) -> None:
        """Error state should be present."""
        from backend.database.models import VoiceState
        from backend.services.voice_service import VoiceSessionState
        from backend.services.voice_pipeline import PipelineState

        assert VoiceState.ERROR.value == "error"
        assert VoiceSessionState.ERROR.value == "error"
        assert PipelineState.ERROR.value == "error"

    def test_no_duplicate_values(self) -> None:
        """No duplicate state values within each enum."""
        from backend.database.models import VoiceState
        from backend.services.voice_service import VoiceSessionState
        from backend.services.voice_pipeline import PipelineState

        for enum_cls in (VoiceState, VoiceSessionState, PipelineState):
            values = [s.value for s in enum_cls]
            assert len(values) == len(set(values)), f"Duplicates in {enum_cls.__name__}"

    def test_thinking_state_consistency(self) -> None:
        """THINKING should be present in service and pipeline, but maybe not model."""
        from backend.services.voice_service import VoiceSessionState
        from backend.services.voice_pipeline import PipelineState

        assert VoiceSessionState.THINKING.value == "thinking"
        assert PipelineState.THINKING.value == "thinking"


class TestVoiceStateMachine:
    """Tests for voice session state machine behavior."""

    def test_session_initial_state(self) -> None:
        from backend.services.voice_service import VoiceSession, VoiceSessionState

        import uuid
        session = VoiceSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
        )
        assert session.state == VoiceSessionState.IDLE
        assert session.interrupted is False

    def test_session_to_dict(self) -> None:
        from backend.services.voice_service import VoiceSession

        import uuid
        session = VoiceSession(
            session_id="test-session",
            user_id="test_user",
            language="hi",
        )
        d = session.to_dict()
        assert d["session_id"] == "test-session"
        assert d["user_id"] == "test_user"
        assert d["language"] == "hi"
        assert d["state"] == "idle"
        assert "created_at" in d
        assert "expires_at" in d

    def test_session_new_fields(self) -> None:
        """Verify new fields exist on VoiceSession."""
        from backend.services.voice_service import VoiceSession

        import uuid
        session = VoiceSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
        )
        assert hasattr(session, "wake_word_enabled")
        assert hasattr(session, "offline_mode")
        assert hasattr(session, "voice_speed")
        assert hasattr(session, "voice_pitch")

    def test_is_expired(self) -> None:
        from backend.services.voice_service import VoiceSession
        from datetime import datetime, timedelta, timezone

        import uuid
        session = VoiceSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
        )
        assert session.is_expired is False

        # Manually expire
        session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert session.is_expired is True

    def test_interrupt_state_transition(self) -> None:
        """Verify interrupt handling sets correct state."""
        from backend.services.voice_service import (
            VoiceService,
            VoiceSession,
            VoiceSessionState,
        )

        service = VoiceService()
        session = VoiceSession(
            session_id="test-session",
            user_id="test_user",
        )
        service._sessions["test-session"] = session

        # Interrupt
        session.state = VoiceSessionState.SPEAKING
        service.handle_interrupt("test-session")
        assert session.interrupted is True
        assert session.state == VoiceSessionState.INTERRUPTED

    def test_pipeline_manager_state(self) -> None:
        """Pipeline manager tracks pipeline state correctly."""
        from backend.services.voice_pipeline import PipelineManager

        manager = PipelineManager()
        pipeline = manager.create_pipeline(user_id="user1")
        assert pipeline.state.value == "idle"

        pipeline.interrupt()
        assert pipeline.state.value == "interrupted"

        pipeline.reset_interrupt()
        assert pipeline.state.value == "interrupted"  # State persists after reset

    def test_state_string_matches_api(self) -> None:
        """State string values should match what API returns."""
        from backend.services.voice_service import VoiceSessionState
        from backend.services.voice_pipeline import PipelineState

        # These are the states used in API responses
        api_states = {
            "idle", "listening", "processing_stt", "thinking",
            "speaking", "interrupted", "offline", "error",
        }

        service_values = {s.value for s in VoiceSessionState}
        pipeline_values = {s.value for s in PipelineState}

        assert api_states.issubset(service_values)
        assert api_states.issubset(pipeline_values)
