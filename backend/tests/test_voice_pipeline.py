"""
Tests for Voice Pipeline Orchestrator — state machine, audio processing, event flow.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self) -> None:
        from backend.services.voice_pipeline import PipelineConfig

        config = PipelineConfig()
        assert config.language == "en"
        assert config.wake_word_enabled is True
        assert config.voice_speed == 1.0
        assert config.voice_pitch == 1.0
        assert config.silence_timeout_ms == 800
        assert config.interrupt_enabled is True
        assert config.offline_mode is False

    def test_custom_config(self) -> None:
        from backend.services.voice_pipeline import PipelineConfig

        config = PipelineConfig(
            language="hi",
            wake_word_enabled=False,
            voice_speed=1.5,
            offline_mode=True,
        )
        assert config.language == "hi"
        assert config.wake_word_enabled is False
        assert config.voice_speed == 1.5
        assert config.offline_mode is True


class TestPipelineState:
    """Tests for pipeline state enum."""

    def test_all_states(self) -> None:
        from backend.services.voice_pipeline import PipelineState

        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.PROCESSING_STT.value == "processing_stt"
        assert PipelineState.THINKING.value == "thinking"
        assert PipelineState.SPEAKING.value == "speaking"
        assert PipelineState.INTERRUPTED.value == "interrupted"
        assert PipelineState.OFFLINE.value == "offline"
        assert PipelineState.ERROR.value == "error"


class TestPipelineMetrics:
    """Tests for pipeline metrics."""

    def test_default_metrics(self) -> None:
        from backend.services.voice_pipeline import PipelineMetrics

        m = PipelineMetrics()
        assert m.stt_latency_ms == 0.0
        assert m.agent_latency_ms == 0.0
        assert m.tts_latency_ms == 0.0
        assert m.interrupted is False

    def test_to_dict(self) -> None:
        from backend.services.voice_pipeline import PipelineMetrics

        m = PipelineMetrics(
            stt_latency_ms=150.0,
            agent_latency_ms=500.0,
            tts_latency_ms=300.0,
            total_pipeline_ms=950.0,
            stt_confidence=0.95,
        )
        d = m.to_dict()
        assert d["stt_latency_ms"] == 150.0
        assert d["stt_confidence"] == 0.95
        assert d["total_pipeline_ms"] == 950.0
        assert d["interrupted"] is False


class TestPipelineResult:
    """Tests for pipeline result."""

    def test_to_dict(self) -> None:
        from backend.services.voice_pipeline import PipelineResult

        result = PipelineResult(
            transcript="hello",
            response="hi there",
            audio_chunks=[b"data1", b"data2"],
        )
        d = result.to_dict()
        assert d["transcript"] == "hello"
        assert d["response"] == "hi there"
        assert d["audio_size"] == 10


class TestPipelineEvent:
    """Tests for pipeline events."""

    def test_create_event(self) -> None:
        from backend.services.voice_pipeline import PipelineEvent

        event = PipelineEvent("transcript", {"text": "hello"})
        assert event.type == "transcript"
        assert event.data["text"] == "hello"

    def test_state_event(self) -> None:
        from backend.services.voice_pipeline import PipelineEvent

        event = PipelineEvent("state", "listening")
        assert event.type == "state"
        assert event.data == "listening"


class TestVoicePipeline:
    """Tests for the VoicePipeline class."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        from backend.services.voice_pipeline import VoicePipeline, PipelineConfig

        pipeline = VoicePipeline(user_id="test_user", config=PipelineConfig())
        assert pipeline.user_id == "test_user"
        assert pipeline.state.value == "idle"
        assert pipeline._interrupted.is_set() is False

    def test_interrupt(self) -> None:
        from backend.services.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(user_id="test_user")
        assert pipeline.is_interrupted is False
        pipeline.interrupt()
        assert pipeline.is_interrupted is True
        assert pipeline.state.value == "interrupted"

    def test_reset_interrupt(self) -> None:
        from backend.services.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(user_id="test_user")
        pipeline.interrupt()
        assert pipeline.is_interrupted is True
        pipeline.reset_interrupt()
        assert pipeline.is_interrupted is False

    def test_interrupt_then_reset_state(self) -> None:
        from backend.services.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(user_id="test_user")
        pipeline.interrupt()
        assert pipeline.state.value == "interrupted"
        pipeline.reset_interrupt()
        # State should still be interrupted after reset
        assert pipeline.state.value == "interrupted"


class TestPipelineManager:
    """Tests for the PipelineManager."""

    def test_create_and_get(self) -> None:
        from backend.services.voice_pipeline import PipelineManager

        manager = PipelineManager()
        pipeline = manager.create_pipeline(user_id="user1")
        assert pipeline is not None
        assert manager.active_count == 1

        retrieved = manager.get_pipeline(pipeline.conversation_id or "")
        assert retrieved is not None

    def test_remove_pipeline(self) -> None:
        from backend.services.voice_pipeline import PipelineManager

        manager = PipelineManager()
        pipeline = manager.create_pipeline(user_id="user1")
        session_key = pipeline.conversation_id or ""
        manager.remove_pipeline(session_key)
        assert manager.active_count == 0

    def test_interrupt_pipeline(self) -> None:
        from backend.services.voice_pipeline import PipelineManager

        manager = PipelineManager()
        pipeline = manager.create_pipeline(user_id="user1")
        session_key = pipeline.conversation_id or ""

        result = manager.interrupt_pipeline(session_key)
        assert result is True
        assert pipeline.is_interrupted is True

    def test_interrupt_nonexistent(self) -> None:
        from backend.services.voice_pipeline import PipelineManager

        manager = PipelineManager()
        result = manager.interrupt_pipeline("nonexistent")
        assert result is False

    def test_multiple_pipelines(self) -> None:
        from backend.services.voice_pipeline import PipelineManager

        manager = PipelineManager()
        manager.create_pipeline(user_id="user1")
        manager.create_pipeline(user_id="user2")
        manager.create_pipeline(user_id="user3")
        assert manager.active_count == 3

    def test_singleton(self) -> None:
        from backend.services.voice_pipeline import pipeline_manager

        assert pipeline_manager is not None
        assert pipeline_manager.active_count >= 0
