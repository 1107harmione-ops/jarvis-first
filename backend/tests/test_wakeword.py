"""Wake word detection tests."""
from __future__ import annotations

import struct

from app.voice.wakeword import WakeWordDetector


def _make_pcm_frame(size: int = 1024, amplitude: int = 0) -> bytes:
    return struct.pack("<" + "h" * (size // 2), *([amplitude] * (size // 2)))


class TestWakeWordDetector:
    def setup_method(self):
        self.detector = WakeWordDetector(keywords=["jarvis"])

    def test_available_property(self):
        assert isinstance(self.detector.available, bool)

    def test_process_chunk_silence(self):
        frame = _make_pcm_frame(amplitude=0)
        assert not self.detector.process_chunk(frame)

    def test_process_chunk_empty(self):
        assert not self.detector.process_chunk(b"")

    def test_energy_trigger_silence_only(self):
        detector = WakeWordDetector(keywords=["jarvis"])
        frame = _make_pcm_frame(amplitude=0)
        for _ in range(30):
            result = detector.detect_energy_trigger(frame, energy_threshold=500.0, silence_frames_required=20)
        assert not result

    def test_energy_trigger_speech_after_silence(self):
        detector = WakeWordDetector(keywords=["jarvis"])
        # Send silence frames to establish "silent" state
        silence = _make_pcm_frame(amplitude=0)
        for _ in range(25):
            detector.detect_energy_trigger(silence, energy_threshold=500.0, silence_frames_required=20)

        # Send speech frame - should trigger
        speech = _make_pcm_frame(amplitude=8000)
        result = detector.detect_energy_trigger(speech, energy_threshold=500.0, silence_frames_required=20)
        assert result

    def test_energy_trigger_consecutive_speech(self):
        detector = WakeWordDetector(keywords=["jarvis"])
        silence = _make_pcm_frame(amplitude=0)
        for _ in range(25):
            detector.detect_energy_trigger(silence, energy_threshold=500.0, silence_frames_required=20)

        speech = _make_pcm_frame(amplitude=8000)
        # First speech frame triggers
        assert detector.detect_energy_trigger(speech, energy_threshold=500.0, silence_frames_required=20)

        # Immediately following speech should NOT trigger again
        assert not detector.detect_energy_trigger(speech, energy_threshold=500.0, silence_frames_required=20)

    def test_close_no_error(self):
        detector = WakeWordDetector(keywords=["jarvis"])
        detector.close()  # should not raise

    def test_close_twice_no_error(self):
        detector = WakeWordDetector(keywords=["jarvis"])
        detector.close()
        detector.close()  # should not raise
