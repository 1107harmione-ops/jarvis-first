"""VAD tests."""
from __future__ import annotations

import struct

from app.voice.vad import VoiceActivityDetector


def _make_silence_frame(size: int = 960) -> bytes:
    return b"\x00\x00" * (size // 2)


def _make_speech_frame(size: int = 960, amplitude: int = 5000) -> bytes:
    return struct.pack("<" + "h" * (size // 2), *([amplitude] * (size // 2)))


class TestVoiceActivityDetector:
    def setup_method(self):
        self.vad = VoiceActivityDetector()

    def test_is_speech_silence(self):
        frame = _make_silence_frame()
        assert not self.vad.is_speech(frame)

    def test_is_speech_active(self):
        frame = _make_speech_frame()
        assert self.vad.is_speech(frame)

    def test_is_speech_empty(self):
        assert not self.vad.is_speech(b"")

    def test_energy_fallback_silence(self):
        assert not self.vad._energy_is_speech(_make_silence_frame())

    def test_energy_fallback_speech(self):
        assert self.vad._energy_is_speech(_make_speech_frame(amplitude=8000))

    def test_energy_fallback_empty(self):
        assert not self.vad._energy_is_speech(b"")

    def test_detect_activity_no_speech(self):
        audio = _make_silence_frame(48000)
        segments = self.vad.detect_activity(audio)
        assert segments == []

    def test_detect_activity_with_speech(self):
        silence = _make_silence_frame(24000)
        speech = _make_speech_frame(24000, amplitude=8000)
        audio = silence + speech + silence

        segments = self.vad.detect_activity(audio)
        assert len(segments) >= 1

    def test_trim_silence_no_speech(self):
        audio = _make_silence_frame(48000)
        result = self.vad.trim_silence(audio)
        assert result == b""

    def test_trim_silence_with_speech(self, tmp_path):
        silence = _make_silence_frame(24000)
        speech = _make_speech_frame(48000, amplitude=8000)
        audio = silence + speech + silence

        result = self.vad.trim_silence(audio)
        assert len(result) > 0
        assert len(result) < len(audio)

    def test_has_webrtc_property(self):
        assert isinstance(self.vad.has_webrtc, bool)
