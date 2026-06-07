"""
Whisper STT Service — Streaming speech-to-text with partial results.

Supports OpenAI Whisper API for high-quality transcription with:
- Streaming/chunked audio processing
- Partial (interim) results with confidence scores
- Hindi and English language support
- Voice Activity Detection (VAD) for noise robustness
- Low-latency processing
- Fallback to DeepSeek Whisper API
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

import numpy as np

from backend.config.settings import settings

logger = logging.getLogger("jarvis.whisper_stt")

# Audio format constants
WHISPER_SAMPLE_RATE = 16000
WHISPER_SAMPLE_WIDTH = 2  # 16-bit PCM
WHISPER_CHANNELS = 1  # MONO

# VAD constants
VAD_FRAME_MS = 30  # VAD frame duration in ms
VAD_FRAME_SIZE = int(WHISPER_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples

# Silence detection
SILENCE_TIMEOUT_MS = 800  # ms of silence before end-of-utterance
MIN_UTTERANCE_MS = 300  # minimum utterance length in ms
MAX_UTTERANCE_MS = 30000  # maximum utterance length in ms (30s)


@dataclass
class TranscriptionResult:
    """Result from a transcription request."""

    text: str
    confidence: float
    language: str
    duration_ms: float
    is_final: bool = True
    segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "is_final": self.is_final,
        }


@dataclass
class PartialTranscript:
    """Interim transcription result during streaming."""

    text: str
    confidence: float
    is_final: bool = False
    stability: float = 0.0  # 0-1, how stable the partial result is


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection using energy thresholding.

    Detects when speech starts and ends in an audio stream.
    Uses adaptive threshold based on ambient noise floor.
    """

    def __init__(
        self,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        frame_ms: int = VAD_FRAME_MS,
        silence_timeout_ms: int = SILENCE_TIMEOUT_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.silence_timeout_ms = silence_timeout_ms
        self.min_utterance_ms = min_utterance_ms

        self._noise_floor: float = 0.01
        self._noise_floor_alpha: float = 0.01
        self._speech_threshold_db: float = 15.0  # dB above noise floor

        self._is_speaking: bool = False
        self._silence_frames: int = 0
        self._speech_frames: int = 0
        self._silence_frames_required: int = max(
            1, silence_timeout_ms // frame_ms
        )
        self._min_speech_frames: int = max(1, min_utterance_ms // frame_ms)

        # Audio buffer for the current utterance
        self._utterance_buffer: list[bytes] = []
        self._utteration_start_time: float | None = None

    def reset(self) -> None:
        """Reset the VAD state."""
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._utterance_buffer = []
        self._utteration_start_time = None

    def process_chunk(self, audio_chunk: bytes) -> dict[str, Any]:
        """
        Process an audio chunk and return VAD state.

        Returns:
            dict with keys:
                - is_speech: bool — whether speech is currently detected
                - speech_started: bool — transition from silence to speech
                - speech_ended: bool — transition from speech to silence
                - rms: float — RMS energy of the chunk
                - utterance_buffer: bytes — accumulated audio for current utterance
        """
        # Convert bytes to numpy array of int16
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return self._vad_result(False, False, False, 0.0, b"")

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 1.0:
            rms = 0.0

        # Update noise floor (only during silence)
        if not self._is_speaking and rms > 0:
            self._noise_floor = (
                self._noise_floor_alpha * rms
                + (1 - self._noise_floor_alpha) * self._noise_floor
            )

        # Calculate energy threshold
        threshold = self._noise_floor * (10 ** (self._speech_threshold_db / 20))
        threshold = max(threshold, 50.0)  # minimum threshold

        is_speech = rms > threshold
        speech_started = False
        speech_ended = False

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0

            if not self._is_speaking:
                if self._speech_frames >= 2:  # debounce
                    self._is_speaking = True
                    speech_started = True
                    self._utteration_start_time = time.monotonic()
                    self._utterance_buffer = [audio_chunk]
            else:
                self._utterance_buffer.append(audio_chunk)
        else:
            self._silence_frames += 1

            if self._is_speaking:
                self._utterance_buffer.append(audio_chunk)
                if self._silence_frames >= self._silence_frames_required:
                    # Check if utterance was long enough
                    if self._speech_frames >= self._min_speech_frames:
                        self._is_speaking = False
                        speech_ended = True
                    else:
                        # Too short, discard as noise
                        self._is_speaking = False
                        self._utterance_buffer = []
                        self._utteration_start_time = None
                        self._speech_frames = 0

            # Check max utterance duration
            if self._is_speaking and self._utteration_start_time:
                elapsed = (time.monotonic() - self._utteration_start_time) * 1000
                if elapsed >= MAX_UTTERANCE_MS:
                    self._is_speaking = False
                    speech_ended = True

        utterance_bytes = b"".join(self._utterance_buffer) if self._utterance_buffer else b""

        return {
            "is_speech": self._is_speaking,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "rms": rms,
            "utterance_buffer": utterance_bytes if speech_ended else b"",
        }

    def _vad_result(
        self, is_speech: bool, started: bool, ended: bool, rms: float, buf: bytes
    ) -> dict[str, Any]:
        return {
            "is_speech": is_speech,
            "speech_started": started,
            "speech_ended": ended,
            "rms": rms,
            "utterance_buffer": buf,
        }

    @property
    def noise_floor(self) -> float:
        """Current noise floor estimate."""
        return self._noise_floor

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._is_speaking


class WhisperSTTService:
    """
    Speech-to-text service using OpenAI Whisper API with streaming support.

    Features:
    - Streaming recognition with partial results
    - Language auto-detection (Hindi, English)
    - Confidence scores
    - Noise-robust VAD preprocessing
    - Low-latency chunked processing
    - Fallback to DeepSeek Whisper API
    """

    def __init__(self) -> None:
        self._initialized = False
        self._available = False
        self._openai_client: Any = None
        self._deepseek_base_url: str = ""
        self._deepseek_api_key: str = ""

    async def initialize(self) -> None:
        """Initialize the Whisper STT service."""
        if self._initialized:
            return
        self._initialized = True

        # Store DeepSeek config for fallback
        self._deepseek_base_url = settings.DEEPSEEK_BASE_URL
        self._deepseek_api_key = settings.DEEPSEEK_API_KEY

        # Initialize OpenAI client if API key is configured
        if settings.WHISPER_API_KEY:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(
                    api_key=settings.WHISPER_API_KEY,
                )
                self._available = True
                logger.info("Whisper STT initialized with OpenAI API")
            except ImportError:
                logger.warning("openai package not installed. Install with: pip install openai")
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", e)
        else:
            logger.info(
                "No WHISPER_API_KEY configured. Will use DeepSeek Whisper API fallback."
            )

        # If OpenAI unavailable, try DeepSeek
        if not self._available and self._deepseek_api_key:
            self._available = True
            logger.info("Whisper STT initialized with DeepSeek API fallback")

    @property
    def available(self) -> bool:
        return self._available

    def create_vad(self) -> VoiceActivityDetector:
        """Create a new VAD instance for streaming use."""
        return VoiceActivityDetector()

    async def transcribe(
        self,
        audio_data: bytes,
        language: str | None = None,
        sample_rate: int = WHISPER_SAMPLE_RATE,
    ) -> TranscriptionResult:
        """
        Transcribe audio data. Processes complete audio buffer.

        Args:
            audio_data: Raw PCM16 audio bytes
            language: Language hint (None for auto-detect, "en", "hi")
            sample_rate: Sample rate of the audio

        Returns:
            TranscriptionResult with text, confidence, language
        """
        start_time = time.monotonic()

        if not audio_data or len(audio_data) < 32:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language or "en",
                duration_ms=0.0,
            )

        # Convert PCM16 to WAV (Whisper API expects WAV/MP3/...)
        wav_bytes = self._pcm16_to_wav(audio_data, sample_rate)

        result = await self._transcribe_wav(wav_bytes, language)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result.duration_ms = elapsed_ms

        logger.debug(
            "Whisper STT: %d chars, confidence=%.3f, lang=%s, %.0fms",
            len(result.text),
            result.confidence,
            result.language,
            elapsed_ms,
        )

        return result

    async def transcribe_streaming(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        language: str | None = None,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        partial_callback: Callable[[PartialTranscript], None] | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe streaming audio with partial results.

        Processes audio chunks as they arrive, providing interim
        transcriptions via callback.

        Args:
            audio_generator: Async generator yielding PCM16 audio chunks
            language: Language hint
            sample_rate: Sample rate of the audio
            partial_callback: Called with PartialTranscript for interim results

        Returns:
            Final TranscriptionResult
        """
        buffer = bytearray()
        vad = self.create_vad()
        last_partial_time = 0.0

        async for chunk in audio_generator:
            buffer.extend(chunk)
            vad_result = vad.process_chunk(bytes(chunk))

            # Periodically send partial results (every 500ms during speech)
            now = time.monotonic()
            if vad_result["is_speech"] and (now - last_partial_time) > 0.5:
                last_partial_time = now
                if partial_callback and len(buffer) > WHISPER_SAMPLE_RATE * 2:
                    # Only send partial if we have enough audio
                    try:
                        partial = await self._get_partial(
                            bytes(buffer), language
                        )
                        partial_callback(partial)
                    except Exception as e:
                        logger.debug("Partial transcription error (non-fatal): %s", e)

            # Check for utterance end
            if vad_result["speech_ended"]:
                break

        # Final transcription
        return await self.transcribe(bytes(buffer), language, sample_rate)

    async def _transcribe_wav(
        self,
        wav_bytes: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe WAV audio using available API."""
        # Try OpenAI first
        if self._openai_client:
            try:
                return await self._transcribe_openai(wav_bytes, language)
            except Exception as e:
                logger.warning("OpenAI Whisper failed: %s. Trying DeepSeek.", e)

        # Fall back to DeepSeek
        return await self._transcribe_deepseek(wav_bytes, language)

    async def _transcribe_openai(
        self,
        wav_bytes: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API."""
        from openai import AsyncOpenAI

        if not self._openai_client:
            self._openai_client = AsyncOpenAI(api_key=settings.WHISPER_API_KEY)

        kwargs = {
            "model": settings.WHISPER_MODEL,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        # Create a file-like object from the WAV bytes
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"

        response = await self._openai_client.audio.transcriptions.create(
            file=audio_file,
            **kwargs,
        )

        # Parse response
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = response

        text = data.get("text", "")
        segments = data.get("segments", [])

        # Calculate overall confidence from segments
        confidence = 0.0
        if segments:
            confidences = [s.get("confidence", 1.0) for s in segments]
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.95 if text else 0.0

        # Detect language from segments or response
        detected_lang = language or "en"
        if segments and segments[0].get("language"):
            detected_lang = segments[0]["language"]
        elif data.get("language"):
            detected_lang = data["language"]

        return TranscriptionResult(
            text=text.strip(),
            confidence=confidence,
            language=detected_lang,
            duration_ms=0.0,
            is_final=True,
            segments=segments,
        )

    async def _transcribe_deepseek(
        self,
        wav_bytes: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe using DeepSeek Whisper API."""
        import httpx

        url = f"{self._deepseek_base_url}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {self._deepseek_api_key}",
        }

        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "model": (None, settings.STT_MODEL),
        }
        if language:
            files["language"] = (None, language)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, files=files)

        if response.status_code != 200:
            logger.error(
                "DeepSeek Whisper error: %d %s",
                response.status_code,
                response.text,
            )
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language or "en",
                duration_ms=0.0,
            )

        data = response.json()
        text = data.get("text", "") or data.get("transcript", "")

        return TranscriptionResult(
            text=text.strip(),
            confidence=0.9 if text else 0.0,
            language=language or "en",
            duration_ms=0.0,
            is_final=True,
        )

    async def _get_partial(
        self,
        audio_data: bytes,
        language: str | None = None,
    ) -> PartialTranscript:
        """
        Get a partial (interim) transcription.

        Uses a shorter timeout and returns quickly.
        """
        try:
            wav = self._pcm16_to_wav(audio_data, WHISPER_SAMPLE_RATE)
            result = await self._transcribe_wav(wav, language)
            return PartialTranscript(
                text=result.text,
                confidence=result.confidence * 0.8,  # Lower confidence for partials
                is_final=False,
                stability=0.5,
            )
        except Exception:
            return PartialTranscript(
                text="",
                confidence=0.0,
                is_final=False,
                stability=0.0,
            )

    def _pcm16_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM16 bytes to WAV format."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(WHISPER_CHANNELS)
            wf.setsampwidth(WHISPER_SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def resample_audio(
        self, audio_data: bytes, orig_rate: int, target_rate: int = WHISPER_SAMPLE_RATE
    ) -> bytes:
        """Resample audio to target sample rate using linear interpolation."""
        if orig_rate == target_rate:
            return audio_data

        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return b""

        # Calculate resampling ratio
        ratio = target_rate / orig_rate
        new_length = int(len(samples) * ratio)

        # Linear interpolation
        indices = np.arange(new_length) / ratio
        left = indices.astype(np.int64)
        right = np.clip(left + 1, 0, len(samples) - 1)
        frac = indices - left

        resampled = samples[left] * (1 - frac) + samples[right] * frac
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

        return resampled.tobytes()

    async def close(self) -> None:
        """Clean up resources."""
        self._openai_client = None
        self._available = False
        self._initialized = False


# Singleton
whisper_service = WhisperSTTService()
