"""Streaming Whisper STT using faster-whisper."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import numpy as np

from jarvis_voice.config import VoiceConfig
from jarvis_voice.models import STTResult
from jarvis_voice.stt.base import BaseSTTProvider

logger = logging.getLogger("jarvis_voice.stt")


class WhisperSTT(BaseSTTProvider):
    """Streaming speech-to-text using faster-whisper.

    Accumulates audio from an async queue and periodically runs Whisper
    to produce partial results. On final signal, runs a full transcription.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        self._WhisperModel = None
        self._language_map = {
            "en": "en",
            "hi": "hi",
            "auto": None,
        }
        self._init_model()

    def _init_model(self) -> None:
        """Lazy initialisation of the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            self._WhisperModel = WhisperModel
            self.model = self._WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
        except Exception as exc:
            logger.warning(
                "faster-whisper not available (will use mock in tests): %s", exc
            )
            self.model = None

    async def transcribe_stream(
        self,
        audio_queue: asyncio.Queue[bytes],
        sample_rate: int,
        partial_callback: Callable[[str, float], None],
        language: str = "auto",
    ) -> tuple[str, float, str]:
        """Accumulate audio chunks and produce streaming partial + final results.

        Args:
            audio_queue: Queue of raw PCM16 audio byte chunks.
            sample_rate: Sample rate of incoming audio.
            partial_callback: Called with (partial_text, confidence) periodically.
            language: Language hint ("auto", "en", "hi").

        Returns:
            Tuple of (final_text, confidence, detected_language).
        """
        accumulated = bytearray()
        last_partial_time = 0.0
        partial_interval = self.config.partial_results_interval
        final_text = ""
        final_confidence = 0.0
        detected_lang = language
        seen_end = False
        consecutive_timeouts = 0

        while True:
            try:
                # Wait for audio with a timeout so we can detect end-of-stream
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                accumulated.extend(chunk)
                seen_end = False
                consecutive_timeouts = 0

                # Run partial transcription at regular intervals
                now = time.time()
                if len(accumulated) > self.config.sample_rate * 2 and (
                    now - last_partial_time
                ) >= partial_interval:
                    last_partial_time = now
                    await self._run_partial(
                        bytes(accumulated), sample_rate, partial_callback, language
                    )

            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                # No new audio received — assume end of utterance if we have data
                if len(accumulated) > self.config.sample_rate * self.config.min_command_duration:
                    if seen_end:
                        break  # Second consecutive timeout — finalize
                    seen_end = True
                    # Run a partial before finalizing
                    if len(accumulated) > 0:
                        await self._run_partial(
                            bytes(accumulated), sample_rate, partial_callback, language
                        )
                elif consecutive_timeouts >= 4:
                    # No audio received for 2 seconds — give up
                    break
                else:
                    # Not enough audio yet, keep waiting
                    continue

        # Final full transcription
        if len(accumulated) > 0:
            try:
                audio_array = self._audio_bytes_to_float(bytes(accumulated))
                text, confidence, lang = await self._transcribe_segments(
                    audio_array, language=language
                )
                final_text = text
                final_confidence = confidence
                detected_lang = lang if lang else language
                logger.info(
                    "STT final: text=%.60s confidence=%.3f lang=%s",
                    final_text, final_confidence, detected_lang,
                )
            except Exception as exc:
                logger.exception("Final transcription failed: %s", exc)
                final_text = ""
                final_confidence = 0.0
        else:
            logger.debug("STT: no audio accumulated")

        return final_text, final_confidence, detected_lang

    async def _run_partial(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        partial_callback: Callable[[str, float], None],
        language: str,
    ) -> None:
        """Run a partial transcription for streaming results."""
        try:
            audio_array = self._audio_bytes_to_float(audio_bytes)

            # Use the last ~4 seconds for partial to keep it responsive
            chunk_duration = len(audio_array) / sample_rate
            if chunk_duration > 4.0:
                start_sample = int((chunk_duration - 4.0) * sample_rate)
                audio_array = audio_array[start_sample:]

            text, confidence, _detected = await self._transcribe_segments(
                audio_array, language=language
            )
            if text.strip():
                await partial_callback(text, confidence)
        except Exception as exc:
            logger.debug("Partial transcription error (non-fatal): %s", exc)

    async def transcribe_file(self, audio_path: str, language: str = "auto") -> STTResult:
        """Full file transcription as a non-streaming fallback."""
        try:
            segments, info = await asyncio.to_thread(
                self.model.transcribe,
                audio_path,
                language=self._language_map.get(language, None),
                beam_size=self.config.stt_beam_size,
                vad_filter=self.config.stt_vad_filter,
            )
            texts: list[str] = []
            seg_list: list[dict] = []
            for seg in segments:
                texts.append(seg.text)
                seg_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": getattr(seg, "confidence", 0.0),
                })
            full_text = " ".join(texts)
            confidence = info.average_log_prob if info else 0.0
            lang = info.language if info else language

            return STTResult(
                text=full_text,
                confidence=float(confidence),
                language=lang or language,
                segments=seg_list,
            )
        except Exception as exc:
            logger.exception("File transcription failed: %s", exc)
            return STTResult(text="", confidence=0.0, language=language)

    async def _transcribe_segments(
        self,
        audio_array: np.ndarray,
        language: str | None = None,
    ) -> tuple[str, float, str | None]:
        """Transcribe a float32 audio array and return (text, confidence, language).

        Runs the faster-whisper model in a thread to avoid blocking the event loop.
        """
        lang_param = self._language_map.get(language, language) if language else None
        use_vad = self.config.stt_vad_filter

        def _run() -> tuple[str, float, str | None]:
            segments, info = self.model.transcribe(
                audio_array,
                language=lang_param,
                beam_size=self.config.stt_beam_size,
                vad_filter=use_vad,
            )
            texts: list[str] = []
            total_logprob = 0.0
            seg_count = 0
            for seg in segments:
                texts.append(seg.text)
                total_logprob += getattr(seg, "confidence", seg.avg_logprob if hasattr(seg, "avg_logprob") else 0.0)
                seg_count += 1
            full_text = " ".join(texts).strip()
            avg_confidence = total_logprob / max(seg_count, 1)
            # Convert log probability to a pseudo-confidence in [0, 1]
            confidence = float(max(0.0, min(1.0, (avg_confidence + 5.0) / 5.0)))
            detected = info.language if info else None
            return full_text, confidence, detected

        return await asyncio.to_thread(_run)

    def _audio_bytes_to_float(self, audio_bytes: bytes) -> np.ndarray:
        """Convert PCM16 bytes to float32 array normalized to [-1, 1]."""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        # Clip to prevent outliers
        np.clip(samples, -1.0, 1.0, out=samples)
        return samples
