"""Streaming Piper TTS via subprocess with --output-raw."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Coroutine

from jarvis_voice.config import VoiceConfig
from jarvis_voice.tts.base import BaseTTSProvider

logger = logging.getLogger("jarvis_voice.tts")


class PiperTTS(BaseTTSProvider):
    """Streaming TTS using the Piper binary via subprocess.

    Launches `piper` with ``--output-raw`` which writes raw PCM16 mono
    audio to stdout. Reads stdout in chunks and sends them via callback.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.executable = config.piper_executable or self._find_piper()
        self.voices = {
            "en": config.piper_voice_en,
            "hi": config.piper_voice_hi,
        }
        self.voices_dir = Path(config.piper_voices_dir)
        self.length_scale = config.tts_length_scale
        self.noise_scale = config.tts_noise_scale

    @staticmethod
    def _find_piper() -> str:
        """Find the piper binary in PATH or common locations."""
        for candidate in ["piper", "piper-tts", "/usr/local/bin/piper"]:
            if os.access(candidate, os.X_OK):
                return candidate
        # Fallback — let the subprocess fail with a clear error
        logger.warning("Piper executable not found; will try 'piper' from PATH")
        return "piper"

    async def speak_stream(
        self,
        text: str,
        language: str,
        chunk_callback: Callable[[bytes], None],
        interrupt_event: asyncio.Event,
    ) -> None:
        """Synthesise text with Piper and stream PCM16 chunks.

        Args:
            text: Text to convert to speech.
            language: Language code for voice selection ("en", "hi").
            chunk_callback: Async callback called with each raw PCM16 chunk.
            interrupt_event: When set, the subprocess is killed and the
                method returns immediately.
        """
        model_path, config_path = await self._get_voice_paths(language)

        if not model_path.exists():
            logger.error(
                "Piper voice model not found: %s (language=%s)",
                model_path, language,
            )
            # Generate silence as graceful fallback
            await self._fallback_silence(chunk_callback, interrupt_event)
            return

        args = [
            str(self.executable),
            "--model", str(model_path.absolute()),
            "--output-raw",
            "--length-scale", str(self.length_scale),
            "--noise-scale", str(self.noise_scale),
        ]
        if config_path.exists():
            args.extend(["--config", str(config_path.absolute())])
        else:
            args.extend(["--sample-rate", str(self.config.tts_sample_rate)])

        logger.debug(
            "Piper subprocess: %s (lang=%s, model=%s)",
            " ".join(args[:4]), language, model_path.name,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            async def feed_stdin() -> None:
                """Write text to piper's stdin."""
                try:
                    proc.stdin.write(text.encode("utf-8"))
                    await proc.stdin.drain()
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                except Exception as exc:
                    logger.debug("Piper stdin error: %s", exc)

            async def read_stdout() -> None:
                """Read PCM16 chunks from piper's stdout."""
                try:
                    while True:
                        chunk = await proc.stdout.read(4096)
                        if not chunk:
                            break
                        if interrupt_event.is_set():
                            logger.debug("TTS interrupted — killing piper")
                            proc.kill()
                            return
                        await chunk_callback(chunk)
                except Exception as exc:
                    logger.debug("Piper stdout error: %s", exc)

            await asyncio.gather(feed_stdin(), read_stdout())
            await proc.wait()

        except FileNotFoundError:
            logger.error("Piper executable not found: %s", self.executable)
            await self._fallback_silence(chunk_callback, interrupt_event)
        except Exception as exc:
            logger.exception("Piper subprocess error: %s", exc)
            raise

    async def _get_voice_paths(self, language: str) -> tuple[Path, Path]:
        """Resolve the model and config paths for the given language.

        Returns:
            Tuple of (model_path, config_path).
        """
        voice_name = self.voices.get(language, self.voices["en"])
        model_path = self.voices_dir / f"{voice_name}.onnx"
        config_path = self.voices_dir / f"{voice_name}.onnx.json"
        return model_path, config_path

    async def _fallback_silence(
        self,
        chunk_callback: Callable[[bytes], None],
        interrupt_event: asyncio.Event,
    ) -> None:
        """Generate a short silence as graceful fallback when Piper is unavailable."""
        silence_duration = 0.5  # seconds
        sample_rate = self.config.tts_sample_rate
        total_samples = int(sample_rate * silence_duration)
        chunk_size = 4096
        bytes_per_sample = 2  # PCM16

        offset = 0
        while offset < total_samples * bytes_per_sample:
            if interrupt_event.is_set():
                break
            remaining = (total_samples * bytes_per_sample) - offset
            this_chunk = min(chunk_size, remaining)
            silence_chunk = b"\x00" * this_chunk
            await chunk_callback(silence_chunk)
            offset += this_chunk
            await asyncio.sleep(0)
