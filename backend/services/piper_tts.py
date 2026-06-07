"""
Piper TTS Service — Local neural text-to-speech with streaming support.

Runs Piper (https://github.com/rhasspy/piper) as an async subprocess
for high-quality, low-latency TTS generation. Supports multiple voices,
streaming audio output, speed/pitch controls, and automatic model download.

Falls back to DeepSeek/OpenAI TTS API when Piper is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Callable
from urllib.request import urlretrieve

import numpy as np

from backend.config.settings import settings

logger = logging.getLogger("jarvis.piper_tts")

# Piper produces 16-bit MONO PCM at 22050 Hz by default
PIPER_SAMPLE_RATE = 22050
PIPER_SAMPLE_WIDTH = 2  # 16-bit
PIPER_CHANNELS = 1
PIPER_OUTPUT_FORMAT = "raw"  # raw PCM16, no WAV headers

# Voice model URLs (opensource voices from Rhasspy)
PIPER_VOICE_MODELS: dict[str, dict[str, str]] = {
    "en_US-amy-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "language": "en",
        "quality": "medium",
    },
    "en_US-lessac-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "language": "en",
        "quality": "medium",
    },
    "en_US-ryan-high": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
        "language": "en",
        "quality": "high",
    },
    "en_GB-semaine-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json",
        "language": "en",
        "quality": "medium",
    },
    "hi_IN-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/medium/hi_IN-medium.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/medium/hi_IN-medium.onnx.json",
        "language": "hi",
        "quality": "medium",
    },
}

# Map from settings voice names to Piper model keys
DEFAULT_VOICE_MAP: dict[str, str] = {
    "en": "en_US-lessac-medium",
    "en-US": "en_US-lessac-medium",
    "en-GB": "en_GB-semaine-medium",
    "hi": "hi_IN-medium",
    "hi-IN": "hi_IN-medium",
}


class PiperError(Exception):
    """Raised when Piper TTS encounters an error."""


class PiperVoiceModel:
    """Represents a downloaded Piper voice model."""

    def __init__(self, model_key: str, model_path: Path, config_path: Path) -> None:
        self.model_key = model_key
        self.model_path = model_path
        self.config_path = config_path
        self.meta = PIPER_VOICE_MODELS.get(model_key, {})
        self.language = self.meta.get("language", "en")
        self.quality = self.meta.get("quality", "medium")

    @property
    def is_ready(self) -> bool:
        return self.model_path.exists() and self.config_path.exists()

    def __repr__(self) -> str:
        return f"PiperVoiceModel(key={self.model_key}, lang={self.language})"


class PiperTTSService:
    """
    Service for text-to-speech synthesis using Piper.

    Features:
    - Streaming audio generation (yields PCM16 chunks)
    - Multiple voices (English, Hindi)
    - Speed and pitch controls
    - Automatic model download
    - Fallback to API-based TTS when Piper unavailable
    """

    def __init__(self) -> None:
        self._initialized = False
        self._available = False
        self._piper_bin: str | None = None
        self._models_dir = Path(settings.PIPER_MODEL_PATH)
        self._models: dict[str, PiperVoiceModel] = {}
        self._current_model_key: str | None = None

    async def initialize(self) -> None:
        """Initialize the Piper service. Finds the binary and downloads models."""
        if self._initialized:
            return
        self._initialized = True

        # Find Piper binary
        self._piper_bin = await self._find_piper_binary()
        if not self._piper_bin:
            logger.warning(
                "Piper binary not found. TTS will fall back to API-based synthesis. "
                "Install Piper from https://github.com/rhasspy/piper"
            )
            self._available = False
            return

        # Ensure models directory exists
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Discover downloaded models
        self._discover_models()

        self._available = True
        logger.info(
            "Piper TTS initialized. Binary: %s, Models: %d",
            self._piper_bin,
            len(self._models),
        )

    @property
    def available(self) -> bool:
        """Whether Piper is available for use."""
        return self._available

    async def _find_piper_binary(self) -> str | None:
        """Find the Piper binary on the system."""
        # Check configured path first
        configured = settings.PIPER_EXECUTABLE_PATH
        if configured and Path(configured).exists():
            return str(Path(configured).resolve())

        # Check PATH
        system_piper = shutil.which("piper")
        if system_piper:
            return system_piper

        # Check common locations
        common_paths = [
            "./bin/piper",
            "./piper/piper",
            "/usr/local/bin/piper",
            "/usr/bin/piper",
        ]
        for path in common_paths:
            if Path(path).exists():
                return str(Path(path).resolve())

        return None

    def _discover_models(self) -> None:
        """Discover already-downloaded Piper models."""
        for model_key in PIPER_VOICE_MODELS:
            model_path = self._models_dir / f"{model_key}.onnx"
            config_path = self._models_dir / f"{model_key}.onnx.json"
            if model_path.exists() and config_path.exists():
                self._models[model_key] = PiperVoiceModel(
                    model_key=model_key,
                    model_path=model_path,
                    config_path=config_path,
                )

    async def download_model(self, model_key: str, force: bool = False) -> PiperVoiceModel:
        """Download a Piper voice model from HuggingFace."""
        if model_key not in PIPER_VOICE_MODELS:
            available = list(PIPER_VOICE_MODELS.keys())
            raise PiperError(
                f"Unknown model '{model_key}'. Available: {', '.join(available)}"
            )

        model_info = PIPER_VOICE_MODELS[model_key]
        model_path = self._models_dir / f"{model_key}.onnx"
        config_path = self._models_dir / f"{model_key}.onnx.json"

        if model_path.exists() and config_path.exists() and not force:
            return self._models[model_key]

        logger.info("Downloading Piper model: %s", model_key)
        logger.info("  Model: %s", model_info["url"])
        logger.info("  Config: %s", model_info["config_url"])

        def _download(url: str, dest: Path) -> None:
            """Synchronous download with progress reporting."""
            def report(cur: int, total: int, _: int) -> None:
                if total > 0:
                    pct = cur * 100 // total
                    if pct % 25 == 0:
                        logger.debug("  Download %s: %d%%", dest.name, pct)
            urlretrieve(url, str(dest), reporthook=report)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download, model_info["url"], model_path)
        await loop.run_in_executor(None, _download, model_info["config_url"], config_path)

        model = PiperVoiceModel(
            model_key=model_key,
            model_path=model_path,
            config_path=config_path,
        )
        self._models[model_key] = model
        logger.info("Downloaded model: %s", model_key)
        return model

    async def set_voice(self, language: str = "en", voice_name: str | None = None) -> str:
        """Set the active voice model by language or explicit name."""
        if not self._available:
            logger.warning("Piper not available, voice selection deferred")
            return ""

        if voice_name and voice_name in self._models:
            self._current_model_key = voice_name
            return voice_name

        # Resolve from language
        model_key = DEFAULT_VOICE_MAP.get(language, "en_US-lessac-medium")

        # If not downloaded, try to download
        if model_key not in self._models:
            try:
                await self.download_model(model_key)
            except Exception as e:
                logger.warning("Failed to download model %s: %s", model_key, e)
                # Fall back to any available model
                if self._models:
                    model_key = list(self._models.keys())[0]
                else:
                    self._available = False
                    return ""

        self._current_model_key = model_key
        return model_key

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str | None = None,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> bytes:
        """
        Synthesize text to audio. Returns complete PCM16 audio bytes.

        Args:
            text: Text to synthesize
            language: Language code (en, hi)
            voice: Specific voice name override
            speed: Speaking speed (0.5-2.0, default 1.0)
            pitch: Voice pitch (0.5-2.0, default 1.0)

        Returns:
            Raw PCM16 audio bytes at 22050Hz MONO
        """
        chunks: list[bytes] = []
        async for chunk in self.stream_synthesize(text, language, voice, speed, pitch):
            chunks.append(chunk)
        return b"".join(chunks)

    async def stream_synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str | None = None,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text-to-speech audio chunk by chunk.

        Yields raw PCM16 chunks at 22050Hz MONO.
        Falls back to API-based TTS if Piper is unavailable or errors.
        """
        if not self._available:
            async for chunk in self._fallback_tts(text, language):
                yield chunk
            return

        # Set the voice
        model_key = voice or ""
        if not model_key or model_key not in self._models:
            model_key = await self.set_voice(language, voice)

        if not model_key or model_key not in self._models:
            logger.warning("No Piper model available, falling back to API TTS")
            async for chunk in self._fallback_tts(text, language):
                yield chunk
            return

        model = self._models[model_key]

        # Map speed to Piper's --length-scale (inverse relationship)
        # speed=2.0 → half length → faster speech
        length_scale = 1.0 / max(0.5, min(speed, 2.0))

        # Map pitch to Piper's --noise-scale
        noise_scale = max(0.3, min(pitch * 0.667, 1.0))

        try:
            async for chunk in self._run_piper(
                text=text,
                model_path=str(model.model_path),
                config_path=str(model.config_path),
                length_scale=length_scale,
                noise_scale=noise_scale,
            ):
                yield chunk
        except Exception as e:
            logger.error("Piper synthesis failed: %s. Falling back to API.", e)
            async for chunk in self._fallback_tts(text, language):
                yield chunk

    async def _run_piper(
        self,
        text: str,
        model_path: str,
        config_path: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
    ) -> AsyncGenerator[bytes, None]:
        """
        Run Piper as a subprocess and stream the output.

        Piper reads text from stdin and outputs raw PCM16 to stdout.
        """
        if not self._piper_bin:
            raise PiperError("Piper binary not available")

        cmd = [
            self._piper_bin,
            "--model", model_path,
            "--config", config_path,
            "--output-raw",
            "--length-scale", str(length_scale),
            "--noise-scale", str(noise_scale),
            "--sentence-silence", "0.2",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if process.stdin is None or process.stdout is None:
            raise PiperError("Failed to create Piper subprocess pipes")

        # Send text to stdin and close it
        process.stdin.write(text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        # Read audio data from stdout in chunks
        chunk_size = 4096  # 4096 bytes = ~2048 samples = ~93ms at 22050Hz
        total_bytes = 0
        while True:
            chunk = await process.stdout.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            yield chunk

        # Wait for process to finish
        stderr = await asyncio.wait_for(process.stderr.read(), timeout=5.0)
        await process.wait()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
            logger.warning("Piper exited with code %d: %s", process.returncode, error_msg)

        logger.debug(
            "Piper TTS: %d bytes, %d chars, returncode=%d",
            total_bytes,
            len(text),
            process.returncode,
        )

    async def _fallback_tts(
        self, text: str, language: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        """
        Fallback TTS using DeepSeek/OpenAI API when Piper is unavailable.

        Yields PCM16 audio chunks.
        """
        from backend.services.voice_service import voice_service

        try:
            audio_data = await voice_service.synthesize(
                text=text,
                voice="alloy",
                speed=1.0,
            )
            # The API returns opus audio — yield as a single chunk
            # In production, decode opus to PCM16 for streaming
            yield audio_data
        except Exception as e:
            logger.error("Fallback TTS failed: %s", e)
            # Return silence as last resort (1 second of silence)
            silence = struct.pack("<h", 0) * int(PIPER_SAMPLE_RATE * 0.5)
            yield silence

    async def get_available_voices(self) -> list[dict[str, str]]:
        """Get list of available (ready) voices."""
        voices = []
        for model_key, model in self._models.items():
            if model.is_ready:
                voices.append({
                    "key": model_key,
                    "language": model.language,
                    "quality": model.quality,
                    "name": model_key,
                })
        # Add available but not yet downloaded voices
        for model_key, info in PIPER_VOICE_MODELS.items():
            if model_key not in self._models:
                voices.append({
                    "key": model_key,
                    "language": info["language"],
                    "quality": info["quality"],
                    "name": model_key,
                    "downloadable": True,
                })
        return voices

    async def close(self) -> None:
        """Clean up resources."""
        self._models.clear()
        self._available = False
        self._initialized = False


# Singleton
piper_service = PiperTTSService()
