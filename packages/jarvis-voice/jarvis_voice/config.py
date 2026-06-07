"""Voice configuration using pydantic-settings BaseSettings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar


class VoiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Audio ──────────────────────────────────────────────────────────
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    bytes_per_sample: int = 2  # PCM16

    @property
    def frame_size(self) -> int:
        """Number of bytes per audio frame."""
        return int(self.sample_rate * self.frame_ms / 1000) * self.bytes_per_sample * self.channels

    # ── STT ────────────────────────────────────────────────────────────
    stt_provider: str = "faster_whisper"
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    stt_language: str = "auto"
    stt_beam_size: int = 5
    stt_vad_filter: bool = True
    partial_results_interval: float = 0.3

    # ── TTS ────────────────────────────────────────────────────────────
    tts_provider: str = "piper"
    piper_executable: str = ""
    piper_voice_en: str = "en_US-lessac-medium"
    piper_voice_hi: str = "hi_IN-medium"
    piper_voices_dir: str = "voices"
    tts_length_scale: float = 1.0
    tts_noise_scale: float = 0.667
    tts_sample_rate: int = 22050

    # ── Wake Word ──────────────────────────────────────────────────────
    wake_word: str = "hey jarvis"
    wake_word_provider: str = "openwakeword"
    wake_word_sensitivity: float = 0.5
    wake_word_cooldown: float = 2.0

    # ── Session ────────────────────────────────────────────────────────
    silence_timeout_sec: float = 1.5
    max_command_duration: float = 30.0
    min_command_duration: float = 0.3
    voice_timeout_sec: float = 10.0

    # ── Interrupt ──────────────────────────────────────────────────────
    interrupt_energy_threshold: float = 0.03
    interrupt_min_duration: float = 0.15

    # ── Multilingual ───────────────────────────────────────────────────
    default_language: str = "en"
    supported_languages: list[str] = ["en", "hi"]
    auto_detect_language: bool = True

    # ── Memory ─────────────────────────────────────────────────────────
    voice_history_ttl_days: int = 90
    max_frequent_commands: int = 50

    # ── Server ─────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "INFO"

    # Pre-computed helper durations (in samples / bytes)
    _silence_samples: ClassVar[int] = 0
    _max_command_samples: ClassVar[int] = 0
    _interrupt_samples: ClassVar[int] = 0

    def model_post_init(self, __context) -> None:
        # Cached duration-derived values
        object.__setattr__(
            self, "_silence_samples",
            int(self.sample_rate * self.silence_timeout_sec),
        )
        object.__setattr__(
            self, "_max_command_samples",
            int(self.sample_rate * self.max_command_duration),
        )
        object.__setattr__(
            self, "_interrupt_samples",
            int(self.sample_rate * self.interrupt_min_duration),
        )
