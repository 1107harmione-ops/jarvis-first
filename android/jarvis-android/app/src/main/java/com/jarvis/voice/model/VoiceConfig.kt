package com.jarvis.voice.model

/**
 * Voice configuration with sensible defaults.
 * Can be persisted via VoiceSettingsStore and updated at runtime.
 */
data class VoiceConfig(
    val serverUrl: String = "ws://localhost:8002/ws/voice",
    val sampleRate: Int = 16000,
    val language: String = "en",
    val voiceSpeed: Float = 1.0f,
    val wakeWordEnabled: Boolean = true,
    val wakeWordSensitivity: Float = 0.5f,
    val wakeWord: String = "hey jarvis",
    val autoReconnect: Boolean = true,
    val maxReconnectAttempts: Int = 20,
    val interruptEnabled: Boolean = true,
    val interruptEnergyThreshold: Float = 0.03f,
    val bluetoothScoEnabled: Boolean = true,
    val silenceTimeoutMs: Long = 1500L,
    val maxCommandDurationMs: Long = 30_000L,
)
