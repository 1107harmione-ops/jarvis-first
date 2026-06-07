package com.jarvis.voice.model

/**
 * Runtime metrics for voice system performance monitoring.
 */
data class VoiceMetrics(
    val commandsProcessed: Int = 0,
    val interruptsTriggered: Int = 0,
    val errorsEncountered: Int = 0,
    val totalAudioBytesSent: Long = 0L,
    val totalAudioBytesReceived: Long = 0L,
    val sttLatencyMs: Long = 0L,
    val ttsLatencyMs: Long = 0L,
    val roundTripLatencyMs: Long = 0L,
    val reconnections: Int = 0,
    val uptimeMs: Long = 0L,
    val bluetoothScoDurationMs: Long = 0L,
) {
    /** Reset all counters to zero. */
    fun reset(): VoiceMetrics = VoiceMetrics()
}
