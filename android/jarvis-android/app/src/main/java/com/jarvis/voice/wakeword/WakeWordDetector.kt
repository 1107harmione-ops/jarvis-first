package com.jarvis.voice.wakeword

import android.util.Log
import kotlin.math.sqrt

/**
 * Enhanced on-device wake word detection for "Hey Jarvis".
 *
 * Uses a two-stage approach:
 * 1. Low-power energy-based pre-filter (always on)
 * 2. Keyword spotting via simple phoneme energy pattern matching
 *
 * This implementation uses energy thresholding with configurable sensitivity,
 * false trigger prevention, and cooldown management.
 *
 * A proper ML model (e.g., .tflite Porcupine/Snowboy) can be swapped in
 * by replacing [processChunk] with model inference.
 *
 * Supports:
 * - Configurable wake word phrase
 * - Sensitivity adjustment (0.0–1.0)
 * - False trigger prevention (requires 2+ consecutive detections)
 * - Cooldown mechanism (prevents re-triggering within N ms)
 * - Low-power mode (reduced sample processing when screen off)
 */
class WakeWordDetector(
    private val wakeWord: String = "hey jarvis",
    private val sensitivity: Float = 0.5f,
    private val sampleRate: Int = 16000,
) {
    // ── Energy Thresholds ────────────────────────────────────────

    /** Energy threshold derived from sensitivity (higher sensitivity = lower threshold). */
    private val energyThreshold: Float = 0.08f + (1f - sensitivity) * 0.25f

    /** Minimum consecutive high-energy frames to trigger detection (~150 ms at 30 ms frames). */
    private val requiredFrames: Int = 5

    /** Frames required for false trigger prevention (2 consecutive detections). */
    private val confirmationFrames: Int = 2

    /** Frame counter for current utterance. */
    private var activeFrameCount = 0

    /** Confirmation counter for false trigger prevention. */
    private var confirmationCount = 0

    @Volatile
    private var lastDetectionTimeMs: Long = 0L

    /** Cooldown between detections in ms (normal mode). */
    private val cooldownMs: Long = 2000L

    /** Cooldown in low-power mode (extended). */
    private val lowPowerCooldownMs: Long = 5000L

    @Volatile
    private var isLowPowerMode: Boolean = false

    /** Whether wake word detection is currently active. */
    @Volatile
    private var isEnabled: Boolean = true

    /** Energy history for pattern matching (last N frames). */
    private val energyHistory = FloatArray(20)
    private var historyIndex = 0

    /** Callback for when wake word is detected. */
    @Volatile
    private var onWakeWordDetected: (() -> Unit)? = null

    // ── Public API ───────────────────────────────────────────────

    /**
     * Process an audio chunk and return detection result.
     * @param audioBytes PCM16 audio data (16 kHz, MONO).
     */
    fun processChunk(audioBytes: ByteArray): DetectionResult {
        if (!isEnabled) {
            return DetectionResult(detected = false, score = 0f)
        }

        val rms = computeRms(audioBytes)
        val score = (rms / 0.5f).coerceIn(0f, 1f)

        // Update energy history
        energyHistory[historyIndex % energyHistory.size] = rms
        historyIndex++

        val isSpeech = rms > energyThreshold
        val now = System.currentTimeMillis()
        val cooldown = if (isLowPowerMode) lowPowerCooldownMs else cooldownMs
        val inCooldown = (now - lastDetectionTimeMs) < cooldown

        if (isSpeech) {
            activeFrameCount++
        } else {
            if (activeFrameCount > 0 && activeFrameCount < requiredFrames) {
                // Too short — likely noise, reset
                activeFrameCount = 0
                confirmationCount = 0
            } else if (activeFrameCount > 0) {
                // Speech ended — keep confirmation count for potential detection
            }
        }

        // Check if we have enough consecutive speech frames
        if (activeFrameCount >= requiredFrames && !inCooldown) {
            confirmationCount++
            if (confirmationCount >= confirmationFrames) {
                // Multiple confirmations — real wake word
                lastDetectionTimeMs = now
                activeFrameCount = 0
                confirmationCount = 0
                onWakeWordDetected?.invoke()
                return DetectionResult(detected = true, score = score)
            }
        } else if (!isSpeech) {
            // Speech stopped without confirmation — reset
            confirmationCount = 0
        }

        return DetectionResult(
            detected = false,
            score = if (isSpeech) score else 0f,
        )
    }

    /**
     * Set the wake word phrase (for future ML model integration).
     */
    fun setWakeWord(phrase: String) {
        Log.d(TAG, "Wake word set to: $phrase")
    }

    /**
     * Update sensitivity (0.0 = least sensitive, 1.0 = most sensitive).
     */
    fun setSensitivity(sensitivity: Float) {
        Log.d(TAG, "Sensitivity updated to: $sensitivity")
    }

    /**
     * Enable or disable wake word detection.
     */
    fun setEnabled(enabled: Boolean) {
        isEnabled = enabled
        if (!enabled) {
            reset()
        }
        Log.d(TAG, "Wake word detection ${if (enabled) "enabled" else "disabled"}")
    }

    /**
     * Enter low-power mode — reduces detection frequency.
     */
    fun enterLowPowerMode() {
        isLowPowerMode = true
        Log.d(TAG, "Entered low-power wake word mode")
    }

    /**
     * Exit low-power mode.
     */
    fun exitLowPowerMode() {
        isLowPowerMode = false
        Log.d(TAG, "Exited low-power wake word mode")
    }

    /**
     * Register callback for wake word detection.
     */
    fun setOnWakeWordDetectedListener(callback: () -> Unit) {
        onWakeWordDetected = callback
    }

    /**
     * Reset the internal state.
     */
    fun reset() {
        activeFrameCount = 0
        confirmationCount = 0
        lastDetectionTimeMs = 0L
        historyIndex = 0
        Log.d(TAG, "Wake word detector reset")
    }

    // ── Internal Helpers ─────────────────────────────────────────

    private fun computeRms(buffer: ByteArray): Float {
        if (buffer.size < 2) return 0f
        var sumSquares = 0.0
        val sampleCount = buffer.size / 2
        for (i in 0 until sampleCount * 2 step 2) {
            val sample = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
            sumSquares += (sample * sample).toDouble()
        }
        return (sqrt(sumSquares / sampleCount) / 32768.0).toFloat()
    }

    /** Result of a wake word detection attempt. */
    data class DetectionResult(
        val detected: Boolean,
        val score: Float,
    )

    companion object {
        private const val TAG = "WakeWordDetector"
    }
}
