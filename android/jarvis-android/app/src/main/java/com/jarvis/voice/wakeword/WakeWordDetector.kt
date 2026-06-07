package com.jarvis.voice.wakeword

import android.util.Log
import kotlin.math.sqrt

/**
 * On-device wake word detection.
 *
 * This is a lightweight, energy‑based fallback that detects when someone is speaking
 * (voice activity).  A proper wake word model (e.g. .tflite or .onnx) can be swapped in
 * by replacing [processChunk] with a model inference call.
 *
 * Detection result contains a normalised confidence score.
 */
class WakeWordDetector(
    private val wakeWord: String = "hey jarvis",
    private val sensitivity: Float = 0.5f,
) {
    /** Energy threshold derived from sensitivity (higher sensitivity = lower threshold). */
    private val energyThreshold: Float = 0.1f + (1f - sensitivity) * 0.3f

    /** Minimum consecutive high‑energy frames to trigger detection (~150 ms at 30 ms frames). */
    private val requiredFrames: Int = 5

    /** Frame counter for current utterance. */
    private var activeFrameCount = 0

    @Volatile
    private var lastDetectionTimeMs: Long = 0L

    /** Cooldown between detections in ms. */
    private val cooldownMs: Long = 2000L

    // ---- Public API ----

    /**
     * Process an audio chunk and return detection result.
     * @param audioBytes PCM16 audio data (16 kHz, MONO).
     */
    fun processChunk(audioBytes: ByteArray): DetectionResult {
        val rms = computeRms(audioBytes)
        val score = (rms / 0.5f).coerceIn(0f, 1f) // Normalise: 0.5 RMS ≈ full speech

        val isSpeech = rms > energyThreshold

        if (isSpeech) {
            activeFrameCount++
        } else {
            activeFrameCount = 0
        }

        val now = System.currentTimeMillis()
        val detected = activeFrameCount >= requiredFrames &&
                (now - lastDetectionTimeMs) > cooldownMs

        return if (detected) {
            lastDetectionTimeMs = now
            activeFrameCount = 0
            DetectionResult(detected = true, score = score)
        } else {
            DetectionResult(detected = false, score = if (isSpeech) score else 0f)
        }
    }

    /**
     * Reset the internal state (frame counter, cooldown).
     */
    fun reset() {
        activeFrameCount = 0
        lastDetectionTimeMs = 0L
    }

    // ---- Internal helpers ----

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
