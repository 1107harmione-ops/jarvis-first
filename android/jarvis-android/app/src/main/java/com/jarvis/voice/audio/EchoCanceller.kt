package com.jarvis.voice.audio

import android.util.Log
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Basic Acoustic Echo Canceller (AEC) using Normalised Least Mean Squares (NLMS) adaptive filter.
 *
 * This is a software AEC that can reduce echo caused by the loudspeaker bleeding
 * into the microphone.  It is most effective when the audio path is linear and the
 * filter length is sufficient to cover the echo tail (~50 ms at 16 kHz = 800 taps).
 *
 * Usage:
 *   val aec = EchoCanceller(filterLength = 800, stepSize = 0.1f)
 *   while (capturing) {
 *       val micSignal = readMicChunk()     // FloatArray
 *       val refSignal = getPlaybackBuffer() // FloatArray of the same length
 *       val cleaned = aec.process(micSignal, refSignal)
 *   }
 */
class EchoCanceller(
    /** Filter length in taps (samples).  800 ≈ 50 ms @ 16 kHz. */
    private val filterLength: Int = 800,
    /** NLMS step size (0.0 – 1.0).  Lower = slower adaptation but more stable. */
    private val stepSize: Float = 0.1f,
    /** Regularisation constant to prevent division by zero. */
    private val regularization: Float = 1e-6f,
) {

    private val filterWeights: FloatArray = FloatArray(filterLength) { 0f }

    /** Number of samples processed (for diagnostic logging). */
    private var totalSamplesProcessed: Long = 0L

    /**
     * Process one frame of microphone audio with the corresponding playback reference.
     *
     * @param micSignal  Microphone input (PCM16 converted to floats in [-1, 1]).
     * @param refSignal  Playback reference signal (same length and format).
     * @return Echo‑cancelled output (same length as input).
     */
    fun process(micSignal: FloatArray, refSignal: FloatArray): FloatArray {
        val n = min(micSignal.size, refSignal.size)
        if (n == 0) return micSignal

        val output = FloatArray(n)

        for (i in 0 until n) {
            // Build the reference vector (most recent filterLength samples of refSignal)
            val startIdx = maxOf(0, i - filterLength + 1)
            val endIdx = i + 1
            val refWindowLen = endIdx - startIdx
            val refWindow = FloatArray(filterLength) { 0f }
            for (j in startIdx until endIdx) {
                refWindow[filterLength - (endIdx - j)] = refSignal[j]
            }

            // Estimate echo: dot product of filter weights and reference window
            var estimatedEcho = 0f
            for (j in 0 until filterLength) {
                estimatedEcho += filterWeights[j] * refWindow[j]
            }

            // Error signal = microphone - estimated echo
            val error = micSignal[i] - estimatedEcho
            output[i] = error

            // NLMS weight update
            var refEnergy = regularization
            for (j in 0 until filterLength) {
                refEnergy += refWindow[j] * refWindow[j]
            }

            val mu = stepSize / refEnergy
            for (j in 0 until filterLength) {
                filterWeights[j] += mu * error * refWindow[j]
            }

            totalSamplesProcessed++
        }

        return output
    }

    /**
     * Reset the adaptive filter to its initial state (all zeros).
     */
    fun reset() {
        filterWeights.fill(0f)
        totalSamplesProcessed = 0L
        Log.d(TAG, "Echo canceller reset")
    }

    /** Current mean squared weight magnitude (diagnostic). */
    fun filterNorm(): Float {
        var sumSq = 0f
        for (w in filterWeights) sumSq += w * w
        return sqrt(sumSq / filterLength)
    }

    companion object {
        private const val TAG = "EchoCanceller"
    }
}
