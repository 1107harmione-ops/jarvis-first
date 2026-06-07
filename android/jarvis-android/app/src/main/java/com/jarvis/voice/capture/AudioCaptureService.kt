package com.jarvis.voice.capture

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import com.jarvis.voice.state.VoiceStateHolder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.sqrt

/**
 * Streaming PCM audio capture from the device microphone.
 *
 * Captures at 16 kHz, MONO, PCM_16BIT and delivers chunks to the provided callback.
 * Designed to run on [Dispatchers.IO] via a coroutine so the main thread is never blocked.
 */
class AudioCaptureService {

    private var audioRecord: AudioRecord? = null
    private var isCapturing = false
    private var captureJob: Job? = null
    private var scope: CoroutineScope? = null

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    /** Buffer size determined by the platform, multiplied for stability. */
    private val minBufferSize: Int by lazy {
        maxOf(
            AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat),
            4096 // ensure at least 4 KiB
        )
    }

    /** Actual read buffer (one chunk delivered per read). */
    private val readBufferSize = minBufferSize

    /** RMS of the most recently read buffer, for UI visualisation. */
    @Volatile
    private var lastRms: Float = 0f

    /** Whether the service is actively capturing. */
    val isActive: Boolean get() = isCapturing

    /** RMS amplitude of the last captured buffer (normalised 0‑1). */
    val currentAmplitude: Float get() = lastRms

    /**
     * Start capturing audio.
     * @param onAudioChunk Called on [Dispatchers.IO] for every PCM chunk read.
     * @return true if capture started successfully, false otherwise.
     */
    fun start(onAudioChunk: (ByteArray) -> Unit): Boolean {
        if (isCapturing) {
            Log.w(TAG, "start() called while already capturing — ignoring")
            return true
        }

        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                readBufferSize,
            )
        } catch (e: SecurityException) {
            Log.e(TAG, "RECORD_AUDIO permission not granted", e)
            return false
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "Invalid AudioRecord parameters", e)
            return false
        }

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to initialise")
            record.release()
            return false
        }

        audioRecord = record
        isCapturing = true

        val coroutineScope = CoroutineScope(Dispatchers.IO + Job())
        scope = coroutineScope

        captureJob = coroutineScope.launch {
            record.startRecording()
            Log.d(TAG, "Audio capture started (sampleRate=$sampleRate, buffer=$readBufferSize)")

            val buffer = ByteArray(readBufferSize)
            while (isActive && isCapturing) {
                val bytesRead = try {
                    record.read(buffer, 0, buffer.size)
                } catch (e: Exception) {
                    Log.e(TAG, "AudioRecord read failed", e)
                    break
                }

                if (bytesRead > 0) {
                    val chunk = if (bytesRead == buffer.size) buffer else buffer.copyOf(bytesRead)
                    lastRms = computeRms(chunk)
                    VoiceStateHolder.updateAudioLevel(lastRms)
                    onAudioChunk(chunk)
                } else if (bytesRead == AudioRecord.ERROR_INVALID_OPERATION) {
                    Log.e(TAG, "AudioRecord read returned ERROR_INVALID_OPERATION")
                    break
                }
            }
        }

        return true
    }

    /**
     * Stop capturing and release the AudioRecord resource.
     * Safe to call even if not currently capturing.
     */
    fun stop() {
        isCapturing = false
        captureJob?.cancel()
        captureJob = null
        scope?.cancel()
        scope = null

        try {
            audioRecord?.let {
                if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    it.stop()
                }
                it.release()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error while stopping AudioRecord", e)
        }
        audioRecord = null
        lastRms = 0f
        Log.d(TAG, "Audio capture stopped")
    }

    /**
     * Quick restart that flushes the audio hardware buffer.
     * Useful after an interrupt to discard stale audio.
     */
    fun restart() {
        Log.d(TAG, "Restarting audio capture (flushing HW buffer)")
        stop()
        // A tiny delay gives the hardware time to drain its internal buffer.
        try {
            Thread.sleep(80)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    // ---- Internal helpers ----

    /** Compute normalised RMS for a PCM16 buffer. */
    private fun computeRms(buffer: ByteArray): Float {
        if (buffer.size < 2) return 0f
        var sumSquares = 0.0
        val sampleCount = buffer.size / 2
        for (i in 0 until sampleCount * 2 step 2) {
            val sample = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
            sumSquares += (sample * sample).toDouble()
        }
        val rms = sqrt(sumSquares / sampleCount)
        return (rms / 32768.0).toFloat().coerceIn(0f, 1f)
    }

    companion object {
        private const val TAG = "AudioCapture"
    }
}
