package com.jarvis.voice.interrupt

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.sqrt

/**
 * Monitors microphone energy during TTS playback to detect when the user starts speaking.
 *
 * Uses a dedicated [AudioRecord] instance so it does not interfere with the main
 * capture pipeline.  When energy exceeds [energyThreshold] for a minimum number of
 * consecutive frames (~150 ms), the [onInterrupt] callback fires.
 *
 * The interrupt controller should be active only while TTS is playing.
 */
class InterruptController(
    private val energyThreshold: Float = 0.03f,
    private val onInterrupt: () -> Unit,
) {
    private var monitorJob: Job? = null
    private var scope: CoroutineScope? = null
    private var audioRecord: AudioRecord? = null

    @Volatile
    private var isMonitoring = false

    // ---- Public API ----

    /**
     * Start monitoring the microphone for voice energy.
     * Spawns a dedicated AudioRecord on [Dispatchers.IO].
     */
    fun startMonitoring() {
        if (isMonitoring) {
            Log.w(TAG, "Already monitoring — ignoring startMonitoring()")
            return
        }
        isMonitoring = true

        val cs = CoroutineScope(Dispatchers.IO + Job())
        scope = cs

        monitorJob = cs.launch {
            val buffer = ByteArray(1024) // Small buffer for fast ~32 ms reads @16kHz
            val minBufferSize = AudioRecord.getMinBufferSize(
                16000,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            val recordBufferSize = maxOf(minBufferSize, buffer.size * 4)

            val record = try {
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    16000,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    recordBufferSize,
                )
            } catch (e: SecurityException) {
                Log.e(TAG, "RECORD_AUDIO not granted for interrupt monitoring", e)
                isMonitoring = false
                return@launch
            }

            if (record.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "Interrupt AudioRecord failed to initialise")
                record.release()
                isMonitoring = false
                return@launch
            }

            audioRecord = record
            record.startRecording()
            Log.d(TAG, "Interrupt monitoring started")

            var consecutiveSpeechFrames = 0
            val requiredConsecutiveFrames = 5 // ~150 ms at 32 ms per frame

            while (isActive && isMonitoring) {
                val bytesRead = try {
                    record.read(buffer, 0, buffer.size)
                } catch (e: Exception) {
                    Log.e(TAG, "Interrupt read failed", e)
                    break
                }
                if (bytesRead <= 0) continue

                val rms = computeRms(buffer, bytesRead)

                if (rms > energyThreshold) {
                    consecutiveSpeechFrames++
                    if (consecutiveSpeechFrames >= requiredConsecutiveFrames) {
                        Log.i(TAG, "Interrupt triggered (rms=$rms, consecutive=$consecutiveSpeechFrames)")
                        consecutiveSpeechFrames = 0
                        onInterrupt()
                    }
                } else {
                    consecutiveSpeechFrames = 0
                }
            }

            // Cleanup
            try {
                record.stop()
            } catch (_: Exception) {}
            record.release()
            audioRecord = null
            Log.d(TAG, "Interrupt monitoring ended")
        }
    }

    /**
     * Stop monitoring.  Safe to call when not monitoring.
     */
    fun stopMonitoring() {
        isMonitoring = false
        monitorJob?.cancel()
        monitorJob = null
        scope?.cancel()
        scope = null

        audioRecord?.let { record ->
            try {
                if (record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    record.stop()
                }
                record.release()
            } catch (_: Exception) {}
        }
        audioRecord = null
    }

    /** Whether the controller is actively monitoring. */
    val isActive: Boolean get() = isMonitoring

    // ---- Internal ----

    private fun computeRms(buffer: ByteArray, bytesRead: Int): Float {
        if (bytesRead < 2) return 0f
        val sampleCount = bytesRead / 2
        var sumSquares = 0.0
        for (i in 0 until sampleCount * 2 step 2) {
            val sample = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
            sumSquares += (sample * sample).toDouble()
        }
        return (sqrt(sumSquares / sampleCount) / 32768.0).toFloat()
    }

    companion object {
        private const val TAG = "InterruptCtrl"
    }
}
