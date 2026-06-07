package com.jarvis.voice.capture

import com.jarvis.voice.state.VoiceStateHolder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Continuously monitors and publishes audio level updates to [VoiceStateHolder].
 *
 * Reads the current amplitude from [AudioCaptureService] at a regular interval
 * so that UI components can render a live level meter without being coupled
 * to the capture callback frequency.
 */
class AudioLevelMonitor {

    private var monitorJob: Job? = null
    private var scope: CoroutineScope? = null

    /** Interval between level samples (milliseconds). */
    @Volatile
    var pollIntervalMs: Long = 80L // ~12 Hz — smooth enough for a meter

    /**
     * Start polling [capture] for amplitude updates.
     * @param capture The active audio capture to read from.
     */
    fun start(capture: AudioCaptureService) {
        stop()

        val cs = CoroutineScope(Dispatchers.Default + Job())
        scope = cs

        monitorJob = cs.launch {
            while (isActive) {
                val level = capture.currentAmplitude
                VoiceStateHolder.updateAudioLevel(level)
                delay(pollIntervalMs)
            }
        }
    }

    /** Stop monitoring. */
    fun stop() {
        monitorJob?.cancel()
        monitorJob = null
        scope?.cancel()
        scope = null
    }
}
