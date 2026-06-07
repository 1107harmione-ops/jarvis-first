package com.jarvis.voice.playback

import android.content.Context
import android.media.AudioManager
import android.util.Log
import com.jarvis.voice.state.VoiceStateHolder

/**
 * Manages audio focus to coordinate JARVIS voice output with other audio sources
 * (music players, navigation, calls, etc.).
 *
 * Uses [AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK] so that other apps
 * can temporarily lower their volume while JARVIS speaks.
 */
class AudioFocusManager(private val context: Context) {

    private val audioManager: AudioManager =
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    @Volatile
    private var hasFocus: Boolean = false

    /**
     * Whether this manager currently holds audio focus.
     */
    val isHoldingFocus: Boolean get() = hasFocus

    /**
     * Request transient audio focus with ducking.
     * @return The result of [AudioManager.requestAudioFocus] (AUDIOFOCUS_REQUEST_GRANTED etc.).
     */
    fun requestFocus(): Int {
        val result = audioManager.requestAudioFocus(
            focusChangeListener,
            AudioManager.STREAM_MUSIC,
            AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK,
        )
        hasFocus = (result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED)
        Log.d(TAG, "Audio focus requested: ${focusResultToString(result)}")
        return result
    }

    /**
     * Abandon audio focus — call when TTS playback finishes.
     */
    fun abandonFocus() {
        audioManager.abandonAudioFocus(focusChangeListener)
        hasFocus = false
        Log.d(TAG, "Audio focus abandoned")
    }

    // ---- Focus change listener ----

    private val focusChangeListener = AudioManager.OnAudioFocusChangeListener { change ->
        Log.d(TAG, "Audio focus changed: ${focusChangeToString(change)}")
        when (change) {
            AudioManager.AUDIOFOCUS_GAIN -> {
                hasFocus = true
                VoiceStateHolder.onAudioFocusGain()
            }

            AudioManager.AUDIOFOCUS_LOSS -> {
                hasFocus = false
                VoiceStateHolder.onAudioFocusLoss()
            }

            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> {
                hasFocus = false
                VoiceStateHolder.onAudioFocusLossTransient()
            }

            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> {
                // We still "have focus" but should lower volume.
                VoiceStateHolder.onAudioFocusDuck()
            }
        }
    }

    companion object {
        private const val TAG = "AudioFocusManager"

        private fun focusResultToString(result: Int): String = when (result) {
            AudioManager.AUDIOFOCUS_REQUEST_GRANTED -> "GRANTED"
            AudioManager.AUDIOFOCUS_REQUEST_FAILED -> "FAILED"
            AudioManager.AUDIOFOCUS_REQUEST_DELAYED -> "DELAYED"
            else -> "UNKNOWN($result)"
        }

        private fun focusChangeToString(change: Int): String = when (change) {
            AudioManager.AUDIOFOCUS_GAIN -> "GAIN"
            AudioManager.AUDIOFOCUS_LOSS -> "LOSS"
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> "LOSS_TRANSIENT"
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> "LOSS_TRANSIENT_CAN_DUCK"
            else -> "UNKNOWN($change)"
        }
    }
}
