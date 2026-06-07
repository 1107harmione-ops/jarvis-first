package com.jarvis.voice.audio

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.util.Log

/**
 * Routes audio output to the appropriate device: speaker, earpiece, wired headset, or Bluetooth SCO.
 *
 * AudioTrack instances select their output device automatically when built with
 * [android.media.AudioAttributes].  This router provides explicit control for
 * cases where the default routing is undesirable (e.g. forcing earpiece for privacy).
 */
class AudioRouter(private val context: Context) {

    private val audioManager: AudioManager =
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    /** Available audio output device types. */
    enum class AudioOutputDevice {
        SPEAKER,
        EARPIECE,
        WIRED_HEADSET,
        BLUETOOTH_SCO,
        BLUETOOTH_A2DP,
    }

    /**
     * Route playback to the earpiece (phone call receiver).
     * Useful for private listening in public spaces.
     */
    fun routeToEarpiece() {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.isSpeakerphoneOn = false
        Log.d(TAG, "Routed to earpiece")
    }

    /**
     * Route playback to the loudspeaker.
     */
    fun routeToSpeaker() {
        audioManager.mode = AudioManager.MODE_NORMAL
        audioManager.isSpeakerphoneOn = true
        Log.d(TAG, "Routed to speaker")
    }

    /**
     * Route playback to a wired headset.
     */
    fun routeToWiredHeadset() {
        audioManager.mode = AudioManager.MODE_NORMAL
        audioManager.isSpeakerphoneOn = false
        // AudioTrack will automatically select the wired headset when plugged in.
        Log.d(TAG, "Routed to wired headset (preferring wired)")
    }

    /**
     * Detect the current preferred audio output device.
     */
    fun getCurrentDevice(): AudioOutputDevice {
        // 1. Check Bluetooth SCO
        if (audioManager.isBluetoothScoOn) {
            return AudioOutputDevice.BLUETOOTH_SCO
        }

        // 2. Check wired headset via device list
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        for (device in devices) {
            when (device.type) {
                AudioDeviceInfo.TYPE_WIRED_HEADSET,
                AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> {
                    return AudioOutputDevice.WIRED_HEADSET
                }
                AudioDeviceInfo.TYPE_BLUETOOTH_A2DP,
                AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> {
                    return AudioOutputDevice.BLUETOOTH_A2DP
                }
                else -> { /* continue */ }
            }
        }

        // 3. Check speakerphone state
        return if (audioManager.isSpeakerphoneOn) {
            AudioOutputDevice.SPEAKER
        } else {
            AudioOutputDevice.EARPIECE
        }
    }

    /** True if a wired headset is currently plugged in. */
    fun isWiredHeadsetConnected(): Boolean {
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        return devices.any { device ->
            device.type == AudioDeviceInfo.TYPE_WIRED_HEADSET ||
                    device.type == AudioDeviceInfo.TYPE_WIRED_HEADPHONES
        }
    }

    /** True if a Bluetooth A2DP or SCO device is connected. */
    fun isBluetoothDeviceConnected(): Boolean {
        val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
        return devices.any { device ->
            device.type == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP ||
                    device.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO
        }
    }

    /** Reset routing to platform default (restore mode NORMAL). */
    fun resetToDefault() {
        audioManager.mode = AudioManager.MODE_NORMAL
        audioManager.isSpeakerphoneOn = false
        Log.d(TAG, "Audio routing reset to default")
    }

    companion object {
        private const val TAG = "AudioRouter"
    }
}
