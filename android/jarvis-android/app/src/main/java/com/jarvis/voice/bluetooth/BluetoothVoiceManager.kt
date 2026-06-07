package com.jarvis.voice.bluetooth

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothHeadset
import android.bluetooth.BluetoothProfile
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.os.Build
import android.util.Log
import com.jarvis.voice.state.VoiceStateHolder

/**
 * Manages Bluetooth SCO (Synchronous Connection Oriented) for voice communication.
 *
 * SCO provides a higher‑quality audio path optimised for voice, with better
 * echo cancellation and noise suppression than the default A2DP media path.
 */
class BluetoothVoiceManager(private val context: Context) {

    private var scoReceiverRegistered = false

    /**
     * Start Bluetooth SCO for voice communication.
     *
     * This puts the audio subsystem into communication mode and requests the
     * Bluetooth SCO link, routing both capture and playback through the headset.
     *
     * @param audioManager The system [AudioManager] instance.
     * @return true if SCO was successfully requested, false otherwise.
     */
    fun startSco(audioManager: AudioManager): Boolean {
        if (!isBluetoothAvailable()) {
            Log.w(TAG, "Bluetooth not available on this device")
            return false
        }

        if (!isHeadsetConnected()) {
            Log.w(TAG, "No Bluetooth headset connected — SCO not started")
            return false
        }

        try {
            // Switch to communication mode for optimal voice routing
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION

            // Register for SCO state updates
            registerScoReceiver()

            // Start SCO
            audioManager.startBluetoothSco()
            Log.i(TAG, "Bluetooth SCO requested")

            // The SCO connection is asynchronous; actual connection will be
            // reported via the BroadcastReceiver.
            return true
        } catch (e: SecurityException) {
            Log.e(TAG, "BLUETOOTH_CONNECT permission not granted", e)
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start Bluetooth SCO", e)
            return false
        }
    }

    /**
     * Stop Bluetooth SCO and restore normal audio mode.
     *
     * @param audioManager The system [AudioManager] instance.
     */
    fun stopSco(audioManager: AudioManager) {
        try {
            audioManager.stopBluetoothSco()
            audioManager.mode = AudioManager.MODE_NORMAL
            Log.i(TAG, "Bluetooth SCO stopped")
        } catch (e: SecurityException) {
            Log.e(TAG, "BLUETOOTH_CONNECT permission not granted", e)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop Bluetooth SCO", e)
        }

        unregisterScoReceiver()
    }

    /** Whether a Bluetooth SCO audio channel is currently active. */
    val isScoConnected: Boolean
        get() {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            return try {
                audioManager.isBluetoothScoOn
            } catch (e: Exception) {
                false
            }
        }

    /** Whether a Bluetooth headset is connected (any profile). */
    fun isHeadsetConnected(): Boolean {
        return try {
            val adapter = BluetoothAdapter.getDefaultAdapter() ?: return false
            // Check if SCO audio is already on as a proxy for headset connection
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            audioManager.isBluetoothScoOn
        } catch (e: SecurityException) {
            Log.e(TAG, "BLUETOOTH_CONNECT permission not granted", e)
            false
        }
    }

    // ---- Private helpers ----

    private fun isBluetoothAvailable(): Boolean {
        return BluetoothAdapter.getDefaultAdapter() != null
    }

    private val scoReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val state = intent.getIntExtra(
                AudioManager.EXTRA_SCO_AUDIO_STATE,
                AudioManager.SCO_AUDIO_STATE_ERROR,
            )
            Log.d(TAG, "SCO audio state changed: ${scoStateToString(state)}")

            when (state) {
                AudioManager.SCO_AUDIO_STATE_CONNECTED -> {
                    VoiceStateHolder.onBluetoothScoConnected()
                }
                AudioManager.SCO_AUDIO_STATE_DISCONNECTED -> {
                    VoiceStateHolder.onBluetoothScoDisconnected()
                }
            }
        }
    }

    private fun registerScoReceiver() {
        if (scoReceiverRegistered) return
        try {
            context.registerReceiver(scoReceiver, IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED))
            scoReceiverRegistered = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to register SCO receiver", e)
        }
    }

    private fun unregisterScoReceiver() {
        if (!scoReceiverRegistered) return
        try {
            context.unregisterReceiver(scoReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister SCO receiver", e)
        }
        scoReceiverRegistered = false
    }

    companion object {
        private const val TAG = "BtVoiceManager"

        private fun scoStateToString(state: Int): String = when (state) {
            AudioManager.SCO_AUDIO_STATE_CONNECTED -> "CONNECTED"
            AudioManager.SCO_AUDIO_STATE_CONNECTING -> "CONNECTING"
            AudioManager.SCO_AUDIO_STATE_DISCONNECTED -> "DISCONNECTED"
            AudioManager.SCO_AUDIO_STATE_ERROR -> "ERROR"
            else -> "UNKNOWN($state)"
        }
    }
}
