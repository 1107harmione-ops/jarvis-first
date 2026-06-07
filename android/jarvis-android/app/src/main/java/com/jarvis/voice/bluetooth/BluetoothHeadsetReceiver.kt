package com.jarvis.voice.bluetooth

import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothHeadset
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.jarvis.voice.service.VoiceForegroundService

/**
 * BroadcastReceiver for Bluetooth headset connection state changes.
 *
 * Declared in AndroidManifest.xml so the system wakes our app when a headset
 * connects or disconnects.  The receiver forwards relevant events to the
 * foreground service which can then start/stop SCO as appropriate.
 */
class BluetoothHeadsetReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return

        if (action != BluetoothHeadset.ACTION_CONNECTION_STATE_CHANGED) {
            return
        }

        val device: BluetoothDevice? =
            intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
        val state = intent.getIntExtra(
            BluetoothHeadset.EXTRA_STATE,
            BluetoothHeadset.STATE_DISCONNECTED,
        )
        val previousState = intent.getIntExtra(
            BluetoothHeadset.EXTRA_PREVIOUS_STATE,
            BluetoothHeadset.STATE_DISCONNECTED,
        )

        val deviceName = device?.name ?: "unknown"
        Log.i(TAG, "Headset connection state changed: $deviceName " +
                "${stateToString(previousState)} → ${stateToString(state)}")

        when (state) {
            BluetoothHeadset.STATE_CONNECTED -> {
                Log.i(TAG, "Bluetooth headset connected: $deviceName")
                // Notify the foreground service so it can start SCO voice routing.
                val serviceIntent = Intent(context, VoiceForegroundService::class.java)
                serviceIntent.action = VoiceForegroundService.ACTION_BLUETOOTH_CONNECTED
                serviceIntent.putExtra(VoiceForegroundService.EXTRA_DEVICE_NAME, deviceName)
                try {
                    context.startForegroundService(serviceIntent)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to start service on headset connect", e)
                }
            }

            BluetoothHeadset.STATE_DISCONNECTED -> {
                Log.i(TAG, "Bluetooth headset disconnected: $deviceName")
                val serviceIntent = Intent(context, VoiceForegroundService::class.java)
                serviceIntent.action = VoiceForegroundService.ACTION_BLUETOOTH_DISCONNECTED
                try {
                    context.startForegroundService(serviceIntent)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to start service on headset disconnect", e)
                }
            }

            BluetoothHeadset.STATE_CONNECTING,
            BluetoothHeadset.STATE_DISCONNECTING -> {
                // Transitional states — no action needed.
            }
        }
    }

    companion object {
        private const val TAG = "BtHeadsetReceiver"

        private fun stateToString(state: Int): String = when (state) {
            BluetoothHeadset.STATE_DISCONNECTED -> "DISCONNECTED"
            BluetoothHeadset.STATE_CONNECTING -> "CONNECTING"
            BluetoothHeadset.STATE_CONNECTED -> "CONNECTED"
            BluetoothHeadset.STATE_DISCONNECTING -> "DISCONNECTING"
            else -> "UNKNOWN($state)"
        }
    }
}
