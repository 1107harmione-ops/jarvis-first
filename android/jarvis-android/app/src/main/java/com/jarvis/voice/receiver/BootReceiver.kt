package com.jarvis.voice.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.jarvis.voice.service.VoiceForegroundService

/**
 * Boot completed receiver — auto-starts the voice foreground service
 * when the device boots up or after a system update.
 *
 * Requires:
 * <uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />
 */
class BootReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "BootReceiver"
    }

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_MY_PACKAGE_REPLACED -> {
                Log.i(TAG, "Boot completed / package replaced — starting voice service")
                startVoiceService(context)
            }
        }
    }

    private fun startVoiceService(context: Context) {
        try {
            val serviceIntent = Intent(context, VoiceForegroundService::class.java).apply {
                action = VoiceForegroundService.ACTION_CONNECT
            }
            context.startForegroundService(serviceIntent)
            Log.i(TAG, "VoiceForegroundService started after boot")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start voice service after boot: ${e.message}")
        }
    }
}
