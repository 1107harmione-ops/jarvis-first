package com.jarvis.voice.service

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Manages the binding between an Activity (or other UI component) and
 * [VoiceForegroundService].
 *
 * Use [bind] to start binding and [unbind] to release.  The [onConnected] /
 * [onDisconnected] callbacks inform the UI when the service is ready.
 */
class VoiceServiceConnection(
    private val context: Context,
    private val onConnected: (VoiceForegroundService) -> Unit = {},
    private val onDisconnected: () -> Unit = {},
) {

    private var boundService: VoiceForegroundService? = null
    private var isBound = false

    /** Reference to the bound service, or null if not bound. */
    val service: VoiceForegroundService? get() = boundService

    /** Whether the connection is currently bound. */
    val isConnected: Boolean get() = isBound && boundService != null

    /**
     * Start the service (foreground) and bind to it.
     *
     * Call this from Activity.onCreate() or onStart().
     */
    fun bind() {
        val intent = Intent(context, VoiceForegroundService::class.java)

        // Start the service first so it runs indefinitely (START_STICKY).
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start foreground service", e)
        }

        // Bind for direct communication.
        try {
            context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
            isBound = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind to service", e)
        }
    }

    /**
     * Unbind from the service.
     *
     * Call this from Activity.onStop() or onDestroy().
     * The service continues running until [VoiceForegroundService.stopService]
     * is called explicitly.
     */
    fun unbind() {
        if (isBound) {
            try {
                context.unbindService(connection)
            } catch (e: Exception) {
                Log.e(TAG, "Error unbinding service", e)
            }
            isBound = false
        }
        boundService = null
    }

    // ---- ServiceConnection implementation ----

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            Log.d(TAG, "Service connected: $name")
            if (binder is VoiceForegroundService.LocalBinder) {
                boundService = binder.getService()
                onConnected(boundService!!)
            }
        }

        override fun onServiceDisconnected(name: ComponentName) {
            Log.d(TAG, "Service disconnected: $name")
            boundService = null
            onDisconnected()
        }

        override fun onBindingDied(name: ComponentName) {
            Log.e(TAG, "Binding died: $name")
            boundService = null
            onDisconnected()
        }

        override fun onNullBinding(name: ComponentName) {
            Log.w(TAG, "Null binding: $name")
        }
    }

    companion object {
        private const val TAG = "VoiceServiceConn"
    }
}
