package com.jarvis.voice.notification

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.jarvis.voice.service.VoiceForegroundService

/**
 * Manages the persistent foreground notification for the voice service.
 *
 * The notification changes appearance based on the current voice state so the
 * user always knows what JARVIS is doing.  It also provides a "Stop" action
 * to end the listening session.
 */
class VoiceNotificationManager(private val context: Context) {

    private val notificationManager: NotificationManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    init {
        createNotificationChannel()
    }

    // ---- Notifications by state ----

    /**
     * Notification shown when the service is actively listening for voice input.
     */
    fun buildListeningNotification(): Notification {
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText("Listening\u2026")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                createStopPendingIntent(),
            )
            .setStyle(NotificationCompat.DecoratedCustomViewStyle())
            .build()
    }

    /**
     * Notification shown when JARVIS is speaking (TTS playback).
     * @param text The response text being spoken (will be truncated).
     */
    fun buildSpeakingNotification(text: String): Notification {
        val displayText = text.take(60).replace('\n', ' ')
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText(displayText.ifEmpty { "Speaking\u2026" })
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                createInterruptPendingIntent(),
            )
            .build()
    }

    /**
     * Notification shown when JARVIS is processing (thinking / STT).
     */
    fun buildProcessingNotification(label: String): Notification {
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText(label)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    /**
     * Notification shown when the service is idle but still alive.
     */
    fun buildIdleNotification(): Notification {
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText("Ready")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    /**
     * Notification shown when the service is offline.
     */
    fun buildOfflineNotification(): Notification {
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText("Offline \u2014 reconnecting\u2026")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    // ---- Internal helpers ----

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val channel = NotificationChannel(
            CHANNEL_ID,
            CHANNEL_NAME,
            NotificationManager.IMPORTANCE_LOW, // LOW so heads-up is suppressed
        ).apply {
            description = CHANNEL_DESCRIPTION
            setShowBadge(false)
            enableVibration(false)
            setSound(null, null)
        }

        notificationManager.createNotificationChannel(channel)
        Log.d(TAG, "Notification channel created: $CHANNEL_ID")
    }

    private fun createStopPendingIntent(): PendingIntent {
        val intent = Intent(context, VoiceForegroundService::class.java).apply {
            action = VoiceForegroundService.ACTION_STOP_LISTENING
        }
        return PendingIntent.getService(
            context,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
    }

    private fun createInterruptPendingIntent(): PendingIntent {
        val intent = Intent(context, VoiceForegroundService::class.java).apply {
            action = VoiceForegroundService.ACTION_INTERRUPT
        }
        return PendingIntent.getService(
            context,
            1,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
    }

    companion object {
        private const val TAG = "VoiceNotification"

        const val CHANNEL_ID = "jarvis_voice_channel"
        const val CHANNEL_NAME = "JARVIS Voice"
        const val CHANNEL_DESCRIPTION = "Voice assistant status"
        const val NOTIFICATION_ID = 2001
    }
}
