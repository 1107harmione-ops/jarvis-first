package com.jarvis.voice.service

import android.app.Service
import android.bluetooth.BluetoothHeadset
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import com.jarvis.voice.audio.AudioRouter
import com.jarvis.voice.audio.EchoCanceller
import com.jarvis.voice.bluetooth.BluetoothVoiceManager
import com.jarvis.voice.capture.AudioCaptureService
import com.jarvis.voice.capture.AudioLevelMonitor
import com.jarvis.voice.client.VoiceWebSocketClient
import com.jarvis.voice.interrupt.InterruptController
import com.jarvis.voice.model.VoiceConfig
import com.jarvis.voice.model.VoiceMetrics
import com.jarvis.voice.notification.VoiceNotificationManager
import com.jarvis.voice.playback.AudioFocusManager
import com.jarvis.voice.playback.AudioPlayerService
import com.jarvis.voice.state.VoiceSettingsStore
import com.jarvis.voice.state.VoiceState
import com.jarvis.voice.state.VoiceStateHolder
import com.jarvis.voice.wakeword.WakeWordDetector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Foreground service that powers the JARVIS voice system.
 *
 * This service runs with `foregroundServiceType="microphone"` and holds
 * a persistent notification so the system does not kill it.  It orchestrates
 * audio capture, WebSocket streaming, playback, wake word detection, interrupt
 * handling, Bluetooth SCO, and audio focus.
 *
 * ## Intent actions
 *
 * - [ACTION_START_LISTENING] — begin microphone capture and connect to server.
 * - [ACTION_STOP_LISTENING] — stop capture, disconnect, and go idle.
 * - [ACTION_INTERRUPT] — interrupt current TTS playback.
 * - [ACTION_SET_LANGUAGE] — set server language for STT/TTS.
 * - [ACTION_SET_WAKE_WORD] — enable/disable on-device wake word.
 * - [ACTION_BLUETOOTH_CONNECTED] — internal; fired by [BluetoothHeadsetReceiver].
 * - [ACTION_BLUETOOTH_DISCONNECTED] — internal; fired by [BluetoothHeadsetReceiver].
 */
class VoiceForegroundService : Service() {

    // ---- Component instances (initialised in initializeComponents) ----
    private lateinit var audioCapture: AudioCaptureService
    private lateinit var audioLevelMonitor: AudioLevelMonitor
    private lateinit var audioPlayer: AudioPlayerService
    private lateinit var audioFocusManager: AudioFocusManager
    private lateinit var audioRouter: AudioRouter
    private lateinit var wsClient: VoiceWebSocketClient
    private lateinit var interruptController: InterruptController
    private lateinit var wakeWordDetector: WakeWordDetector
    private lateinit var bluetoothVoiceManager: BluetoothVoiceManager
    private lateinit var notificationManager: VoiceNotificationManager
    private lateinit var settingsStore: VoiceSettingsStore
    private lateinit var echoCanceller: EchoCanceller

    private var config: VoiceConfig = VoiceConfig()
    private var metrics: VoiceMetrics = VoiceMetrics()
    private var currentPartialText: String = ""

    // Coroutine scope for background tasks
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // Wake lock to keep CPU alive during capture
    private var wakeLock: PowerManager.WakeLock? = null

    // SCO state receiver
    private var scoReceiverRegistered = false

    // ---- Service lifecycle ----

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "VoiceForegroundService created")
        initializeComponents()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_LISTENING -> startListening()
            ACTION_STOP_LISTENING -> stopListening()
            ACTION_INTERRUPT -> interrupt()
            ACTION_SET_LANGUAGE -> {
                val language = intent.getStringExtra(EXTRA_LANGUAGE) ?: "en"
                setLanguage(language)
            }
            ACTION_SET_WAKE_WORD -> {
                val enabled = intent.getBooleanExtra(EXTRA_WAKE_WORD_ENABLED, true)
                setWakeWord(enabled)
            }
            ACTION_CONNECT -> connectToServer()
            ACTION_DISCONNECT -> disconnectFromServer()
            ACTION_BLUETOOTH_CONNECTED -> {
                val deviceName = intent.getStringExtra(EXTRA_DEVICE_NAME) ?: "unknown"
                onBluetoothHeadsetConnected(deviceName)
            }
            ACTION_BLUETOOTH_DISCONNECTED -> onBluetoothHeadsetDisconnected()
            else -> {
                // No action or unknown action — ensure service is running.
                if (intent?.action == null) {
                    Log.d(TAG, "Service started with no action, staying alive")
                }
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = LocalBinder()

    override fun onDestroy() {
        Log.i(TAG, "VoiceForegroundService destroyed")
        shutdown()
        super.onDestroy()
    }

    // ---- Binder for service connection ----

    inner class LocalBinder : android.os.Binder() {
        fun getService(): VoiceForegroundService = this@VoiceForegroundService
    }

    // ---- Initialisation ----

    private fun initializeComponents() {
        settingsStore = VoiceSettingsStore(this)
        config = settingsStore.loadConfig()

        audioCapture = AudioCaptureService()
        audioLevelMonitor = AudioLevelMonitor()
        audioPlayer = AudioPlayerService()
        audioFocusManager = AudioFocusManager(this)
        audioRouter = AudioRouter(this)
        bluetoothVoiceManager = BluetoothVoiceManager(this)
        notificationManager = VoiceNotificationManager(this)
        echoCanceller = EchoCanceller()

        wakeWordDetector = WakeWordDetector(
            wakeWord = config.wakeWord,
            sensitivity = config.wakeWordSensitivity,
        )

        wsClient = VoiceWebSocketClient(
            onStateChange = { state -> handleServerStateChange(state) },
            onTtsChunk = { chunk -> audioPlayer.playChunk(chunk) },
            onTtsStart = { onTtsStarted() },
            onTtsEnd = { onTtsEnded() },
            onTranscript = { text -> onTranscriptReceived(text) },
            onPartial = { text -> onPartialReceived(text) },
            onError = { msg -> onServerError(msg) },
        )
        wsClient.autoReconnect = config.autoReconnect
        wsClient.maxReconnectAttempts = config.maxReconnectAttempts

        interruptController = InterruptController(
            energyThreshold = config.interruptEnergyThreshold,
            onInterrupt = { onInterruptDetected() },
        )

        // Register for SCO state updates
        registerScoReceiver()

        // Observe state to update notification
        serviceScope.launch {
            VoiceStateHolder.state.collect { state ->
                updateNotificationForState(state)
            }
        }

        Log.i(TAG, "All components initialised")
    }

    // ---- Public command API (used by VoiceServiceConnection + intents) ----

    /** Start microphone capture and connect to the voice server. */
    fun startListening() {
        if (VoiceStateHolder.state.value == VoiceState.LISTENING) {
            Log.d(TAG, "Already listening")
            return
        }

        acquireWakeLock()
        connectToServer()

        val started = audioCapture.start { chunk ->
            // Apply echo cancellation if we are speaking
            val processed = if (audioPlayer.isPlaying) {
                // TODO: obtain reference signal from AudioPlayer for full AEC
                chunk
            } else {
                chunk
            }
            wsClient.sendAudio(processed)

            // On-device wake word detection (optional parallel path)
            if (config.wakeWordEnabled) {
                val result = wakeWordDetector.processChunk(chunk)
                if (result.detected) {
                    Log.i(TAG, "On-device wake word detected (score=${result.score})")
                }
            }
        }

        if (started) {
            audioLevelMonitor.start(audioCapture)
            VoiceStateHolder.updateState(VoiceState.LISTENING)
            Log.i(TAG, "Listening started")
        } else {
            VoiceStateHolder.updateState(VoiceState.ERROR)
            Log.e(TAG, "Failed to start audio capture")
        }
    }

    /** Stop microphone capture and disconnect from the server. */
    fun stopListening() {
        if (VoiceStateHolder.state.value == VoiceState.IDLE) return

        audioLevelMonitor.stop()
        audioCapture.stop()
        interruptController.stopMonitoring()
        audioPlayer.stop()
        disconnectFromServer()
        releaseWakeLock()
        wakeWordDetector.reset()
        VoiceStateHolder.updateState(VoiceState.IDLE)
        Log.i(TAG, "Listening stopped")
    }

    /** Interrupt current TTS playback and flush audio buffers. */
    fun interrupt() {
        Log.i(TAG, "Interrupt requested")
        wsClient.sendInterrupt()
        audioPlayer.flush()
        audioPlayer.stop()
        interruptController.stopMonitoring()
        audioCapture.restart()
        VoiceStateHolder.updateState(VoiceState.INTERRUPTED)
    }

    /** Update the server-side language setting. */
    fun setLanguage(language: String) {
        config = config.copy(language = language)
        wsClient.sendConfig(language, config.voiceSpeed)
        settingsStore.setLanguage(language)
        Log.i(TAG, "Language set to $language")
    }

    /** Enable or disable on-device wake word. */
    fun setWakeWord(enabled: Boolean) {
        config = config.copy(wakeWordEnabled = enabled)
        settingsStore.setWakeWordEnabled(enabled)
        Log.i(TAG, "Wake word ${if (enabled) "enabled" else "disabled"}")
    }

    /** Connect to the voice WebSocket server. */
    fun connectToServer() {
        wsClient.connect(config.serverUrl)
    }

    /** Disconnect from the voice WebSocket server. */
    fun disconnectFromServer() {
        wsClient.disconnect()
    }

    /** Update the voice speed. */
    fun setVoiceSpeed(speed: Float) {
        config = config.copy(voiceSpeed = speed)
        wsClient.sendConfig(config.language, speed)
        settingsStore.setVoiceSpeed(speed)
    }

    /** Reload config from persistent store. */
    fun reloadConfig() {
        config = settingsStore.loadConfig()
        wsClient.autoReconnect = config.autoReconnect
        wsClient.maxReconnectAttempts = config.maxReconnectAttempts
    }

    /** Access current config for UI binding. */
    fun getConfig(): VoiceConfig = config

    /** Snapshot of runtime metrics. */
    fun getMetrics(): VoiceMetrics = metrics

    /** Reference to the WebSocket client (for advanced use). */
    fun getWebSocketClient(): VoiceWebSocketClient = wsClient

    // ---- Internal event handlers ----

    private fun handleServerStateChange(state: VoiceState) {
        Log.d(TAG, "Server state change: $state")
        VoiceStateHolder.updateState(state)

        when (state) {
            VoiceState.SPEAKING -> {
                if (config.interruptEnabled) {
                    interruptController.startMonitoring()
                }
            }
            VoiceState.IDLE, VoiceState.INTERRUPTED, VoiceState.ERROR -> {
                interruptController.stopMonitoring()
            }
            else -> { /* no action */ }
        }
    }

    private fun onTtsStarted() {
        Log.d(TAG, "TTS started")
        audioPlayer.start()
        audioFocusManager.requestFocus()

        // Route to Bluetooth SCO if connected
        if (config.bluetoothScoEnabled && bluetoothVoiceManager.isHeadsetConnected()) {
            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            bluetoothVoiceManager.startSco(audioManager)
        }
    }

    private fun onTtsEnded() {
        Log.d(TAG, "TTS ended")
        audioFocusManager.abandonFocus()
        interruptController.stopMonitoring()
        audioPlayer.drainAndStop()

        // Stop Bluetooth SCO
        if (config.bluetoothScoEnabled) {
            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            bluetoothVoiceManager.stopSco(audioManager)
        }
    }

    private fun onInterruptDetected() {
        Log.i(TAG, "Voice interrupt detected during TTS")
        metrics = metrics.copy(interruptsTriggered = metrics.interruptsTriggered + 1)
        interrupt()
    }

    private fun onTranscriptReceived(text: String) {
        Log.d(TAG, "Transcript: $text")
        metrics = metrics.copy(commandsProcessed = metrics.commandsProcessed + 1)
        currentPartialText = ""
    }

    private fun onPartialReceived(text: String) {
        currentPartialText = text
    }

    private fun onServerError(msg: String) {
        Log.e(TAG, "Server error: $msg")
        metrics = metrics.copy(errorsEncountered = metrics.errorsEncountered + 1)
        VoiceStateHolder.updateState(VoiceState.ERROR)
    }

    // ---- Bluetooth headset events ----

    private fun onBluetoothHeadsetConnected(deviceName: String) {
        Log.i(TAG, "Bluetooth headset connected: $deviceName")
        if (config.bluetoothScoEnabled && VoiceStateHolder.state.value == VoiceState.LISTENING) {
            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            bluetoothVoiceManager.startSco(audioManager)
        }
    }

    private fun onBluetoothHeadsetDisconnected() {
        Log.i(TAG, "Bluetooth headset disconnected")
        audioRouter.resetToDefault()
        VoiceStateHolder.onBluetoothScoDisconnected()
    }

    // ---- Notification ----

    private fun updateNotificationForState(state: VoiceState) {
        val notification = when (state) {
            VoiceState.LISTENING -> notificationManager.buildListeningNotification()
            VoiceState.SPEAKING -> notificationManager.buildSpeakingNotification(currentPartialText)
            VoiceState.THINKING, VoiceState.STT_PROCESSING ->
                notificationManager.buildProcessingNotification(
                    if (state == VoiceState.THINKING) "Thinking\u2026" else "Processing\u2026"
                )
            VoiceState.OFFLINE -> notificationManager.buildOfflineNotification()
            VoiceState.ERROR -> notificationManager.buildProcessingNotification("Error")
            else -> notificationManager.buildIdleNotification()
        }

        try {
            val nm = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
            nm.notify(VoiceNotificationManager.NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update notification", e)
        }
    }

    // ---- Wake lock ----

    private fun acquireWakeLock() {
        if (wakeLock == null) {
            val powerManager = getSystemService(POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "jarvis:voice_wakelock"
            )
        }
        wakeLock?.let {
            if (!it.isHeld) {
                it.acquire(30_000L) // 30 second timeout to prevent runaway
                Log.d(TAG, "Wake lock acquired")
            }
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.d(TAG, "Wake lock released")
            }
        }
    }

    // ---- SCO receiver ----

    private fun registerScoReceiver() {
        if (scoReceiverRegistered) return
        try {
            registerReceiver(scoReceiver, IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED))
            scoReceiverRegistered = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to register SCO receiver", e)
        }
    }

    private fun unregisterScoReceiver() {
        if (!scoReceiverRegistered) return
        try {
            unregisterReceiver(scoReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister SCO receiver", e)
        }
        scoReceiverRegistered = false
    }

    private val scoReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val state = intent.getIntExtra(
                AudioManager.EXTRA_SCO_AUDIO_STATE,
                AudioManager.SCO_AUDIO_STATE_ERROR,
            )
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

    // ---- Shutdown ----

    private fun shutdown() {
        Log.i(TAG, "Shutting down voice service")
        interruptController.stopMonitoring()
        audioLevelMonitor.stop()
        audioCapture.stop()
        audioPlayer.stop()
        audioFocusManager.abandonFocus()
        wsClient.disconnect()
        unregisterScoReceiver()
        releaseWakeLock()
        serviceScope.cancel()
        VoiceStateHolder.updateState(VoiceState.IDLE)

        try {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping foreground", e)
        }
    }

    // ---- Companion / constants ----

    companion object {
        private const val TAG = "VoiceFgService"

        /** Intent actions for controlling the service. */
        const val ACTION_START_LISTENING = "com.jarvis.voice.START_LISTENING"
        const val ACTION_STOP_LISTENING = "com.jarvis.voice.STOP_LISTENING"
        const val ACTION_INTERRUPT = "com.jarvis.voice.INTERRUPT"
        const val ACTION_SET_LANGUAGE = "com.jarvis.voice.SET_LANGUAGE"
        const val ACTION_SET_WAKE_WORD = "com.jarvis.voice.SET_WAKE_WORD"
        const val ACTION_CONNECT = "com.jarvis.voice.CONNECT"
        const val ACTION_DISCONNECT = "com.jarvis.voice.DISCONNECT"
        const val ACTION_BLUETOOTH_CONNECTED = "com.jarvis.voice.BLUETOOTH_CONNECTED"
        const val ACTION_BLUETOOTH_DISCONNECTED = "com.jarvis.voice.BLUETOOTH_DISCONNECTED"

        /** Intent extras. */
        const val EXTRA_LANGUAGE = "extra_language"
        const val EXTRA_WAKE_WORD_ENABLED = "extra_wake_word_enabled"
        const val EXTRA_DEVICE_NAME = "extra_device_name"
    }
}
