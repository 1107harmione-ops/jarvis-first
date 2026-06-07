package com.jarvis.voice.service

import android.app.Service
import android.bluetooth.BluetoothHeadset
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioManager
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import com.jarvis.voice.audio.AudioRouter
import com.jarvis.voice.audio.EchoCanceller
import com.jarvis.voice.bluetooth.BluetoothVoiceManager
import com.jarvis.voice.capture.AudioCaptureService
import com.jarvis.voice.capture.AudioLevelMonitor
import com.jarvis.voice.client.VoiceProtocol
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
 * ## Enhanced capabilities
 *
 * - **Offline mode**: queues commands locally for processing when connectivity
 *   is restored.
 * - **Multilingual UI**: independent server language (STT/TTS) and UI display
 *   language with localized notification text (English / Hindi).
 * - **Battery awareness**: adaptive wake-lock timeout, auto-stop on critical
 *   battery, low-power listening mode via [WakeWordDetector.enterLowPowerMode].
 *
 * ## Intent actions
 *
 * - [ACTION_START_LISTENING] — begin microphone capture and connect to server.
 * - [ACTION_STOP_LISTENING] — stop capture, disconnect, and go idle.
 * - [ACTION_INTERRUPT] — interrupt current TTS playback.
 * - [ACTION_SET_LANGUAGE] — set server language for STT/TTS.
 * - [ACTION_SET_UI_LANGUAGE] — set UI display language (independent of server).
 * - [ACTION_SET_WAKE_WORD] — enable/disable on-device wake word.
 * - [ACTION_SET_OFFLINE_MODE] — toggle offline/online mode.
 * - [ACTION_SET_VOICE_SPEED] — set TTS voice speed.
 * - [ACTION_PROCESS_QUEUE] — flush queued offline commands to the server.
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

    // ---- Enhanced state fields ----

    /** True when the WebSocket is disconnected and offline mode is active. */
    private var isOffline: Boolean = false

    /** UI language for notifications and display text (independent of server STT/TTS language). */
    private var uiLanguage: String = "en"

    /** True when battery is low / power saver is active. */
    private var isBatterySaverActive: Boolean = false

    /** True when running in reduced-power listening mode (offline + battery saver). */
    private var isLowPowerListening: Boolean = false

    /** Commands queued during offline mode, awaiting server reconnection. */
    private val queuedCommands = mutableListOf<QueuedCommand>()

    // Coroutine scope for background tasks
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // Wake lock to keep CPU alive during capture
    private var wakeLock: PowerManager.WakeLock? = null

    // SCO state receiver
    private var scoReceiverRegistered = false

    // Battery / power state receiver
    private var batteryReceiverRegistered = false

    // ---- Service lifecycle ----

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "VoiceForegroundService created")
        initializeComponents()
        registerBatteryReceiver()
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
            ACTION_SET_UI_LANGUAGE -> {
                val lang = intent.getStringExtra(EXTRA_UI_LANGUAGE) ?: "en"
                setUiLanguage(lang)
            }
            ACTION_SET_WAKE_WORD -> {
                val enabled = intent.getBooleanExtra(EXTRA_WAKE_WORD_ENABLED, true)
                setWakeWord(enabled)
            }
            ACTION_SET_OFFLINE_MODE -> {
                val enabled = intent.getBooleanExtra(EXTRA_OFFLINE_MODE, false)
                setOfflineMode(enabled)
            }
            ACTION_SET_VOICE_SPEED -> {
                val speed = intent.getFloatExtra(EXTRA_VOICE_SPEED, 1.0f)
                setVoiceSpeed(speed)
            }
            ACTION_CONNECT -> connectToServer()
            ACTION_DISCONNECT -> disconnectFromServer()
            ACTION_PROCESS_QUEUE -> processCommandQueue()
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
        uiLanguage = config.uiLanguage
        // Initialise state holder with loaded values
        VoiceStateHolder.updateServerLanguage(config.language)
        VoiceStateHolder.updateUiLanguage(uiLanguage)
        VoiceStateHolder.updateOfflineState(false)
        VoiceStateHolder.updateQueuedCommandCount(0)

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

        // Battery-aware: refuse to start if battery is critically low
        if (config.batteryAwareEnabled && (isBatterySaverActive || isBatteryCritical())) {
            Log.w(TAG, "Battery too low to start listening")
            VoiceStateHolder.updateState(VoiceState.ERROR)
            showBatteryLowNotification()
            return
        }

        acquireAdaptiveWakeLock()

        // Offline mode: start local-only listening (wake word detection, command queuing)
        if (isOffline || config.offlineModeEnabled) {
            startOfflineListening()
            return
        }

        connectToServer()

        val started = audioCapture.start { chunk ->
            // Apply echo cancellation if we are speaking
            val processed = if (audioPlayer.isPlaying) {
                // TODO: obtain reference signal from AudioPlayer for full AEC
                chunk
            } else {
                chunk
            }

            // Send audio to server unless in offline mode
            if (!isOffline) {
                wsClient.sendAudio(processed)
            }

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
        if (!isOffline) {
            disconnectFromServer()
        }
        releaseAdaptiveWakeLock()
        wakeWordDetector.reset()
        isLowPowerListening = false
        VoiceStateHolder.updateState(VoiceState.IDLE)
        Log.i(TAG, "Listening stopped")
    }

    /** Interrupt current TTS playback and flush audio buffers. */
    fun interrupt() {
        Log.i(TAG, "Interrupt requested")
        if (!isOffline) {
            wsClient.sendInterrupt()
        }
        audioPlayer.flush()
        audioPlayer.stop()
        interruptController.stopMonitoring()
        audioCapture.restart()
        VoiceStateHolder.updateState(VoiceState.INTERRUPTED)
    }

    /** Update the server-side STT/TTS language. */
    fun setLanguage(language: String) {
        config = config.copy(language = language)
        if (!isOffline) {
            wsClient.sendConfig(
                language = language,
                voiceSpeed = config.voiceSpeed,
                voicePitch = config.voicePitch,
            )
        }
        settingsStore.setLanguage(language)
        VoiceStateHolder.updateServerLanguage(language)
        Log.i(TAG, "Language set to $language")
    }

    /** Set the UI display language (independent of server STT/TTS language). */
    fun setUiLanguage(lang: String) {
        uiLanguage = lang
        config = config.copy(uiLanguage = lang)
        settingsStore.setUiLanguage(lang)
        VoiceStateHolder.updateUiLanguage(lang)
        Log.i(TAG, "UI language set to $lang")
    }

    /** Enable or disable on-device wake word. */
    fun setWakeWord(enabled: Boolean) {
        config = config.copy(wakeWordEnabled = enabled)
        settingsStore.setWakeWordEnabled(enabled)
        Log.i(TAG, "Wake word ${if (enabled) "enabled" else "disabled"}")
    }

    /** Toggle offline mode. When enabled, commands are queued locally. */
    fun setOfflineMode(enabled: Boolean) {
        config = config.copy(offlineModeEnabled = enabled)
        settingsStore.setOfflineModeEnabled(enabled)
        isOffline = enabled
        VoiceStateHolder.updateOfflineState(enabled)
        if (enabled) {
            disconnectFromServer()
            VoiceStateHolder.updateState(VoiceState.OFFLINE)
        } else {
            connectToServer()
            // Process any queued commands when coming back online
            processCommandQueue()
        }
        Log.i(TAG, "Offline mode ${if (enabled) "enabled" else "disabled"}")
    }

    /** Connect to the voice WebSocket server. */
    fun connectToServer() {
        isOffline = false
        wsClient.connect(config.serverUrl)
    }

    /** Disconnect from the voice WebSocket server. */
    fun disconnectFromServer() {
        isOffline = false
        wsClient.disconnect()
    }

    /** Update the voice speed and sync to the server. */
    fun setVoiceSpeed(speed: Float) {
        config = config.copy(voiceSpeed = speed)
        if (!isOffline) {
            wsClient.sendConfig(
                language = config.language,
                voiceSpeed = speed,
                voicePitch = config.voicePitch,
            )
        }
        settingsStore.setVoiceSpeed(speed)
    }

    /** Reload config from persistent store. */
    fun reloadConfig() {
        config = settingsStore.loadConfig()
        uiLanguage = config.uiLanguage
        wsClient.autoReconnect = config.autoReconnect
        wsClient.maxReconnectAttempts = config.maxReconnectAttempts
    }

    /** Access current config for UI binding. */
    fun getConfig(): VoiceConfig = config

    /** Snapshot of runtime metrics. */
    fun getMetrics(): VoiceMetrics = metrics

    /** Reference to the WebSocket client (for advanced use). */
    fun getWebSocketClient(): VoiceWebSocketClient = wsClient

    /** Number of commands queued during offline mode. */
    fun getQueuedCommandCount(): Int = queuedCommands.size

    /** Whether the service is currently in offline mode. */
    fun isOfflineMode(): Boolean = isOffline || config.offlineModeEnabled

    /** Current UI display language. */
    fun getUiLanguage(): String = uiLanguage

    // ---- Offline mode ----

    /**
     * Start a local-only listening session.
     *
     * Audio is processed by the on-device wake word detector only.  Commands
     * are queued for later processing when the server reconnects.
     */
    private fun startOfflineListening() {
        Log.i(TAG, "Starting offline listening mode")
        isLowPowerListening = true

        // Enter low-power wake word mode if battery saver is active
        if (isBatterySaverActive) {
            wakeWordDetector.enterLowPowerMode()
        }

        val started = audioCapture.start { chunk ->
            // On-device wake word detection only
            if (config.wakeWordEnabled) {
                val result = wakeWordDetector.processChunk(chunk)
                if (result.detected) {
                    Log.i(TAG, "Offline wake word detected")
                }
            }
        }

        if (started) {
            audioLevelMonitor.start(audioCapture)
            VoiceStateHolder.updateState(VoiceState.LISTENING)
            Log.i(TAG, "Offline listening started")
        } else {
            VoiceStateHolder.updateState(VoiceState.ERROR)
            Log.e(TAG, "Failed to start offline audio capture")
        }
    }

    /**
     * Queue a command during offline mode.
     *
     * The command will be sent to the server once connectivity is restored
     * (either via [processCommandQueue] or automatic reconnect flow).
     *
     * @param transcript The spoken command text.
     * @param language The language of the command (defaults to current config language).
     */
    fun queueCommand(transcript: String, language: String = config.language) {
        if (!isOffline) {
            Log.w(TAG, "Not offline, command will be sent directly")
            return
        }
        val command = QueuedCommand(
            transcript = transcript,
            language = language,
            timestamp = System.currentTimeMillis(),
        )
        queuedCommands.add(command)
        VoiceStateHolder.updateQueuedCommandCount(queuedCommands.size)
        Log.i(TAG, "Command queued (total: ${queuedCommands.size}): $transcript")
        // Keep the offline state visible
        VoiceStateHolder.updateState(VoiceState.OFFLINE)
    }

    /**
     * Process all queued commands by sending them to the server.
     *
     * Called automatically when connectivity is restored after an offline period.
     * Can also be triggered manually via [ACTION_PROCESS_QUEUE].
     */
    fun processCommandQueue() {
        if (queuedCommands.isEmpty()) {
            Log.d(TAG, "No queued commands to process")
            return
        }

        if (isOffline) {
            Log.w(TAG, "Still offline, cannot process queue yet")
            return
        }

        val batch = queuedCommands.toList()
        queuedCommands.clear()
        VoiceStateHolder.updateQueuedCommandCount(0)
        Log.i(TAG, "Processing ${batch.size} queued commands")

        serviceScope.launch {
            for ((index, cmd) in batch.withIndex()) {
                if (!isActive) break
                Log.d(TAG, "Sending queued command ${index + 1}/${batch.size}: ${cmd.transcript}")
                // Send the language so the server processes this command correctly
                wsClient.sendConfig(
                    language = cmd.language,
                    voiceSpeed = config.voiceSpeed,
                )
                // TODO: In a full implementation, this would replay the full audio
                // or send the transcript for NL processing.  Currently we just
                // restore the language context.
                delay(300) // Brief pause between commands
            }
            Log.i(TAG, "Finished processing queued commands")
        }
    }

    /** Clear all queued commands without processing them. */
    fun clearCommandQueue() {
        queuedCommands.clear()
        VoiceStateHolder.updateQueuedCommandCount(0)
        Log.i(TAG, "Queued commands cleared")
    }

    // ---- Battery-aware features ----

    /**
     * Acquire wake lock with adaptive timeout based on current power state.
     *
     * Uses longer timeouts during active listening and shorter timeouts when
     * battery saver is active to conserve power.
     */
    private fun acquireAdaptiveWakeLock() {
        if (wakeLock == null) {
            val powerManager = getSystemService(POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "jarvis:voice_wakelock"
            )
        }
        wakeLock?.let {
            if (!it.isHeld) {
                val timeout = when {
                    isBatterySaverActive -> 15_000L  // Very short during battery saver
                    isLowPowerListening -> 20_000L    // Short during offline listening
                    else -> 45_000L                   // Normal active listening
                }
                it.acquire(timeout)
                Log.d(TAG, "Wake lock acquired (timeout=${timeout}ms)")
            }
        }
    }

    private fun releaseAdaptiveWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.d(TAG, "Wake lock released")
            }
        }
    }

    /**
     * Check if battery level is critically low (< 10 %).
     */
    private fun isBatteryCritical(): Boolean {
        if (!config.batteryAwareEnabled) return false
        val batteryManager = getSystemService(BATTERY_SERVICE) as? BatteryManager ?: return false
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            val level = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            level in 0..9
        } else false
    }

    private fun showBatteryLowNotification() {
        try {
            val nm = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
            val notification = android.app.Notification.Builder(this, VoiceNotificationManager.CHANNEL_ID)
                .setContentTitle("JARVIS")
                .setContentText(getUiString("battery_low"))
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setOngoing(false)
                .setPriority(android.app.Notification.PRIORITY_HIGH)
                .build()
            nm.notify(1002, notification)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to show battery notification", e)
        }
    }

    // ---- Battery / power state receiver ----

    private fun registerBatteryReceiver() {
        if (batteryReceiverRegistered) return
        try {
            val filter = IntentFilter().apply {
                addAction(Intent.ACTION_BATTERY_LOW)
                addAction(Intent.ACTION_BATTERY_OKAY)
                addAction(Intent.ACTION_POWER_CONNECTED)
                addAction(Intent.ACTION_POWER_DISCONNECTED)
            }
            registerReceiver(batteryReceiver, filter)
            batteryReceiverRegistered = true
            Log.d(TAG, "Battery receiver registered")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to register battery receiver", e)
        }
    }

    private fun unregisterBatteryReceiver() {
        if (!batteryReceiverRegistered) return
        try {
            unregisterReceiver(batteryReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to unregister battery receiver", e)
        }
        batteryReceiverRegistered = false
    }

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                Intent.ACTION_BATTERY_LOW -> {
                    Log.w(TAG, "Battery low — entering power save mode")
                    isBatterySaverActive = true
                    onBatteryLow()
                }
                Intent.ACTION_BATTERY_OKAY -> {
                    Log.i(TAG, "Battery okay — exiting power save mode")
                    isBatterySaverActive = false
                    onBatteryOkay()
                }
                Intent.ACTION_POWER_CONNECTED -> {
                    Log.i(TAG, "Power connected")
                    if (isBatterySaverActive) {
                        isBatterySaverActive = false
                        onBatteryOkay()
                    }
                }
                Intent.ACTION_POWER_DISCONNECTED -> {
                    Log.i(TAG, "Power disconnected")
                    if (isBatteryCritical()) {
                        isBatterySaverActive = true
                        onBatteryLow()
                    }
                }
            }
        }
    }

    private fun onBatteryLow() {
        VoiceStateHolder.updateBatterySaverState(true)
        metrics = metrics.copy(
            batterySaverActivations = metrics.batterySaverActivations + 1,
        )
        // Stop active listening if battery is critical
        if (isBatteryCritical() && VoiceStateHolder.state.value == VoiceState.LISTENING) {
            Log.w(TAG, "Critical battery — stopping listening")
            stopListening()
            return
        }
        // Switch to low-power wake word mode
        wakeWordDetector.enterLowPowerMode()
        // Release and re-acquire with shorter timeout
        releaseAdaptiveWakeLock()
        Log.d(TAG, "Entered low-power mode due to battery state")
    }

    private fun onBatteryOkay() {
        VoiceStateHolder.updateBatterySaverState(false)
        wakeWordDetector.exitLowPowerMode()
        Log.d(TAG, "Exited low-power mode — battery is okay")
    }

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
            VoiceState.OFFLINE -> {
                isOffline = true
                VoiceStateHolder.updateOfflineState(true)
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
            VoiceState.LISTENING -> {
                if (isOffline || config.offlineModeEnabled) {
                    notificationManager.buildProcessingNotification(getUiString("listening_offline"))
                } else {
                    notificationManager.buildListeningNotification()
                }
            }
            VoiceState.SPEAKING -> notificationManager.buildSpeakingNotification(currentPartialText)
            VoiceState.THINKING, VoiceState.STT_PROCESSING ->
                notificationManager.buildProcessingNotification(
                    getUiString(if (state == VoiceState.THINKING) "thinking" else "processing")
                )
            VoiceState.OFFLINE -> {
                val text = if (queuedCommands.isNotEmpty()) {
                    getUiString("offline_queued").format(queuedCommands.size)
                } else {
                    getUiString("offline")
                }
                notificationManager.buildProcessingNotification(text)
            }
            VoiceState.ERROR -> notificationManager.buildProcessingNotification(getUiString("error"))
            else -> notificationManager.buildIdleNotification()
        }

        try {
            val nm = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
            nm.notify(VoiceNotificationManager.NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update notification", e)
        }
    }

    /**
     * Return a localized UI string for the given key.
     *
     * Supports English (en) and Hindi (hi). Falls back to the key name
     * when the key is unknown.
     */
    private fun getUiString(key: String): String {
        return when (uiLanguage) {
            "hi" -> when (key) {
                "listening_offline" -> "\u0938\u094D\u0925\u093E\u0928\u0940\u092F \u0938\u0941\u0928\u0928\u093E"
                "thinking" -> "\u0938\u094B\u091A \u0930\u0939\u093E \u0939\u0948\u2026"
                "processing" -> "\u0938\u0902\u0938\u093E\u0927\u093F\u0924 \u0915\u0930 \u0930\u0939\u093E \u0939\u0948\u2026"
                "offline" -> "\u0911\u092B\u0932\u093E\u0907\u0928"
                "offline_queued" -> "\u0911\u092B\u0932\u093E\u0907\u0928 \u2014 %d \u090F\u0915 \u0915\u0924\u093E\u0930 \u092E\u0947\u0902 \u0939\u0948\u0902"
                "error" -> "\u0924\u094D\u0930\u0941\u091F\u093F"
                "battery_low" -> "\u092C\u0948\u091F\u0930\u0940 \u092C\u0939\u0941\u0924 \u0915\u092E \u0939\u0948"
                "idle" -> "\u0924\u0948\u092F\u093E\u0930"
                else -> key
            }
            else -> when (key) {
                "listening_offline" -> "Local listening"
                "thinking" -> "Thinking\u2026"
                "processing" -> "Processing\u2026"
                "offline" -> "Offline \u2014 reconnecting\u2026"
                "offline_queued" -> "Offline \u2014 %d queued"
                "error" -> "Error"
                "battery_low" -> "Battery too low"
                "idle" -> "Ready"
                else -> key
            }
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
        unregisterBatteryReceiver()
        releaseAdaptiveWakeLock()
        serviceScope.cancel()
        queuedCommands.clear()
        VoiceStateHolder.updateState(VoiceState.IDLE)

        try {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping foreground", e)
        }
    }

    // ---- Data class for queued offline commands ----

    /**
     * A command queued during offline mode for later server processing.
     *
     * @property transcript The spoken command text.
     * @property language The language of the command.
     * @property timestamp When the command was queued (System.currentTimeMillis).
     */
    data class QueuedCommand(
        val transcript: String,
        val language: String,
        val timestamp: Long,
    )

    // ---- Companion / constants ----

    companion object {
        private const val TAG = "VoiceFgService"

        /** Intent actions for controlling the service. */
        const val ACTION_START_LISTENING = "com.jarvis.voice.START_LISTENING"
        const val ACTION_STOP_LISTENING = "com.jarvis.voice.STOP_LISTENING"
        const val ACTION_INTERRUPT = "com.jarvis.voice.INTERRUPT"
        const val ACTION_SET_LANGUAGE = "com.jarvis.voice.SET_LANGUAGE"
        const val ACTION_SET_UI_LANGUAGE = "com.jarvis.voice.SET_UI_LANGUAGE"
        const val ACTION_SET_WAKE_WORD = "com.jarvis.voice.SET_WAKE_WORD"
        const val ACTION_SET_OFFLINE_MODE = "com.jarvis.voice.SET_OFFLINE_MODE"
        const val ACTION_SET_VOICE_SPEED = "com.jarvis.voice.SET_VOICE_SPEED"
        const val ACTION_CONNECT = "com.jarvis.voice.CONNECT"
        const val ACTION_DISCONNECT = "com.jarvis.voice.DISCONNECT"
        const val ACTION_PROCESS_QUEUE = "com.jarvis.voice.PROCESS_QUEUE"
        const val ACTION_BLUETOOTH_CONNECTED = "com.jarvis.voice.BLUETOOTH_CONNECTED"
        const val ACTION_BLUETOOTH_DISCONNECTED = "com.jarvis.voice.BLUETOOTH_DISCONNECTED"

        /** Intent extras. */
        const val EXTRA_LANGUAGE = "extra_language"
        const val EXTRA_UI_LANGUAGE = "extra_ui_language"
        const val EXTRA_WAKE_WORD_ENABLED = "extra_wake_word_enabled"
        const val EXTRA_OFFLINE_MODE = "extra_offline_mode"
        const val EXTRA_VOICE_SPEED = "extra_voice_speed"
        const val EXTRA_DEVICE_NAME = "extra_device_name"
    }
}
