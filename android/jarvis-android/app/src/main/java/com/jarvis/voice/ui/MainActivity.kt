package com.jarvis.voice.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.ToggleButton
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.jarvis.voice.R
import com.jarvis.voice.service.VoiceForegroundService
import com.jarvis.voice.state.VoiceState
import com.jarvis.voice.state.VoiceStateHolder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Main voice interaction activity for JARVIS.
 *
 * Shows:
 * - Voice wave animation (audio level visualization)
 * - Real-time transcript display
 * - State indicator (Listening, Speaking, Thinking, etc.)
 * - Language toggle (Hindi/English for both server and UI)
 * - Offline mode toggle
 * - Interrupt button
 * - Connection and battery status
 * - Queued command count during offline mode
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    // ── UI Components ────────────────────────────────────────────

    private lateinit var stateText: TextView
    private lateinit var transcriptText: TextView
    private lateinit var responseText: TextView
    private lateinit var audioLevelView: AudioLevelView
    private lateinit var interruptButton: ImageButton
    private lateinit var languageToggle: ToggleButton
    private lateinit var wakeWordToggle: ToggleButton
    private lateinit var offlineModeToggle: ToggleButton
    private lateinit var uiLanguageToggle: ToggleButton
    private lateinit var connectionStatus: TextView
    private lateinit var offlineIndicator: TextView
    private lateinit var queuedCountText: TextView
    private lateinit var batteryStatus: TextView

    // ── Service Connection ──────────────────────────────────────

    private var voiceService: VoiceForegroundService? = null
    private var isBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? VoiceForegroundService.LocalBinder
            voiceService = binder?.getService()
            isBound = true
            Log.d(TAG, "Bound to VoiceForegroundService")
            updateConnectionStatus(true)
            // Sync toggles with service state
            voiceService?.let { svc ->
                val isHi = svc.getConfig().language == "hi"
                languageToggle.isChecked = isHi
                offlineModeToggle.isChecked = svc.isOfflineMode()
                val uiIsHi = svc.getUiLanguage() == "hi"
                uiLanguageToggle.isChecked = uiIsHi
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            voiceService = null
            isBound = false
            Log.d(TAG, "Unbound from VoiceForegroundService")
            updateConnectionStatus(false)
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupListeners()
        observeState()
        bindVoiceService()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
    }

    // ── Initialization ──────────────────────────────────────────

    private fun initViews() {
        stateText = findViewById(R.id.stateText)
        transcriptText = findViewById(R.id.transcriptText)
        responseText = findViewById(R.id.responseText)
        audioLevelView = findViewById(R.id.audioLevelView)
        interruptButton = findViewById(R.id.interruptButton)
        languageToggle = findViewById(R.id.languageToggle)
        wakeWordToggle = findViewById(R.id.wakeWordToggle)
        offlineModeToggle = findViewById(R.id.offlineModeToggle)
        uiLanguageToggle = findViewById(R.id.uiLanguageToggle)
        connectionStatus = findViewById(R.id.connectionStatus)
        offlineIndicator = findViewById(R.id.offlineIndicator)
        queuedCountText = findViewById(R.id.queuedCountText)
        batteryStatus = findViewById(R.id.batteryStatus)
    }

    private fun setupListeners() {
        interruptButton.setOnClickListener {
            voiceService?.interrupt()
            Log.d(TAG, "User triggered interrupt")
        }

        // Server STT/TTS language toggle (EN ↔ HI)
        languageToggle.setOnCheckedChangeListener { _, isChecked ->
            val language = if (isChecked) "hi" else "en"
            voiceService?.setLanguage(language)
            Log.d(TAG, "Server language changed to: $language")
        }

        // UI display language toggle (independent of server language)
        uiLanguageToggle.setOnCheckedChangeListener { _, isChecked ->
            val lang = if (isChecked) "hi" else "en"
            voiceService?.setUiLanguage(lang)
            Log.d(TAG, "UI language changed to: $lang")
        }

        wakeWordToggle.setOnCheckedChangeListener { _, isChecked ->
            voiceService?.setWakeWord(isChecked)
            Log.d(TAG, "Wake word ${if (isChecked) "enabled" else "disabled"}")
        }

        offlineModeToggle.setOnCheckedChangeListener { _, isChecked ->
            voiceService?.setOfflineMode(isChecked)
            Log.d(TAG, "Offline mode ${if (isChecked) "enabled" else "disabled"}")
        }
    }

    private fun observeState() {
        // Observe state changes from VoiceStateHolder
        lifecycleScope.launch {
            VoiceStateHolder.state.collect { state ->
                withContext(Dispatchers.Main) {
                    updateStateUI(state)
                }
            }
        }

        lifecycleScope.launch {
            VoiceStateHolder.transcript.collect { text ->
                withContext(Dispatchers.Main) {
                    transcriptText.text = text
                }
            }
        }

        lifecycleScope.launch {
            VoiceStateHolder.partialText.collect { text ->
                withContext(Dispatchers.Main) {
                    // Show partial transcript below the final transcript
                    if (text.isNotEmpty()) {
                        responseText.text = text
                    }
                }
            }
        }

        lifecycleScope.launch {
            VoiceStateHolder.audioLevel.collect { level ->
                withContext(Dispatchers.Main) {
                    audioLevelView.setLevel(level)
                }
            }
        }

        // Observe offline state
        lifecycleScope.launch {
            VoiceStateHolder.isOffline.collect { offline ->
                withContext(Dispatchers.Main) {
                    offlineIndicator.visibility = if (offline) android.view.View.VISIBLE
                        else android.view.View.GONE
                    offlineModeToggle.isChecked = offline
                }
            }
        }

        // Observe queued command count
        lifecycleScope.launch {
            VoiceStateHolder.queuedCommandCount.collect { count ->
                withContext(Dispatchers.Main) {
                    queuedCountText.text = if (count > 0) {
                        "Queued: $count"
                    } else {
                        ""
                    }
                    queuedCountText.visibility = if (count > 0) android.view.View.VISIBLE
                        else android.view.View.GONE
                }
            }
        }

        // Observe battery saver state
        lifecycleScope.launch {
            VoiceStateHolder.isBatterySaverActive.collect { active ->
                withContext(Dispatchers.Main) {
                    batteryStatus.text = if (active) "Battery Saver" else ""
                    batteryStatus.visibility = if (active) android.view.View.VISIBLE
                        else android.view.View.GONE
                }
            }
        }

        // Observe server language changes to sync toggle
        lifecycleScope.launch {
            VoiceStateHolder.serverLanguage.collect { lang ->
                withContext(Dispatchers.Main) {
                    languageToggle.isChecked = lang == "hi"
                }
            }
        }

        // Observe UI language changes to sync toggle
        lifecycleScope.launch {
            VoiceStateHolder.uiLanguage.collect { lang ->
                withContext(Dispatchers.Main) {
                    uiLanguageToggle.isChecked = lang == "hi"
                }
            }
        }
    }

    private fun bindVoiceService() {
        val intent = Intent(this, VoiceForegroundService::class.java)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    // ── UI Updates ──────────────────────────────────────────────

    private fun updateStateUI(state: VoiceState) {
        stateText.text = when (state) {
            VoiceState.IDLE -> getString(R.string.idle_text)
            VoiceState.LISTENING -> getString(R.string.listening_text)
            VoiceState.SPEAKING -> getString(R.string.speaking_text)
            VoiceState.THINKING -> getString(R.string.thinking_text)
            VoiceState.STT_PROCESSING -> getString(R.string.processing_text)
            VoiceState.INTERRUPTED -> "Interrupted"
            VoiceState.OFFLINE -> getString(R.string.offline_text)
            VoiceState.ERROR -> getString(R.string.error_text)
        }

        // Update visual indicators
        interruptButton.isEnabled = state == VoiceState.SPEAKING || state == VoiceState.THINKING
        offlineIndicator.visibility = if (state == VoiceState.OFFLINE) android.view.View.VISIBLE
            else android.view.View.GONE
    }

    private fun updateConnectionStatus(connected: Boolean) {
        connectionStatus.text = if (connected) "Connected" else "Disconnected"
        connectionStatus.setTextColor(
            if (connected) 0xFF4CAF50.toInt()
            else 0xFFFF5252.toInt()
        )
    }
}
