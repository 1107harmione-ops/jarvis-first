package com.jarvis.voice.state

import android.content.Context
import android.content.SharedPreferences
import com.jarvis.voice.model.VoiceConfig

/**
 * Persists voice settings to SharedPreferences.
 * This is the single source of truth for user-visible settings.
 */
class VoiceSettingsStore(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // ---- Getters with defaults matching VoiceConfig ----

    fun getServerUrl(): String =
        prefs.getString(KEY_SERVER_URL, "wss://jarvis-first-ko81.onrender.com/ws/voice") ?: "wss://jarvis-first-ko81.onrender.com/ws/voice"

    fun getLanguage(): String =
        prefs.getString(KEY_LANGUAGE, "en") ?: "en"

    fun getVoiceSpeed(): Float =
        prefs.getFloat(KEY_VOICE_SPEED, 1.0f)

    fun isWakeWordEnabled(): Boolean =
        prefs.getBoolean(KEY_WAKE_WORD_ENABLED, true)

    fun getWakeWordSensitivity(): Float =
        prefs.getFloat(KEY_WAKE_WORD_SENSITIVITY, 0.5f)

    fun getWakeWord(): String =
        prefs.getString(KEY_WAKE_WORD, "hey jarvis") ?: "hey jarvis"

    fun isAutoReconnectEnabled(): Boolean =
        prefs.getBoolean(KEY_AUTO_RECONNECT, true)

    fun getMaxReconnectAttempts(): Int =
        prefs.getInt(KEY_MAX_RECONNECT_ATTEMPTS, 20)

    fun isInterruptEnabled(): Boolean =
        prefs.getBoolean(KEY_INTERRUPT_ENABLED, true)

    fun isBluetoothScoEnabled(): Boolean =
        prefs.getBoolean(KEY_BLUETOOTH_SCO_ENABLED, true)

    fun getUiLanguage(): String =
        prefs.getString(KEY_UI_LANGUAGE, "en") ?: "en"

    fun getVoicePitch(): Float =
        prefs.getFloat(KEY_VOICE_PITCH, 1.0f)

    fun isOfflineModeEnabled(): Boolean =
        prefs.getBoolean(KEY_OFFLINE_MODE, false)

    fun isBatteryAwareEnabled(): Boolean =
        prefs.getBoolean(KEY_BATTERY_AWARE, true)

    fun isLowPowerListeningEnabled(): Boolean =
        prefs.getBoolean(KEY_LOW_POWER_LISTENING, true)

    /** Load full config from preferences. */
    fun loadConfig(): VoiceConfig = VoiceConfig(
        serverUrl = getServerUrl(),
        language = getLanguage(),
        uiLanguage = getUiLanguage(),
        voiceSpeed = getVoiceSpeed(),
        voicePitch = getVoicePitch(),
        wakeWordEnabled = isWakeWordEnabled(),
        wakeWordSensitivity = getWakeWordSensitivity(),
        wakeWord = getWakeWord(),
        autoReconnect = isAutoReconnectEnabled(),
        maxReconnectAttempts = getMaxReconnectAttempts(),
        interruptEnabled = isInterruptEnabled(),
        bluetoothScoEnabled = isBluetoothScoEnabled(),
        offlineModeEnabled = isOfflineModeEnabled(),
        batteryAwareEnabled = isBatteryAwareEnabled(),
        lowPowerListeningEnabled = isLowPowerListeningEnabled(),
    )

    // ---- Setters ----

    fun setServerUrl(url: String) {
        prefs.edit().putString(KEY_SERVER_URL, url).apply()
    }

    fun setLanguage(language: String) {
        prefs.edit().putString(KEY_LANGUAGE, language).apply()
    }

    fun setVoiceSpeed(speed: Float) {
        prefs.edit().putFloat(KEY_VOICE_SPEED, speed).apply()
    }

    fun setWakeWordEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_WAKE_WORD_ENABLED, enabled).apply()
    }

    fun setWakeWordSensitivity(sensitivity: Float) {
        prefs.edit().putFloat(KEY_WAKE_WORD_SENSITIVITY, sensitivity).apply()
    }

    fun setWakeWord(wakeWord: String) {
        prefs.edit().putString(KEY_WAKE_WORD, wakeWord).apply()
    }

    fun setAutoReconnect(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_AUTO_RECONNECT, enabled).apply()
    }

    fun setMaxReconnectAttempts(attempts: Int) {
        prefs.edit().putInt(KEY_MAX_RECONNECT_ATTEMPTS, attempts).apply()
    }

    fun setInterruptEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_INTERRUPT_ENABLED, enabled).apply()
    }

    fun setBluetoothScoEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_BLUETOOTH_SCO_ENABLED, enabled).apply()
    }

    fun setUiLanguage(lang: String) {
        prefs.edit().putString(KEY_UI_LANGUAGE, lang).apply()
    }

    fun setVoicePitch(pitch: Float) {
        prefs.edit().putFloat(KEY_VOICE_PITCH, pitch).apply()
    }

    fun setOfflineModeEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_OFFLINE_MODE, enabled).apply()
    }

    fun setBatteryAwareEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_BATTERY_AWARE, enabled).apply()
    }

    fun setLowPowerListeningEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_LOW_POWER_LISTENING, enabled).apply()
    }

    /** Persist the settings portion of a VoiceConfig. */
    fun saveConfig(config: VoiceConfig) {
        prefs.edit()
            .putString(KEY_SERVER_URL, config.serverUrl)
            .putString(KEY_LANGUAGE, config.language)
            .putString(KEY_UI_LANGUAGE, config.uiLanguage)
            .putFloat(KEY_VOICE_SPEED, config.voiceSpeed)
            .putFloat(KEY_VOICE_PITCH, config.voicePitch)
            .putBoolean(KEY_WAKE_WORD_ENABLED, config.wakeWordEnabled)
            .putFloat(KEY_WAKE_WORD_SENSITIVITY, config.wakeWordSensitivity)
            .putString(KEY_WAKE_WORD, config.wakeWord)
            .putBoolean(KEY_AUTO_RECONNECT, config.autoReconnect)
            .putInt(KEY_MAX_RECONNECT_ATTEMPTS, config.maxReconnectAttempts)
            .putBoolean(KEY_INTERRUPT_ENABLED, config.interruptEnabled)
            .putBoolean(KEY_BLUETOOTH_SCO_ENABLED, config.bluetoothScoEnabled)
            .putBoolean(KEY_OFFLINE_MODE, config.offlineModeEnabled)
            .putBoolean(KEY_BATTERY_AWARE, config.batteryAwareEnabled)
            .putBoolean(KEY_LOW_POWER_LISTENING, config.lowPowerListeningEnabled)
            .apply()
    }

    /** Clear all stored settings back to defaults. */
    fun clear() {
        prefs.edit().clear().apply()
    }

    companion object {
        private const val PREFS_NAME = "jarvis_voice_settings"

        private const val KEY_SERVER_URL = "server_url"
        private const val KEY_LANGUAGE = "language"
        private const val KEY_VOICE_SPEED = "voice_speed"
        private const val KEY_WAKE_WORD_ENABLED = "wake_word_enabled"
        private const val KEY_WAKE_WORD_SENSITIVITY = "wake_word_sensitivity"
        private const val KEY_WAKE_WORD = "wake_word"
        private const val KEY_AUTO_RECONNECT = "auto_reconnect"
        private const val KEY_MAX_RECONNECT_ATTEMPTS = "max_reconnect_attempts"
        private const val KEY_INTERRUPT_ENABLED = "interrupt_enabled"
        private const val KEY_BLUETOOTH_SCO_ENABLED = "bluetooth_sco_enabled"
        private const val KEY_UI_LANGUAGE = "ui_language"
        private const val KEY_VOICE_PITCH = "voice_pitch"
        private const val KEY_OFFLINE_MODE = "offline_mode"
        private const val KEY_BATTERY_AWARE = "battery_aware"
        private const val KEY_LOW_POWER_LISTENING = "low_power_listening"
    }
}
