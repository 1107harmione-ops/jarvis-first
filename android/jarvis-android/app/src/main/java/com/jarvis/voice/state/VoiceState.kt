package com.jarvis.voice.state

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Voice service operational states.
 * Mirrors the server-side state machine for consistent UI feedback.
 */
enum class VoiceState {
    IDLE,
    LISTENING,
    STT_PROCESSING,
    THINKING,
    SPEAKING,
    INTERRUPTED,
    OFFLINE,
    ERROR;

    companion object {
        /** Map server-side state strings to the local enum. */
        fun fromString(s: String?): VoiceState = when (s) {
            "idle" -> IDLE
            "listening" -> LISTENING
            "stt_processing" -> STT_PROCESSING
            "thinking" -> THINKING
            "speaking" -> SPEAKING
            "interrupted" -> INTERRUPTED
            "offline" -> OFFLINE
            else -> ERROR
        }
    }
}

/**
 * Singleton observable state holder for the voice system.
 * UI components can collect these flows to react to state changes.
 */
object VoiceStateHolder {
    private val _state = MutableStateFlow(VoiceState.IDLE)
    val state: StateFlow<VoiceState> = _state.asStateFlow()

    private val _audioLevel = MutableStateFlow(0f)
    val audioLevel: StateFlow<Float> = _audioLevel.asStateFlow()

    private val _partialText = MutableStateFlow("")
    val partialText: StateFlow<String> = _partialText.asStateFlow()

    private val _transcript = MutableStateFlow("")
    val transcript: StateFlow<String> = _transcript.asStateFlow()

    private val _isBluetoothScoConnected = MutableStateFlow(false)
    val isBluetoothScoConnected: StateFlow<Boolean> = _isBluetoothScoConnected.asStateFlow()

    private val _audioFocusState = MutableStateFlow(AudioFocusState.NORMAL)
    val audioFocusState: StateFlow<AudioFocusState> = _audioFocusState.asStateFlow()

    fun updateState(newState: VoiceState) {
        _state.value = newState
    }

    fun updateAudioLevel(level: Float) {
        _audioLevel.value = level
    }

    fun updatePartialText(text: String) {
        _partialText.value = text
    }

    fun updateTranscript(text: String) {
        _transcript.value = text
    }

    // --- Audio focus events ---

    fun onAudioFocusGain() {
        _audioFocusState.value = AudioFocusState.NORMAL
    }

    fun onAudioFocusLoss() {
        _audioFocusState.value = AudioFocusState.LOST
    }

    fun onAudioFocusLossTransient() {
        _audioFocusState.value = AudioFocusState.TRANSIENT_LOST
    }

    fun onAudioFocusDuck() {
        _audioFocusState.value = AudioFocusState.DUCKING
    }

    // --- Bluetooth events ---

    fun onBluetoothScoConnected() {
        _isBluetoothScoConnected.value = true
    }

    fun onBluetoothScoDisconnected() {
        _isBluetoothScoConnected.value = false
    }
}

/** Possible audio focus states for the voice system. */
enum class AudioFocusState {
    NORMAL,
    LOST,
    TRANSIENT_LOST,
    DUCKING,
}
