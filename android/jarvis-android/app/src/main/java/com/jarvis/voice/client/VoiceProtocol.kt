package com.jarvis.voice.client

import com.google.gson.Gson
import com.google.gson.JsonObject

/**
 * Voice WebSocket protocol definitions.
 *
 * Defines message types, JSON payload builders, and response parsers
 * for the streaming voice communication between Android and backend.
 *
 * Supports:
 * - Binary PCM16 audio streaming (bidirectional)
 * - JSON control messages
 * - Hindi and English language switching
 * - Offline mode commands
 * - Interrupt handling
 * - Partial and final transcription results
 * - Voice memory sync
 */
object VoiceProtocol {

    // ── Message Types (Client → Server) ──────────────────────────

    const val TYPE_AUDIO_START = "audio_start"
    const val TYPE_AUDIO_END = "audio_end"
    const val TYPE_AUDIO_CHUNK = "audio_chunk"
    const val TYPE_INTERRUPT = "interrupt"
    const val TYPE_CONFIG = "config"
    const val TYPE_PING = "ping"
    const val TYPE_CLOSE = "close"

    // ── Message Types (Server → Client) ──────────────────────────

    const val TYPE_CONNECTED = "connected"
    const val TYPE_PONG = "pong"
    const val TYPE_STATE = "state"
    const val TYPE_PARTIAL = "partial"
    const val TYPE_TRANSCRIPT = "transcript"
    const val TYPE_THINKING = "thinking"
    const val TYPE_TTS_START = "tts_start"
    const val TYPE_TTS_CHUNK = "tts_chunk"
    const val TYPE_TTS_END = "tts_end"
    const val TYPE_RESULT = "result"
    const val TYPE_ERROR = "error"
    const val TYPE_CONFIG_ACK = "config_ack"

    // ── Voice States ─────────────────────────────────────────────

    const val STATE_IDLE = "idle"
    const val STATE_LISTENING = "listening"
    const val STATE_PROCESSING_STT = "processing_stt"
    const val STATE_THINKING = "thinking"
    const val STATE_SPEAKING = "speaking"
    const val STATE_INTERRUPTED = "interrupted"
    const val STATE_OFFLINE = "offline"
    const val STATE_ERROR = "error"

    private val gson = Gson()

    // ── Client → Server Message Builders ─────────────────────────

    /**
     * Create audio_start JSON message.
     */
    fun audioStart(language: String = "en"): String {
        return """{"type":"$TYPE_AUDIO_START","language":"$language"}"""
    }

    /**
     * Create audio_end JSON message.
     */
    fun audioEnd(): String {
        return """{"type":"$TYPE_AUDIO_END"}"""
    }

    /**
     * Create interrupt JSON message.
     */
    fun interrupt(): String {
        return """{"type":"$TYPE_INTERRUPT"}"""
    }

    /**
     * Create config update JSON message.
     */
    fun configUpdate(
        language: String? = null,
        voiceSpeed: Float? = null,
        voicePitch: Float? = null,
        wakeWordEnabled: Boolean? = null,
        offlineMode: Boolean? = null,
    ): String {
        val obj = JsonObject().apply {
            addProperty("type", TYPE_CONFIG)
            language?.let { addProperty("language", it) }
            voiceSpeed?.let { addProperty("voice_speed", it) }
            voicePitch?.let { addProperty("voice_pitch", it) }
            wakeWordEnabled?.let { addProperty("wake_word_enabled", it) }
            offlineMode?.let { addProperty("offline_mode", it) }
        }
        return gson.toJson(obj)
    }

    /**
     * Create ping JSON message.
     */
    fun ping(): String {
        return """{"type":"$TYPE_PING"}"""
    }

    // ── Server Message Parsers ───────────────────────────────────

    /**
     * Parse a server JSON message and return the message type.
     */
    fun parseMessageType(json: String): String? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            obj.get("type")?.asString
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse a connected message.
     */
    fun parseConnected(json: String): ConnectedInfo? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            ConnectedInfo(
                sessionId = obj.get("session_id")?.asString ?: "",
                state = obj.get("state")?.asString ?: STATE_IDLE,
            )
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse a state update message.
     */
    fun parseState(json: String): String? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            obj.get("state")?.asString
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse a partial transcript message.
     */
    fun parsePartial(json: String): PartialInfo? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            PartialInfo(
                text = obj.get("text")?.asString ?: "",
                confidence = obj.get("confidence")?.asFloat ?: 0f,
            )
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse a final transcript message.
     */
    fun parseTranscript(json: String): TranscriptInfo? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            TranscriptInfo(
                text = obj.get("text")?.asString ?: "",
                confidence = obj.get("confidence")?.asFloat ?: 0f,
                language = obj.get("language")?.asString ?: "en",
            )
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse a thinking message.
     */
    fun parseThinking(json: String): String? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            obj.get("agent")?.asString
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse an error message.
     */
    fun parseError(json: String): String? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            obj.get("message")?.asString
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Parse a result message (final pipeline output).
     */
    fun parseResult(json: String): ResultInfo? {
        return try {
            val obj = gson.fromJson(json, JsonObject::class.java)
            ResultInfo(
                transcript = obj.get("transcript")?.asString ?: "",
                response = obj.get("response")?.asString ?: "",
                agent = obj.get("agent")?.asString ?: "router",
            )
        } catch (e: Exception) {
            null
        }
    }

    // ── Data Classes ─────────────────────────────────────────────

    data class ConnectedInfo(
        val sessionId: String,
        val state: String,
    )

    data class PartialInfo(
        val text: String,
        val confidence: Float,
    )

    data class TranscriptInfo(
        val text: String,
        val confidence: Float,
        val language: String,
    )

    data class ResultInfo(
        val transcript: String,
        val response: String,
        val agent: String,
    )

    // ── Helpers ──────────────────────────────────────────────────

    /**
     * Determine if a message is binary audio (TTS) or JSON control.
     */
    fun isBinaryAudio(data: ByteArray): Boolean {
        // JSON messages start with '{', binary audio does not
        return data.isEmpty() || data[0].toInt() != 0x7B  // '{'
    }

    /**
     * Convert PCM16 bytes to normalized float array for visualization.
     */
    fun pcm16ToFloats(data: ByteArray): FloatArray {
        val samples = FloatArray(data.size / 2)
        for (i in samples.indices) {
            val byteIndex = i * 2
            val sample = (data[byteIndex].toInt() and 0xFF) or
                    (data[byteIndex + 1].toInt() shl 8)
            samples[i] = sample.toFloat() / 32768f
        }
        return samples
    }
}
