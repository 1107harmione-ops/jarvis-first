package com.jarvis.voice.client

import com.google.gson.Gson
import com.google.gson.JsonObject

/**
 * WebSocket message type constants used in the voice protocol.
 * Two-way communication: client sends control, server sends state/transcript/TTS chunks.
 */
object VoiceMessageType {
    // Client → Server
    const val AUDIO_START = "audio_start"
    const val AUDIO_END = "audio_end"
    const val INTERRUPT = "interrupt"
    const val CONFIG = "config"

    // Server → Client
    const val STATE_CHANGE = "state_change"
    const val PARTIAL = "partial"
    const val TRANSCRIPT = "transcript"
    const val TTS_START = "tts_start"
    const val TTS_END = "tts_end"
    const val THINKING = "thinking"
    const val ERROR = "error"
}

private val gson = Gson()

/** Create a JSON interrupt message. */
fun createInterruptMessage(): String {
    val msg = JsonObject().apply {
        addProperty("type", VoiceMessageType.INTERRUPT)
    }
    return gson.toJson(msg)
}

/** Create an audio-end message to signal the end of user speech. */
fun createAudioEndMessage(): String {
    val msg = JsonObject().apply {
        addProperty("type", VoiceMessageType.AUDIO_END)
    }
    return gson.toJson(msg)
}

/** Create a configuration message to update server-side settings. */
fun createConfigMessage(language: String, speed: Float): String {
    val msg = JsonObject().apply {
        addProperty("type", VoiceMessageType.CONFIG)
        addProperty("language", language)
        addProperty("voice_speed", speed)
    }
    return gson.toJson(msg)
}

/** Create an audio-start message (sent before sending binary PCM). */
fun createAudioStartMessage(): String {
    val msg = JsonObject().apply {
        addProperty("type", VoiceMessageType.AUDIO_START)
    }
    return gson.toJson(msg)
}

/**
 * Parsed representation of a server text message.
 */
data class ServerMessage(
    val type: String,
    val state: String? = null,
    val text: String? = null,
    val message: String? = null,
    val confidence: Double? = null,
)

/** Parse a server JSON text message into a typed object. */
fun parseServerMessage(json: String): ServerMessage? {
    return try {
        val obj = gson.fromJson(json, JsonObject::class.java)
        ServerMessage(
            type = obj.get("type")?.asString ?: return null,
            state = obj.get("state")?.asString,
            text = obj.get("text")?.asString,
            message = obj.get("message")?.asString,
            confidence = obj.get("confidence")?.asDouble,
        )
    } catch (_: Exception) {
        null
    }
}
