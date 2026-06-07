package com.jarvis.voice.client

import android.util.Log
import com.jarvis.voice.state.VoiceState
import com.jarvis.voice.state.VoiceStateHolder
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.util.concurrent.TimeUnit

/**
 * OkHttp-based WebSocket client for the JARVIS voice server.
 *
 * Sends binary PCM16 audio chunks and receives binary PCM16 TTS chunks
 * along with JSON control messages (state changes, transcripts, partials, errors).
 *
 * All callbacks are invoked on [Dispatchers.IO].
 */
class VoiceWebSocketClient(
    private val onStateChange: (VoiceState) -> Unit,
    private val onTtsChunk: (ByteArray) -> Unit,
    private val onTtsStart: () -> Unit,
    private val onTtsEnd: () -> Unit,
    private val onTranscript: (String) -> Unit,
    private val onPartial: (String) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val client: OkHttpClient = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.SECONDS)       // No read timeout for streaming
        .writeTimeout(30, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .protocols(listOf(Protocol.HTTP_1_1))    // WebSocket requires HTTP/1.1
        .build()

    @Volatile
    private var ws: WebSocket? = null

    @Volatile
    private var url: String = ""

    @Volatile
    private var reconnectAttempts = 0

    private var reconnectJob: Job? = null

    /** Whether the client should attempt to reconnect on failure. */
    @Volatile
    var autoReconnect: Boolean = true

    /** Maximum reconnection attempts before giving up (0 = forever). */
    @Volatile
    var maxReconnectAttempts: Int = 20

    /** Base delay in ms for exponential backoff. */
    @Volatile
    var reconnectBaseDelayMs: Long = 1000L

    /** Whether a connection is currently active. */
    val isConnected: Boolean get() = ws != null

    /** The current server URL. */
    val currentUrl: String get() = url

    // ---- Connection management ----

    /**
     * Connect to the voice WebSocket server.
     * @param serverUrl Full server URL (e.g. "ws://192.168.1.100:8002").
     */
    fun connect(serverUrl: String) {
        url = serverUrl
        reconnectAttempts = 0
        doConnect()
    }

    private fun doConnect() {
        val wsUrl = url.trimEnd('/')
        Log.d(TAG, "Connecting to $wsUrl")

        val request = Request.Builder()
            .url(wsUrl)
            .build()

        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                Log.i(TAG, "WebSocket connected to $wsUrl")
                reconnectAttempts = 0
                onStateChange(VoiceState.IDLE)
            }

            override fun onMessage(ws: WebSocket, bytes: ByteString) {
                // Binary message = PCM16 TTS audio chunk
                val chunk = bytes.toByteArray()
                onTtsChunk(chunk)
            }

            override fun onMessage(ws: WebSocket, text: String) {
                handleTextMessage(text)
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closing: code=$code reason=$reason")
                ws.close(code, reason)
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: code=$code reason=$reason")
                this@VoiceWebSocketClient.ws = null
                onStateChange(VoiceState.IDLE)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}", t)
                this@VoiceWebSocketClient.ws = null
                onStateChange(VoiceState.OFFLINE)
                scheduleReconnect()
            }
        })
    }

    /**
     * Disconnect from the server.  No reconnect will be attempted.
     */
    fun disconnect() {
        autoReconnect = false
        cancelReconnect()
        try {
            ws?.close(1000, "Client closing")
        } catch (_: Exception) {}
        ws = null
        scope.cancel()
        Log.d(TAG, "WebSocket disconnected")
    }

    // ---- Sending ----

    /**
     * Send a binary PCM16 audio chunk to the server.
     */
    fun sendAudio(chunk: ByteArray) {
        try {
            ws?.send(ByteString.of(*chunk))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send audio chunk", e)
        }
    }

    /**
     * Send an interrupt message to stop TTS playback.
     */
    fun sendInterrupt() {
        sendText(createInterruptMessage())
    }

    /**
     * Signal the end of the current audio utterance.
     */
    fun sendAudioEnd() {
        sendText(createAudioEndMessage())
    }

    /**
     * Send a configuration update to the server.
     */
    fun sendConfig(language: String, voiceSpeed: Float) {
        sendText(createConfigMessage(language, voiceSpeed))
    }

    // ---- Internal helpers ----

    private fun sendText(json: String) {
        try {
            ws?.send(json)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send text message", e)
        }
    }

    private fun handleTextMessage(text: String) {
        val msg = parseServerMessage(text) ?: run {
            Log.w(TAG, "Unparseable server message: $text")
            return
        }

        when (msg.type) {
            VoiceMessageType.STATE_CHANGE -> {
                val state = VoiceState.fromString(msg.state)
                onStateChange(state)
                when (state) {
                    VoiceState.SPEAKING -> onTtsStart()
                    VoiceState.IDLE, VoiceState.INTERRUPTED -> onTtsEnd()
                    else -> { /* no action */ }
                }
            }

            VoiceMessageType.PARTIAL -> {
                msg.text?.let { text ->
                    VoiceStateHolder.updatePartialText(text)
                    onPartial(text)
                }
            }

            VoiceMessageType.TRANSCRIPT -> {
                msg.text?.let { text ->
                    VoiceStateHolder.updateTranscript(text)
                    onTranscript(text)
                }
            }

            VoiceMessageType.TTS_START -> onTtsStart()
            VoiceMessageType.TTS_END -> onTtsEnd()

            VoiceMessageType.ERROR -> {
                val errorMsg = msg.message ?: msg.text ?: "Unknown error"
                Log.e(TAG, "Server error: $errorMsg")
                onError(errorMsg)
            }

            else -> {
                Log.d(TAG, "Unhandled message type: ${msg.type}")
            }
        }
    }

    // ---- Reconnect with exponential backoff ----

    private fun scheduleReconnect() {
        if (!autoReconnect) return
        if (maxReconnectAttempts > 0 && reconnectAttempts >= maxReconnectAttempts) {
            Log.w(TAG, "Max reconnect attempts ($maxReconnectAttempts) reached")
            return
        }

        cancelReconnect()

        reconnectJob = scope.launch {
            val delayMs = computeBackoff()
            reconnectAttempts++
            Log.i(TAG, "Reconnecting in ${delayMs}ms (attempt $reconnectAttempts)")
            delay(delayMs)
            if (isActive) {
                doConnect()
            }
        }
    }

    private fun cancelReconnect() {
        reconnectJob?.cancel()
        reconnectJob = null
    }

    /** Exponential backoff capped at 30 seconds with jitter. */
    private fun computeBackoff(): Long {
        val base = reconnectBaseDelayMs
        val exp = base * (1L shl minOf(reconnectAttempts, 10)) // cap exponent at 10
        val capped = minOf(exp, 30_000L)
        // Add ±25 % jitter
        val jitter = (capped * 0.25 * Math.random()).toLong()
        return capped + jitter
    }

    companion object {
        private const val TAG = "VoiceWSClient"
    }
}
