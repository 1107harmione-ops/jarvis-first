package com.jarvis.voice.playback

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log

/**
 * Streaming PCM audio player backed by [AudioTrack].
 *
 * Plays audio at 22 050 Hz (Piper TTS output rate), MONO, PCM_16BIT.
 * Chunks can be written incrementally as they arrive over the WebSocket.
 */
class AudioPlayerService {

    private var audioTrack: AudioTrack? = null

    private val sampleRate = 22050
    private val channelConfig = AudioFormat.CHANNEL_OUT_MONO
    private val audioEncoding = AudioFormat.ENCODING_PCM_16BIT

    /** One second buffer for smooth playback. */
    private val bufferSizeInBytes: Int by lazy {
        maxOf(
            AudioTrack.getMinBufferSize(sampleRate, channelConfig, audioEncoding),
            sampleRate * 2, // 1 s × 2 bytes/sample
        )
    }

    /** Whether the player has been started but not yet stopped. */
    @Volatile
    var isPlaying: Boolean = false
        private set

    // ---- Lifecycle ----

    /**
     * Initialise (or re‑initialise) the AudioTrack and start playing.
     * Safe to call multiple times — existing track is released first.
     */
    fun start() {
        stop()
        isPlaying = true
        // Actual creation is deferred to the first playChunk call.
        Log.d(TAG, "AudioPlayer started")
    }

    /**
     * Write a PCM chunk for streaming playback.
     * The AudioTrack is lazily created on the first call.
     */
    fun playChunk(pcmChunk: ByteArray) {
        if (!isPlaying) {
            Log.w(TAG, "playChunk called but player not started — ignoring")
            return
        }
        val track = getOrCreateTrack()
        val written = track.write(pcmChunk, 0, pcmChunk.size)
        if (written != pcmChunk.size) {
            Log.w(TAG, "Partial write: $written / ${pcmChunk.size}")
        }
    }

    /**
     * Immediately stop playback and release resources.
     * AudioTrack is nulled out so [playChunk] will lazily recreate it.
     */
    fun stop() {
        isPlaying = false
        releaseTrack()
        Log.d(TAG, "AudioPlayer stopped")
    }

    /**
     * Flush the AudioTrack internal buffer for an immediate stop.
     * Does NOT release the track — useful during interrupt so the next
     * TTS chunk can start playing right away.
     */
    fun flush() {
        try {
            audioTrack?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to flush AudioTrack", e)
        }
    }

    /**
     * Drain queued audio (play out remaining buffers) and then stop.
     */
    fun drainAndStop() {
        try {
            audioTrack?.let { track ->
                if (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    track.stop()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error draining AudioTrack", e)
        }
        stop()
    }

    // ---- Internal helpers ----

    private fun getOrCreateTrack(): AudioTrack {
        audioTrack?.let { track ->
            if (track.state == AudioTrack.STATE_INITIALIZED) {
                return track
            }
        }
        return createTrack()
    }

    private fun createTrack(): AudioTrack {
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(audioEncoding)
                    .setSampleRate(sampleRate)
                    .setChannelMask(channelConfig)
                    .build()
            )
            .setBufferSizeInBytes(bufferSizeInBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        if (track.state != AudioTrack.STATE_INITIALIZED) {
            Log.e(TAG, "AudioTrack failed to initialise")
            track.release()
            audioTrack = null
            isPlaying = false
            throw IllegalStateException("AudioTrack initialisation failed")
        }

        track.play()
        audioTrack = track
        Log.d(TAG, "AudioTrack created (sampleRate=$sampleRate, buffer=$bufferSizeInBytes)")
        return track
    }

    private fun releaseTrack() {
        try {
            audioTrack?.let { track ->
                if (track.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    track.stop()
                }
                track.release()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing AudioTrack", e)
        }
        audioTrack = null
    }

    companion object {
        private const val TAG = "AudioPlayer"
    }
}
