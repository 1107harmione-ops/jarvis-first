# JARVIS Voice System — Production Voice Architecture Design

**Date:** 2026-06-06
**Status:** Draft
**Author:** AI Architecture Team

---

## 1. Overview

### 1.1 Purpose

Design and implement a production-grade voice system for **JARVIS**, an AI assistant on Android, that delivers natural real-time voice conversations. The system replaces the existing file-based Whisper.cpp STT and blocking Piper TTS with a streaming, full-duplex, interruptible voice pipeline that supports wake word, offline mode, and multilingual conversations.

### 1.2 Scope

This spec covers a **modular component architecture** across two codebases:

**Python Backend (FastAPI):**
- WebSocket audio streaming endpoint (`/ws/voice`)
- Streaming Whisper STT (via faster-whisper or whisper.cpp streaming)
- Streaming Piper TTS with audio chunk generation
- Voice session manager with full state machine
- Server-side wake word detection
- Interrupt handling (voice activity during TTS)
- Multilingual support (English + Hindi)
- Voice config per user/session

**Android App (Kotlin):**
- Voice foreground service with always-listening
- Streaming PCM audio capture (AudioRecord)
- Streaming PCM audio playback (AudioTrack)
- WebSocket voice client (OkHttp)
- On-device low-power wake word (optional)
- Audio focus management
- Bluetooth headset support
- Interrupt controller (voice energy monitor during TTS)
- Echo cancellation (adaptive filter)
- Persistent notification with media controls

### 1.3 Out of Scope

- On-device on-device STT model (server-side only)
- Custom wake word training pipeline
- Speaker identification/verification
- Audio beamforming or multi-mic array processing
- Cloud TTS/STT API fallback (future)
- Video conferencing integration

---

## 2. Python Backend Architecture

### 2.1 Package Structure

```
jarvis-voice/
├── jarvis_voice/
│   ├── __init__.py
│   ├── config.py                   # Voice configuration (pydantic-settings)
│   ├── server.py                   # FastAPI application + WebSocket routes
│   ├── models.py                   # Pydantic models for voice messages
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py              # VoiceSessionManager (state machine)
│   │   └── state.py                # VoiceState enum, session data
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseSTTProvider (abstract)
│   │   ├── whisper_stt.py          # Whisper/faster-whisper streaming STT
│   │   └── vad.py                  # Voice Activity Detection
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseTTSProvider (abstract)
│   │   └── piper_tts.py            # Piper TTS streaming
│   ├── wakeword/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseWakeWordDetector
│   │   ├── openwakeword_detector.py
│   │   └── energy_detector.py      # Energy-based fallback
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── audio_processor.py      # VAD, noise gate, resampling
│   │   └── echo_cancellation.py    # Server-side AEC (optional)
│   ├── memory/
│   │   ├── __init__.py
│   │   └── voice_memory.py         # Voice history, commands, sessions
│   └── llm/
│       └── __init__.py             # LLM integration contract
├── tests/
│   ├── test_session.py
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_wakeword.py
│   └── conftest.py
├── models/                         # Model storage (whisper, piper, wakeword)
├── voices/                         # Piper voice files (.onnx + .json)
├── requirements.txt
├── pyproject.toml
└── deploy/
    ├── docker-compose.yml
    └── Dockerfile
```

### 2.2 Config (`config.py`)

```python
class VoiceConfig(BaseModel):
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    bytes_per_sample: int = 2  # PCM16

    # STT
    stt_provider: str = "faster_whisper"  # faster_whisper | whisper_cpp
    whisper_model: str = "base"           # tiny, base, small, medium, large
    whisper_device: str = "cpu"           # cpu, cuda
    whisper_compute_type: str = "int8"    # float16, int8_float16
    stt_language: str = "auto"            # auto, en, hi
    stt_beam_size: int = 5
    stt_vad_filter: bool = True
    partial_results_interval: float = 0.3  # seconds between partial results

    # TTS
    tts_provider: str = "piper"
    piper_executable: str = ""
    piper_voice_en: str = "en_US-lessac-medium"
    piper_voice_hi: str = "hi_IN-medium"
    tts_length_scale: float = 1.0
    tts_noise_scale: float = 0.667
    tts_sample_rate: int = 22050

    # Wake Word
    wake_word: str = "hey jarvis"
    wake_word_provider: str = "openwakeword"  # openwakeword | porcupine | energy
    wake_word_sensitivity: float = 0.5
    wake_word_cooldown: float = 2.0

    # Session
    silence_timeout_sec: float = 1.5
    max_command_duration: float = 30.0
    min_command_duration: float = 0.3
    voice_timeout_sec: float = 10.0

    # Interrupt
    interrupt_energy_threshold: float = 0.03  # normalized RMS
    interrupt_min_duration: float = 0.15       # seconds of speech to trigger

    # Multilingual
    default_language: str = "en"
    supported_languages: list[str] = ["en", "hi"]
    auto_detect_language: bool = True

    # Memory
    voice_history_ttl_days: int = 90
    max_frequent_commands: int = 50
```

### 2.3 Voice State Model (`session/state.py`)

```python
class VoiceState(str, Enum):
    IDLE = "idle"
    WAKE_PENDING = "wake_pending"  # Wake word detected, verifying
    LISTENING = "listening"        # Capturing audio, no wake needed
    STT_PROCESSING = "stt_processing"  # Transcribing
    THINKING = "thinking"          # LLM generating response
    SPEAKING = "speaking"          # TTS playback
    INTERRUPTED = "interrupted"    # User interrupted
    OFFLINE = "offline"            # Offline mode
    ERROR = "error"

@dataclass
class VoiceSession:
    session_id: str
    user_id: str
    state: VoiceState = VoiceState.IDLE
    language: str = "en"
    audio_buffer: asyncio.Queue[bytes] = field(default_factory=lambda: asyncio.Queue())
    partial_text: str = ""
    final_text: str = ""
    tts_task: asyncio.Task | None = None
    stt_task: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    speaking_done: asyncio.Event = field(default_factory=asyncio.Event)
    
    # Metrics
    total_commands: int = 0
    total_interrupts: int = 0
    total_errors: int = 0
```

### 2.4 Session Manager (`session/manager.py`)

The `VoiceSessionManager` is the central orchestrator. It:

- Creates/destroys sessions per WebSocket connection
- Runs the state machine with async transitions
- Coordinates STT → LLM → TTS pipeline
- Handles interrupts with immediate state reset
- Enforces timeouts and cleanup

**State transition rules:**

```
IDLE:
  - [audio received + wake word] → WAKE_PENDING
  - [audio received + no wake]   → LISTENING (if push-to-talk)
  - [timeout > 30min]            → cleanup session

WAKE_PENDING:
  - [wake verified]              → LISTENING
  - [wake failed]                → IDLE
  - [timeout > 5s]               → IDLE

LISTENING:
  - [silence timeout]            → STT_PROCESSING
  - [max duration]               → STT_PROCESSING
  - [audio_end signal]           → STT_PROCESSING
  - [interrupt during speaking]  → STT_PROCESSING (new command)

STT_PROCESSING:
  - [transcript ready]           → THINKING
  - [error/no speech]            → IDLE

THINKING:
  - [response ready]             → SPEAKING
  - [error]                      → IDLE

SPEAKING:
  - [TTS complete]               → IDLE
  - [user voice detected]        → INTERRUPTED

INTERRUPTED:
  - [immediately]                → IDLE (then LISTENING if still audio)
```

### 2.5 Streaming STT (`stt/whisper_stt.py`)

Uses `faster-whisper` for streaming transcription with partial results.

```python
class WhisperSTT(BaseSTTProvider):
    def __init__(self, config: VoiceConfig):
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        self.vad = VoiceActivityDetector(...)
        self.language = config.stt_language
        self.audio_buffer: list[float] = []
        self.last_partial_time = 0.0
        self.partial_interval = config.partial_results_interval
    
    async def transcribe_stream(
        self,
        audio_queue: asyncio.Queue[bytes],
        partial_callback: Callable[[str, float], None],
        language: str = "auto",
    ) -> tuple[str, float]:
        """
        Streaming transcription:
        - Accumulate audio chunks from queue
        - Run VAD to detect speech segments
        - At partial_interval, run whisper on buffer for partial results
        - On silence/final, run full transcription
        - Yield partial results via callback
        """
        ...
    
    async def transcribe_file(self, audio_path: str, language: str = "auto") -> tuple[str, float]:
        """Full file transcription (non-streaming fallback)."""
        segments, info = await asyncio.to_thread(
            self.model.transcribe, audio_path,
            language=language if language != "auto" else None,
            beam_size=5, vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments)
        return text, info.average_log_prob
    
    @staticmethod
    def language_code(name: str) -> str:
        """Map 'en' → 'english', detect 'auto' from audio."""
        ...
```

**Streaming approach:** Audio frames are accumulated in a ring buffer. Every `partial_results_interval` seconds, the buffered audio is transcribed. The last N seconds of audio are kept. On silence detection, the full utterance is transcribed once more for the final result.

### 2.6 Streaming TTS (`tts/piper_tts.py`)

Uses Piper TTS via subprocess with stdout pipe for streaming PCM output.

```python
class PiperTTS(BaseTTSProvider):
    def __init__(self, config: VoiceConfig):
        self.executable = config.piper_executable
        self.voices = {
            "en": config.piper_voice_en,
            "hi": config.piper_voice_hi,
        }
        self.length_scale = config.tts_length_scale
        self.noise_scale = config.tts_noise_scale
    
    async def speak_stream(
        self,
        text: str,
        language: str,
        chunk_callback: Callable[[bytes], None],
        interrupt_event: asyncio.Event,
    ) -> None:
        """
        Streaming TTS:
        - Launch Piper subprocess with --output-raw
        - Stream stdout PCM chunks via chunk_callback
        - Check interrupt_event between chunks (immediate stop)
        - Handle language-specific voice selection
        """
        voice = self.voices.get(language, self.voices["en"])
        voice_path = self._resolve_voice_path(voice)
        
        proc = await asyncio.create_subprocess_exec(
            self.executable,
            "--model", str(voice_path),
            "--output-raw",
            "--length-scale", str(self.length_scale),
            "--noise-scale", str(self.noise_scale),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        
        async def feed_stdin():
            proc.stdin.write(text.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
        
        async def read_stdout():
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                if interrupt_event.is_set():
                    proc.kill()
                    break
                await chunk_callback(chunk)
        
        await asyncio.gather(feed_stdin(), read_stdout())
        await proc.wait()
```

**Streaming approach:** Piper is launched as a subprocess with `--output-raw` which writes raw PCM16 to stdout. The `chunk_callback` sends each PCM chunk over WebSocket. The `interrupt_event` is checked after every chunk — if set, the subprocess is killed and the function returns immediately.

### 2.7 Wake Word (`wakeword/`)

Server-side wake word uses OpenWakeWord library for neural detection with energy-based fallback.

```python
class OpenWakeWordDetector(BaseWakeWordDetector):
    def __init__(self, wake_word: str = "hey jarvis", sensitivity: float = 0.5):
        self.model = openwakeword.Model(wakeword_models=[f"hey_{wake_word.split()[-1]}"])
        self.threshold = 0.5 + (1.0 - sensitivity) * 0.3
        self.cooldown = 2.0
        self.last_detection = 0.0
    
    async def process_chunk(self, audio_bytes: bytes, sample_rate: int) -> DetectionResult:
        """Process audio chunk; returns DetectionResult if wake word found."""
        import numpy as np
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        prediction = self.model.predict(pcm)
        score = max(prediction.values())
        
        now = time.time()
        if score > self.threshold and (now - self.last_detection) > self.cooldown:
            self.last_detection = now
            return DetectionResult(detected=True, score=float(score), source="openwakeword")
        return DetectionResult(detected=False, score=float(score))
```

### 2.8 WebSocket Server (`server.py`)

FastAPI WebSocket endpoint `/ws/voice` with bidirectional binary+JSON protocol.

```python
@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = await session_manager.create_session(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive" and "bytes" in message:
                # Binary PCM16 audio
                await session_manager.handle_audio(session, message["bytes"])
            
            elif message["type"] == "websocket.receive" and "text" in message:
                # JSON control message
                data = json.loads(message["text"])
                await session_manager.handle_control(session, data)
    
    except WebSocketDisconnect:
        await session_manager.destroy_session(session)
```

**Audio processing flow in `session_manager.handle_audio()`:**

```
1. Check session state
2. If IDLE or WAKE_PENDING:
   a. Run wake word detection on chunk
   b. If detected → transition to LISTENING
3. If LISTENING:
   a. Run VAD on chunk
   b. Append to audio buffer
   c. If silence timeout elapsed → trigger STT
   d. If max duration reached → trigger STT
4. If SPEAKING:
   a. Check interrupt energy threshold
   b. If exceeded → trigger INTERRUPT
```

### 2.9 Interrupt Handling

```python
async def _check_interrupt(self, session: VoiceSession, audio_chunk: bytes) -> bool:
    """Detect if user spoke during TTS playback."""
    if session.state != VoiceState.SPEAKING:
        return False
    
    # Compute RMS energy of incoming audio
    samples = struct.unpack(f"<{len(audio_chunk)//2}h", audio_chunk)
    rms = math.sqrt(sum(s*s for s in samples) / len(samples)) / 32768.0
    
    if rms > self.config.interrupt_energy_threshold:
        # User is speaking — trigger interrupt
        session.total_interrupts += 1
        session.interrupt_event.set()
        
        await self._transition(session, VoiceState.INTERRUPTED)
        await self._send_control(session.websocket, {
            "type": "state_change", "state": "interrupted"
        })
        
        # Immediately start listening for new command
        session.audio_buffer = asyncio.Queue()
        await self._transition(session, VoiceState.LISTENING)
        return True
    
    return False
```

---

## 3. Android App Architecture

### 3.1 Package Structure

```
app/src/main/java/com/jarvis/voice/
├── service/
│   ├── VoiceForegroundService.kt     # Foreground service for always-listening
│   └── VoiceServiceConnection.kt      # Activity ↔ Service binding
├── capture/
│   ├── AudioCaptureService.kt         # AudioRecord → PCM streaming
│   ├── AudioCaptureConfig.kt          # Sample rate, format, source
│   └── AudioLevelMonitor.kt           # Real-time audio level (for UI)
├── playback/
│   ├── AudioPlayerService.kt          # AudioTrack → PCM streaming playback
│   └── AudioFocusManager.kt          # Audio focus request/transient/duck
├── client/
│   ├── VoiceWebSocketClient.kt        # OkHttp WebSocket for voice
│   ├── VoiceProtocol.kt              # Message types, serialization
│   └── VoiceConnectionState.kt       # Connection state enum
├── wakeword/
│   ├── WakeWordDetector.kt            # On-device low-power wake word
│   └── WakeWordModelLoader.kt        # Load .tflite or .onnx model
├── interrupt/
│   ├── InterruptController.kt         # Voice energy monitor during TTS
│   └── EnergyThresholdCalculator.kt   # Adaptive energy threshold
├── audio/
│   ├── AudioRouter.kt                # Route audio to speaker/bluetooth/headset
│   ├── EchoCanceller.kt              # Adaptive echo cancellation (AEC)
│   └── AudioResampler.kt             # Resample to/from 16kHz
├── bluetooth/
│   ├── BluetoothVoiceManager.kt       # SCO connection management
│   └── BluetoothHeadsetReceiver.kt    # Headset connect/disconnect
├── notification/
│   └── VoiceNotificationManager.kt    # Persistent notification + actions
├── state/
│   ├── VoiceState.kt                  # VoiceState enum + observable state
│   └── VoiceSettingsStore.kt          # Voice settings persistence
└── model/
    ├── VoiceConfig.kt                 # Voice configuration
    └── VoiceMetrics.kt               # Latency, commands, errors
```

### 3.2 Voice Foreground Service (`service/VoiceForegroundService.kt`)

```kotlin
class VoiceForegroundService : Service() {
    private lateinit var audioCapture: AudioCaptureService
    private lateinit var audioPlayer: AudioPlayerService
    private lateinit var wsClient: VoiceWebSocketClient
    private lateinit var interruptController: InterruptController
    private lateinit var audioFocusManager: AudioFocusManager
    private lateinit var notificationManager: VoiceNotificationManager
    
    override fun onCreate() {
        super.onCreate()
        initializeComponents()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_LISTENING -> startListening()
            ACTION_STOP_LISTENING -> stopListening()
            ACTION_INTERRUPT -> interrupt()
            ACTION_SET_LANGUAGE -> setLanguage(intent.getStringExtra("language") ?: "en")
            ACTION_SET_WAKE_WORD -> setWakeWord(intent.getBooleanExtra("enabled", true))
        }
        return START_STICKY
    }
    
    private fun initializeComponents() {
        audioCapture = AudioCaptureService()
        audioPlayer = AudioPlayerService()
        wsClient = VoiceWebSocketClient(
            onStateChange = { state -> updateState(state) },
            onTtsChunk = { chunk -> audioPlayer.playChunk(chunk) },
            onTtsStart = { audioFocusManager.requestFocus() },
            onTtsEnd = { audioFocusManager.abandonFocus() },
            onTranscript = { text -> onTranscript(text) },
        )
        interruptController = InterruptController(
            energyThreshold = 0.03f,
            onInterrupt = { sendInterrupt() }
        )
        audioFocusManager = AudioFocusManager(this)
        notificationManager = VoiceNotificationManager(this)
    }
    
    private fun startListening() {
        audioCapture.start {
            // Each PCM chunk from AudioRecord
            wsClient.sendAudio(it)
        }
        startForeground(NOTIFICATION_ID, notificationManager.buildListeningNotification())
        updateState(VoiceState.LISTENING)
    }
    
    private fun sendInterrupt() {
        wsClient.sendInterrupt()
        audioPlayer.stop()
        audioCapture.restart() // Flush old buffer
    }
}
```

### 3.3 Audio Capture (`capture/AudioCaptureService.kt`)

```kotlin
class AudioCaptureService {
    private var audioRecord: AudioRecord? = null
    private var isCapturing = false
    private var captureJob: Job? = null
    
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    
    fun start(onAudioChunk: (ByteArray) -> Unit) {
        if (isCapturing) return
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate, channelConfig, audioFormat, bufferSize * 4
        )
        audioRecord?.startRecording()
        isCapturing = true
        
        captureJob = CoroutineScope(Dispatchers.IO).launch {
            val buffer = ByteArray(bufferSize)
            while (isCapturing) {
                val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                if (bytesRead > 0) {
                    onAudioChunk(buffer.copyOf(bytesRead))
                }
            }
        }
    }
    
    fun stop() {
        isCapturing = false
        captureJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }
    
    fun restart() {
        // Flush buffer by stopping and re-starting quickly
        stop()
        // Small delay to clear HW buffer
        Thread.sleep(50)
    }
    
    val currentAmplitude: Float
        get() = ... // RMS of last buffer for UI visualization
}
```

### 3.4 Audio Playback (`playback/AudioPlayerService.kt`)

```kotlin
class AudioPlayerService {
    private var audioTrack: AudioTrack? = null
    private var isPlaying = false
    private val sampleRate = 22050 // Piper TTS output rate
    
    fun playChunk(pcmChunk: ByteArray) {
        ensureAudioTrack()
        audioTrack?.write(pcmChunk, 0, pcmChunk.size)
    }
    
    fun start() {
        isPlaying = true
    }
    
    fun stop() {
        isPlaying = false
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
    }
    
    private fun ensureAudioTrack() {
        if (audioTrack == null || audioTrack?.state != AudioTrack.STATE_INITIALIZED) {
            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
                .setAudioFormat(AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
                .setBufferSizeInBytes(sampleRate * 2) // 1 second buffer
                .build()
            audioTrack?.play()
        }
    }
}
```

### 3.5 Audio Focus Manager (`playback/AudioFocusManager.kt`)

```kotlin
class AudioFocusManager(private val context: Context) {
    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    
    private val focusChangeListener = AudioManager.OnAudioFocusChangeListener { focusChange ->
        when (focusChange) {
            AudioManager.AUDIOFOCUS_GAIN -> {
                // Resume playback at normal volume
                VoiceStateHolder.onAudioFocusGain()
            }
            AudioManager.AUDIOFOCUS_LOSS -> {
                // Stop playback permanently
                VoiceStateHolder.onAudioFocusLoss()
            }
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> {
                // Pause playback temporarily
                VoiceStateHolder.onAudioFocusLossTransient()
            }
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> {
                // Lower volume
                VoiceStateHolder.onAudioFocusDuck()
            }
        }
    }
    
    fun requestFocus(): Int {
        return audioManager.requestAudioFocus(
            focusChangeListener,
            AudioManager.STREAM_MUSIC,
            AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK
        )
    }
    
    fun abandonFocus() {
        audioManager.abandonAudioFocus(focusChangeListener)
    }
}
```

### 3.6 WebSocket Voice Client (`client/VoiceWebSocketClient.kt`)

```kotlin
class VoiceWebSocketClient(
    private val onStateChange: (VoiceState) -> Unit,
    private val onTtsChunk: (ByteArray) -> Unit,
    private val onTtsStart: () -> Unit,
    private val onTtsEnd: () -> Unit,
    private val onTranscript: (String) -> Unit,
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.SECONDS)
        .protocols(listOf(Protocol.HTTP_1_1))
        .build()
    private var ws: WebSocket? = null
    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    fun connect(url: String) {
        val request = Request.Builder().url(url.replace("http", "ws") + "/ws/voice").build()
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onMessage(ws: WebSocket, bytes: ByteString) {
                // Binary PCM16 TTS chunk
                onTtsChunk(bytes.toByteArray())
            }
            
            override fun onMessage(ws: WebSocket, text: String) {
                val msg = gson.fromJson(text, JsonObject::class.java)
                when (msg.get("type")?.asString) {
                    "state_change" -> {
                        val state = VoiceState.fromString(msg.get("state")?.asString)
                        onStateChange(state)
                        when (state) {
                            VoiceState.SPEAKING -> onTtsStart()
                            VoiceState.IDLE -> onTtsEnd()
                            else -> {}
                        }
                    }
                    "partial" -> {
                        // Partial STT result (for UI display)
                    }
                    "transcript" -> {
                        onTranscript(msg.get("text")?.asString ?: "")
                    }
                    "error" -> {
                        Log.e(TAG, "Voice error: ${msg.get("message")}")
                    }
                }
            }
            
            override fun onOpen(ws: WebSocket, response: Response) {
                onStateChange(VoiceState.IDLE)
            }
            
            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                onStateChange(VoiceState.OFFLINE)
                scheduleReconnect()
            }
        })
    }
    
    fun sendAudio(chunk: ByteArray) {
        ws?.send(ByteString.of(*chunk))
    }
    
    fun sendInterrupt() {
        val msg = JsonObject().apply { addProperty("type", "interrupt") }
        ws?.send(gson.toJson(msg))
    }
    
    fun sendAudioEnd() {
        val msg = JsonObject().apply { addProperty("type", "audio_end") }
        ws?.send(gson.toJson(msg))
    }
    
    fun sendConfig(language: String, voiceSpeed: Float) {
        val msg = JsonObject().apply {
            addProperty("type", "config")
            addProperty("language", language)
            addProperty("voice_speed", voiceSpeed)
        }
        ws?.send(gson.toJson(msg))
    }
    
    fun disconnect() {
        ws?.close(1000, "Client closing")
        scope.cancel()
    }
}
```

### 3.7 Bluetooth Manager (`bluetooth/BluetoothVoiceManager.kt`)

```kotlin
class BluetoothVoiceManager(private val context: Context) {
    private val SCO_STATE_TIMEOUT_MS = 5000L
    
    // Register for Bluetooth SCO connection events
    private val scoReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.getIntExtra(AudioManager.EXTRA_SCO_AUDIO_STATE, -1)) {
                AudioManager.SCO_AUDIO_STATE_CONNECTED -> {
                    VoiceStateHolder.onBluetoothScoConnected()
                }
                AudioManager.SCO_AUDIO_STATE_DISCONNECTED -> {
                    VoiceStateHolder.onBluetoothScoDisconnected()
                }
            }
        }
    }
    
    fun startSco(audioManager: AudioManager) {
        // Use SCO for voice calls (higher quality for voice)
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.startBluetoothSco()
        // Register receiver to know when SCO is actually connected
        context.registerReceiver(scoReceiver, IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED))
    }
    
    fun stopSco(audioManager: AudioManager) {
        audioManager.stopBluetoothSco()
        audioManager.mode = AudioManager.MODE_NORMAL
        try { context.unregisterReceiver(scoReceiver) } catch (_: Exception) {}
    }
}
```

### 3.8 Interrupt Controller (`interrupt/InterruptController.kt`)

```kotlin
class InterruptController(
    private val energyThreshold: Float = 0.03f,
    private val onInterrupt: () -> Unit,
) {
    private var isMonitoring = false
    private var monitorJob: Job? = null
    
    fun startMonitoring(audioCapture: AudioCaptureService) {
        isMonitoring = true
        monitorJob = CoroutineScope(Dispatchers.IO).launch {
            val buffer = ByteArray(1024) // Small for fast detection
            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                16000, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                buffer.size * 4
            )
            audioRecord.startRecording()
            
            var consecutiveSpeechFrames = 0
            while (isMonitoring) {
                val bytesRead = audioRecord.read(buffer, 0, buffer.size)
                if (bytesRead <= 0) continue
                
                val rms = computeRms(buffer, bytesRead)
                if (rms > energyThreshold) {
                    consecutiveSpeechFrames++
                    if (consecutiveSpeechFrames >= 5) { // ~150ms of speech
                        onInterrupt()
                        consecutiveSpeechFrames = 0
                    }
                } else {
                    consecutiveSpeechFrames = 0
                }
            }
            
            audioRecord.stop()
            audioRecord.release()
        }
    }
    
    fun stopMonitoring() {
        isMonitoring = false
        monitorJob?.cancel()
    }
    
    private fun computeRms(buffer: ByteArray, bytesRead: Int): Float {
        var sum = 0.0
        for (i in 0 until bytesRead step 2) {
            if (i + 1 < bytesRead) {
                val sample = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
                sum += (sample * sample).toDouble()
            }
        }
        return Math.sqrt(sum / (bytesRead / 2)).toFloat() / 32768f
    }
}
```

### 3.9 State Management (`state/VoiceState.kt`)

```kotlin
enum class VoiceState {
    IDLE, LISTENING, STT_PROCESSING, THINKING, SPEAKING, INTERRUPTED, OFFLINE, ERROR;
    
    companion object {
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

// Observable state holder for UI
object VoiceStateHolder {
    private val _state = MutableStateFlow(VoiceState.IDLE)
    val state: StateFlow<VoiceState> = _state.asStateFlow()
    
    private val _audioLevel = MutableStateFlow(0f)
    val audioLevel: StateFlow<Float> = _audioLevel.asStateFlow()
    
    private val _partialText = MutableStateFlow("")
    val partialText: StateFlow<String> = _partialText.asStateFlow()
    
    fun updateState(newState: VoiceState) { _state.value = newState }
    fun updateAudioLevel(level: Float) { _audioLevel.value = level }
    fun updatePartialText(text: String) { _partialText.value = text }
    
    // Audio focus events
    fun onAudioFocusGain() { /* Resume */ }
    fun onAudioFocusLoss() { /* Stop TTS */ }
    fun onAudioFocusLossTransient() { /* Pause TTS */ }
    fun onAudioFocusDuck() { /* Lower volume */ }
    
    // Bluetooth events
    fun onBluetoothScoConnected() { /* Route audio through SCO */ }
    fun onBluetoothScoDisconnected() { /* Route back to speaker */ }
}
```

### 3.10 Notification Manager (`notification/VoiceNotificationManager.kt`)

```kotlin
class VoiceNotificationManager(private val context: Context) {
    private val CHANNEL_ID = "jarvis_voice_channel"
    private val NOTIFICATION_ID = 2001
    
    init {
        val channel = NotificationChannel(
            CHANNEL_ID, "JARVIS Voice",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Voice assistant status"
            setShowBadge(false)
        }
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(channel)
    }
    
    fun buildListeningNotification(): Notification {
        val stopIntent = PendingIntent.getService(
            context, 0,
            Intent(context, VoiceForegroundService::class.java)
                .setAction("STOP_LISTENING"),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText("Listening...")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopIntent)
            .setStyle(NotificationCompat.DecoratedCustomViewStyle())
            .build()
    }
    
    fun buildSpeakingNotification(text: String): Notification {
        return NotificationCompat.Builder(context, CHANNEL_ID)
            .setContentTitle("JARVIS")
            .setContentText(text.take(50))
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()
    }
}
```

---

## 4. Voice Memory (`memory/voice_memory.py`)

```python
class VoiceMemory:
    """Stores voice history, frequent commands, sessions, user preferences."""
    
    async def log_command(self, user_id: str, text: str, language: str, 
                           duration_ms: int, success: bool) -> None
    async def get_frequent_commands(self, user_id: str, limit: int = 20) -> list[str]
    async def log_session(self, user_id: str, session_id: str, 
                           duration: float, commands: int) -> None
    async def get_user_voice_prefs(self, user_id: str) -> dict
    async def update_user_voice_prefs(self, user_id: str, prefs: dict) -> None
    async def get_voice_history(self, user_id: str, days: int = 7) -> list[dict]
    async def get_voice_metrics(self, user_id: str) -> dict
```

Voice memory entries stored in MongoDB (via `jarvis-memory` package):

```python
class VoiceCommandLog(BaseModel):
    user_id: str
    text: str
    language: str
    duration_ms: int
    success: bool
    command_type: str | None  # Classified intent
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VoiceSessionLog(BaseModel):
    session_id: str
    user_id: str
    start_time: datetime
    end_time: datetime | None
    duration_seconds: float
    commands: int
    interrupts: int
    language: str
```

---

## 5. Offline Mode

### 5.1 Fallback Strategy

When the server is unreachable:

| Component | Online | Offline Fallback |
|-----------|--------|-----------------|
| Wake Word | Server-side + optional on-device | On-device (energy/VAD only) |
| STT | faster-whisper | Android SpeechRecognizer / fallback message |
| TTS | Piper streaming | termux-tts-speak / espeak |
| LLM | Full LLM | Keyword-based response / cached responses |
| Voice Pipeline | Full duplex | Reduced (push-to-talk, no interrupt) |

### 5.2 Android Offline Detection

```kotlin
// In VoiceWebSocketClient
override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
    onStateChange(VoiceState.OFFLINE)
    // Fallback to Android SpeechRecognizer
    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
    }
    // Start fallback activity
}
```

### 5.3 Server Health Monitoring

```python
# Health check endpoint for voice server
@app.get("/health/voice")
async def voice_health():
    return {
        "status": "online",
        "stt_model": config.whisper_model,
        "stt_loaded": whisper_stt.is_loaded,
        "tts_voices": list(piper_tts.voices.keys()),
        "wake_word": config.wake_word,
        "languages": config.supported_languages,
        "active_sessions": session_manager.active_count,
    }
```

---

## 6. Multilingual Support

### 6.1 Language Detection Flow

```
Audio → Whisper (auto-detect language)
       ├── detected: "en" → Piper: en_US-lessac-medium
       ├── detected: "hi" → Piper: hi_IN-medium  
       └── detected: other → Piper: en_US-lessac-medium (fallback)
```

### 6.2 Configuration

```python
class MultilingualConfig:
    # Piper voice files
    PIPER_VOICES = {
        "en": {
            "model": "en_US-lessac-medium.onnx",
            "config": "en_US-lessac-medium.onnx.json",
            "sample_rate": 22050,
        },
        "hi": {
            "model": "hi_IN-medium.onnx",
            "config": "hi_IN-medium.onnx.json",
            "sample_rate": 22050,
        },
    }
    
    # Whisper supported languages relevant to our use case
    STT_LANGUAGES = {
        "auto": None,  # Auto-detect
        "en": "en",
        "hi": "hi",
    }
```

---

## 7. Security & Permissions

### 7.1 Android Permissions

```xml
<!-- AndroidManifest.xml permissions -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE" />
<uses-permission android:name="android.permission.BLUETOOTH" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
<uses-permission android:name="android.permission.BLUETOOTH_SCO" />
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.RECEIVE_BOOT_COMPLETED" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
```

### 7.2 Microphone Security

```kotlin
// Secure recording: never store raw audio persistently
class SecureAudioCapture(context: Context) {
    private val cryptoManager = CryptoManager(context)
    
    // Audio is only kept in memory, never written to disk
    // If caching is needed, use encrypted temp files
    private fun createSecureTempFile(): File {
        val file = File(context.cacheDir, "voice_${UUID.randomUUID()}.enc")
        file.deleteOnExit()
        return file
    }
}
```

---

## 8. Testing Strategy

### 8.1 Backend Tests

```python
# test_stt.py — Test Whisper integration
class TestWhisperSTT:
    async def test_transcribe_english(self):
        audio = load_test_audio("hello_world.wav")
        text, confidence = await stt.transcribe_file(audio)
        assert "hello" in text.lower()
        assert confidence > 0.5
    
    async def test_transcribe_hindi(self):
        audio = load_test_audio("hindi_test.wav")
        text, confidence = await stt.transcribe_file(audio, language="hi")
        assert len(text) > 0
        assert confidence > 0.3

# test_session.py — Test state machine
class TestVoiceSession:
    async def test_idle_to_listening(self):
        session = await manager.create_session(mock_ws)
        await manager.handle_audio(session, test_chunk)
        assert session.state == VoiceState.LISTENING
    
    async def test_interrupt_during_speaking(self):
        session = await manager.create_session(mock_ws)
        session.state = VoiceState.SPEAKING
        session.interrupt_event = asyncio.Event()
        await manager.handle_audio(session, high_energy_chunk)
        assert session.state == VoiceState.INTERRUPTED
        assert session.interrupt_event.is_set()
    
    async def test_silence_timeout(self):
        session = await manager.create_session(mock_ws)
        session.state = VoiceState.LISTENING
        session.last_activity = time.time() - 10
        await manager._check_timeouts()
        assert session.state == VoiceState.STT_PROCESSING

# test_tts.py — Test Piper streaming
class TestPiperTTS:
    async def test_stream_output(self):
        chunks = []
        async def collect(chunk):
            chunks.append(chunk)
        await tts.speak_stream("Hello world", "en", collect, asyncio.Event())
        assert len(chunks) > 0
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 1000

# test_wakeword.py — Test wake word detection
class TestWakeWord:
    async def test_detect_wake_word(self):
        audio = load_test_audio("hey_jarvis.wav")
        result = await detector.process_chunk(audio, 16000)
        assert result.detected
        assert result.score > 0.5
    
    async def test_no_false_positive(self):
        audio = load_test_audio("random_speech.wav")
        result = await detector.process_chunk(audio, 16000)
        assert not result.detected
```

### 8.2 Android Tests

```kotlin
// Instrumentation tests
class AudioCaptureTest {
    @Test
    fun testCaptureAndStop() {
        val capture = AudioCaptureService()
        val chunks = mutableListOf<ByteArray>()
        capture.start { chunks.add(it) }
        Thread.sleep(500)
        capture.stop()
        assertTrue(chunks.isNotEmpty())
        assertTrue(chunks.sumOf { it.size } > 0)
    }
}

class VoiceWebSocketClientTest {
    @Test
    fun testSendInterrupt() {
        val client = VoiceWebSocketClient(...)
        client.connect(mockWebServer.url("/ws/voice").toString())
        client.sendInterrupt()
        val request = mockWebServer.takeRequest()
        assertTrue(request.body.readUtf8().contains("interrupt"))
    }
}
```

---

## 9. Performance Targets

| Operation | Target P99 | Strategy |
|-----------|-----------|----------|
| Wake word detection | < 200ms | OpenWakeWord on server, energy on device |
| STT latency (first partial) | < 500ms | Streaming with 300ms partial interval |
| STT latency (final) | < 2s for 5s utterance | faster-whisper with int8 |
| TTS latency (first audio) | < 300ms | Piper --output-raw subprocess |
| TTS streaming rate | Real-time (22050 Hz) | Chunked PCM via WebSocket |
| Interrupt response | < 200ms | Energy threshold on device |
| Full round-trip | < 3s total | Pipeline optimization |
| Battery impact | < 5% per hour | Low-power VAD, streaming not polling |

---

## 10. Production Deployment

### 10.1 Docker Compose (Backend)

```yaml
version: "3.9"
services:
  jarvis-voice:
    build:
      context: .
      args:
        - PIPER_VERSION=2023.11
    ports:
      - "8002:8002"
    environment:
      - WHISPER_MODEL=base
      - WHISPER_DEVICE=cpu
      - PIPER_EXECUTABLE=/usr/local/bin/piper
      - WAKE_WORD=hey jarvis
    volumes:
      - ./models:/app/models
      - ./voices:/app/voices
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 4G
        reservations:
          memory: 2G
```

### 10.2 Model Files

Required model files for offline operation:

| Model | Size | Source |
|-------|------|--------|
| faster-whisper base | ~1.5 GB | Hugging Face |
| Piper en_US-lessac-medium | ~50 MB | GitHub Releases |
| Piper hi_IN-medium | ~50 MB | GitHub Releases |
| OpenWakeWord hey_jarvis | ~5 MB | OpenWakeWord |

Total: ~1.6 GB for full offline multilingual support.

### 10.3 Environment Variables

```bash
# Voice Server
VOICE_HOST=0.0.0.0
VOICE_PORT=8002
LOG_LEVEL=INFO

# Audio
SAMPLE_RATE=16000
FRAME_MS=30

# STT (faster-whisper)
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
STT_LANGUAGE=auto
PARTIAL_RESULTS_INTERVAL=0.3

# TTS (Piper)
PIPER_EXECUTABLE=/usr/local/bin/piper
PIPER_VOICES_DIR=/app/voices
TTS_LENGTH_SCALE=1.0

# Wake Word
WAKE_WORD=hey jarvis
WAKE_WORD_PROVIDER=openwakeword
WAKE_WORD_SENSITIVITY=0.5

# Session
SILENCE_TIMEOUT_SEC=1.5
MAX_COMMAND_DURATION_SEC=30.0
INTERRUPT_ENERGY_THRESHOLD=0.03

# Multilingual
DEFAULT_LANGUAGE=en
SUPPORTED_LANGUAGES=en,hi
```

---

## 11. Implementation Phases

| Phase | Components | Files | Est. Effort |
|-------|-----------|-------|-------------|
| **1. Backend Core** | Config, models, session manager, state machine | 5 files | High |
| **2. Streaming STT** | WhisperSTT, VAD, partial results | 3 files | High |
| **3. Streaming TTS** | PiperTTS, subprocess management, interrupt | 2 files | Medium |
| **4. WebSocket Server** | FastAPI endpoint, binary+JSON protocol | 2 files | Medium |
| **5. Wake Word** | Server-side OpenWakeWord + energy fallback | 3 files | Medium |
| **6. Android Service** | VoiceForegroundService, capture, playback, WS client | 8 files | High |
| **7. Android Interrupt** | InterruptController, audio focus, Bluetooth | 4 files | Medium |
| **8. Voice Memory** | MongoDB integration, command logging | 2 files | Low |
| **9. Offline Mode** | Fallback detection, reduced pipeline | 2 files | Low |
| **10. Testing** | Backend tests, Android tests, integration | 10+ files | Medium |

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Server-side STT + TTS | Keeps Android app lightweight; model upgrades without app updates; faster-whisper more accurate than on-device |
| Binary PCM16 over WebSocket | No encoding overhead; lowest latency; existing Android AudioRecord/AudioTrack native format |
| JSON control + binary audio on same WS | Single connection for control and audio; simpler than separate channels |
| Piper subprocess with stdout pipe | Supports streaming with minimal latency; Piper is the best offline TTS option |
| Faster-whisper with int8 | Best accuracy/speed trade-off for CPU; int8 quantization reduces model size 4x |
| Energy-based interrupt detection | Simple, fast, no model needed; low false-positive with minimum duration gate |
| On-device + server wake word | Optional on-device for instant response; server-side for accuracy with OpenWakeWord |
| AudioTrack for playback | Native Android PCM playback; lowest latency; supports streaming chunks |
| CoroutineScope(Dispatchers.IO) | Ensures audio capture runs on background thread without blocking UI |
| STAR_STICKY service | Android will restart service if killed; maintains always-listening behavior |
