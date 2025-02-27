# Text-to-Speech System Design - Revised Architecture

## Core Components

### 1. Model Server (model_server.py)
- Central resource manager for model instances
- Serves as a hub for voice servers to access model resources
- Receives initial client requests
- Responsible for:
  * Initiating and managing appropriate voice servers
  * Providing model resources to voice servers
  * Routing synthesis requests to the correct voice server
  * Managing the shared model instance for efficiency

### 2. Voice Servers (voice_server.py)
- One server per voice+language combination
- Receives text from model server
- Fully handles the synthesis process for its specific voice
- Responsible for:
  * Performing the actual text-to-speech synthesis
  * Playing audio output
  * Managing audio interruptions (stopping current speech to start new)
  * Saving WAV files when requested
  * May call back to model server for synthesis processing

### 3. Client (say.py)
- Command-line interface for users
- Connects directly to model server only
- Sends text and voice preferences to the model server
- IMPORTANT: Exits immediately after model server acknowledges receipt of request
- Does NOT wait for synthesis completion or audio playback
- Options:
  * --voice: select voice
  * --lang: select language
  * --kill: sends kill command
  * --list: lists available voices/languages
  * --output: save audio to file instead of playing
  * Input via command line args or pipe

## System Architecture

### Communication Flow
```
1. Client -> Model Server:
   - Client sends text and voice selection to model server
   - Model server acknowledges receipt
   - CLIENT EXITS IMMEDIATELY after acknowledgment
   - Model server determines appropriate voice server
   - If needed, starts new voice server for requested voice
   - Routes request to voice server

2. Model Server <-> Voice Server:
   - Voice server receives synthesis request from model server
   - Voice server requests actual audio synthesis from model server
   - Model server performs synthesis using its centralized model
   - Model server sends generated audio data to voice server

3. Voice Server -> Audio Output:
   - Voice server plays resulting audio (client already exited)
   - Handles interruptions when new text arrives
   - Saves audio files when requested
   - Manages its own lifecycle and shuts down after inactivity
```

### Key Design Principles
- Decentralized audio playback handled by voice servers
- Centralized model resources for efficiency (single model instance)
- Voice servers focused on voice-specific operations and audio playback
- Clear separation of responsibilities between components
- Immediate client response (non-blocking) - client never waits for synthesis or playback
- Voice servers persist after client exits to complete audio playback
- System continues functioning after client disconnects

### Performance Considerations
1. Model Server:
   - Single instance of resource-intensive model
   - Efficient resource allocation
   - Manages voice server lifecycle
   - Minimal computational overhead for routing

2. Voice Servers:
   - Lightweight processes for specific voice/language pairs
   - Can be started/stopped dynamically based on inactivity
   - Handles interruption and playback independently
   - Optimized for specific voice characteristics

### Communication Protocol
```json
// Client -> Model Server
{
    "text": "Text to synthesize",
    "voice": "voice_id",
    "lang": "language_code",
    "speed": 1.0
}

// Model Server -> Voice Server
{
    "text": "Text to synthesize",
    "voice": "voice_id",
    "lang": "language_code",
    "speed": 1.0
}

// Voice Server -> Model Server (if needed)
{
    "request": "synthesis_resources",
    "text": "Text to process"
}
```

### Implementation Considerations
- Socket-based communication between components
- Thread safety for concurrent requests
- Graceful handling of server startup/shutdown
- Efficient resource cleanup for inactive voice servers
- Interrupt handling for immediate response to new requests

## Future Enhancements
- Dynamic voice loading without server restart
- Voice quality improvements through server-specific optimizations
- Support for additional audio formats
- Streaming audio output for long-form text
- Performance monitoring and optimization
