# Kokoro TTS System Architecture

## Overview
The Kokoro TTS system is designed as a multi-process architecture that optimizes memory usage and response time by maintaining persistent model and voice servers. The system uses a combination of TCP sockets and process management to handle text-to-speech requests efficiently.

## Key Components

### 1. Command-Line Interface (say.py)
- Primary user interface for text-to-speech generation
- Handles command-line arguments and input validation
- Supports both direct text input and piped input
- Manages communication with the voice servers

### 2. Model Server (model_server.py)
The central server component that:
- Loads and maintains the Kokoro TTS model in memory
- Handles voice downloads and management
- Coordinates voice servers
- Processes text-to-speech requests
- Runs as a persistent daemon

Key features:
- Single instance design to minimize memory usage
- GPU support with CPU fallback
- Automatic voice file downloading
- Connection pooling and request queuing

### 3. Voice Servers (voice_server.py)
Dedicated servers for individual voices that:
- Handle voice-specific text-to-speech processing
- Manage audio playback
- Support streaming output
- Handle interruption and cleanup

Features:
- One server per voice to optimize resource usage
- Automatic registration with model server
- Clean shutdown and resource cleanup
- Real-time audio streaming capabilities

## Communication Flow

1. Initial Request:
```
User → say.py → Model Server → Voice Server
```

2. Subsequent Requests:
```
User → say.py → Existing Voice Server
```

## Process Lifecycle

### Startup Sequence
1. User invokes `say.py` with text and voice selection
2. System checks for existing model server
   - If not running, starts model server daemon
3. Model server initializes and loads base TTS model
4. System checks for required voice server
   - If not running, starts new voice server
5. Voice server registers with model server

### Request Processing
1. Text input is received by say.py
2. Request is routed to appropriate voice server
3. Voice server coordinates with model server for TTS generation
4. Audio is generated and played or saved

### Resource Management
- Model server remains resident for continued use
- Voice servers persist until idle timeout
- Automatic cleanup of unused resources
- Memory-efficient voice loading

## Technical Details

### Server Communication
- TCP sockets for reliable inter-process communication
- Length-prefixed JSON messages for request/response
- Robust error handling and recovery
- Atomic operations for thread safety

### Audio Processing
- Real-time audio generation and playback
- Stream interruption support
- Multiple sample rate support
- Buffer management for smooth playback

### Resource Conservation
1. Memory:
   - Single loaded model instance
   - On-demand voice loading
   - Automatic resource cleanup

2. Processing:
   - Request queuing and batching
   - GPU acceleration when available
   - Efficient process reuse

## Configuration

### Server Settings
- Model server port: 5000 (default)
- Voice server ports: 5001+ (dynamically assigned)
- Socket paths: /tmp/kokoro_socket/
- Cache directory: ~/.cache/kokoro/

### Voice Management
- Automatic voice downloading
- Voice persistence between calls
- British/American voice separation
- Custom pronunciation support

## Error Handling

1. Startup Errors:
   - Model initialization failures
   - Port conflicts
   - Resource unavailability

2. Runtime Errors:
   - Connection failures
   - Process crashes
   - Resource exhaustion

3. Recovery Mechanisms:
   - Automatic server restart
   - Connection retry logic
   - Resource cleanup

## Security Considerations

1. File Permissions:
   - Socket permissions
   - Cache directory access
   - Log file permissions

2. Process Isolation:
   - Separate user permissions
   - Resource limits
   - Network restrictions

## Logging and Monitoring

- Centralized logging
- Component-specific log files
- Debug mode support
- Performance metrics

## Future Improvements

1. Performance:
   - Request batching optimization
   - Memory usage optimization
   - Startup time reduction

2. Features:
   - Additional voice support
   - Language expansion
   - Audio format options

3. Architecture:
   - Container support
   - Distributed deployment
   - Cloud integration
