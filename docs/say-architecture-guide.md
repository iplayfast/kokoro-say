# Kokoro Say Architecture and Usage Guide

This document provides an overview of the Kokoro Say system architecture, explains how the components interact, and offers examples for common use cases.

## System Architecture

![System Architecture Diagram](say-architecture-diagram.png)

The Kokoro Say system uses a distributed architecture with three core components:

1. **Model Server (model_server.py)**
   - Maintains a single, shared TTS model instance in memory
   - Routes client requests to the appropriate voice server
   - Manages the lifecycle of voice servers
   - Provides direct synthesis services to voice servers

2. **Voice Servers (voice_server.py)**
   - One server per voice+language combination
   - Handles voice-specific synthesis and audio playback
   - Processes interruptions, manages playback, and saves audio files
   - Communicates with the model server for synthesis processing

3. **Client Interfaces**
   - **Command-line client (say.py)**: For direct user interaction
   - **API server (api.py)**: For integration with applications

### Communication Flow

```
1. Client → Model Server:
   - Client sends text and voice selection to model server
   - Model server acknowledges receipt
   - CLIENT EXITS IMMEDIATELY after acknowledgment
   - Model server determines appropriate voice server
   - If needed, starts new voice server for the requested voice
   - Routes request to voice server

2. Model Server ↔ Voice Server:
   - Voice server receives synthesis request from model server
   - Voice server requests actual audio synthesis from model server
   - Model server performs synthesis using its centralized model
   - Model server sends generated audio data to voice server

3. Voice Server → Audio Output:
   - Voice server plays resulting audio (client already exited)
   - Handles interruptions when new text arrives
   - Saves audio files when requested
   - Manages its own lifecycle and shuts down after inactivity
```

### Key Design Principles

- **Centralized model resources**: Single model instance for efficiency
- **Decentralized audio playback**: Handled by voice servers
- **Non-blocking client**: Client exits immediately after request acknowledgment
- **Voice persistence**: Voice servers persist after client exits to complete playback
- **Efficient IPC**: Communication via sockets for performance

## Usage Examples

### Command-Line Interface (say.py)

The `say.py` script provides a convenient way to use TTS from the command line.

#### Basic Usage

```bash
# Basic usage with default voice
python say.py "Hello world"

# Specify a voice by name
python say.py --voice af_bella "Hello world"

# Specify a voice by number
python say.py --voice 1 "Hello world"

# Adjust speech speed
python say.py --speed 1.2 "Hello world"

# Save to WAV file
python say.py --output hello.wav "Hello world"

# Use pipe input
echo "Hello world" | python say.py
```

#### Interactive Mode

```bash
# Interactive mode for multiple inputs
python say.py --interactive
```

#### Other Options

```bash
# List available voices and languages
python say.py --list

# Update available voices list
python say.py --update-voices

# Kill running servers
python say.py --kill

# Set log level
python say.py --log-level DEBUG "Hello world"
```

### REST API Server (api.py)

The `api.py` script provides a RESTful API for integrating TTS capabilities into applications.

#### Starting the API Server

```bash
# Start API server on default port (8000)
python api.py

# Specify host and port
python api.py --host 0.0.0.0 --port 8080

# Enable auto-reload for development
python api.py --reload
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Get system information |
| `/health` | GET | Health check |
| `/voices` | GET | Get list of available voices |
| `/languages` | GET | Get list of available languages |
| `/synthesize` | POST | Synthesize speech (audio plays on server) |
| `/synthesize-file` | POST | Synthesize speech and return WAV file |

#### Server-Side Playback Example

Use the `/synthesize` endpoint when you want the audio to play directly on the server:

```bash
# Using curl
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella", "language": "en-us", "speed": 1.0}'
```

This will play the audio on the server's speakers. The client receives a success response but no audio data.

#### Client-Side Playback Example

Use the `/synthesize-file` endpoint when you want to receive the audio file to play or save on the client:

```bash
# Using curl to save to a file
curl -X POST http://localhost:8000/synthesize-file \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella", "language": "en-us", "speed": 1.0}' \
  --output speech.wav
```

This returns a WAV file that you can play on the client's device.

### Web Integration Examples

#### Basic HTML/JavaScript Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Kokoro Say Demo</title>
</head>
<body>
    <h1>Kokoro Say Demo</h1>
    
    <div>
        <label for="text-input">Text to speak:</label>
        <textarea id="text-input" rows="4" cols="50">Hello, this is a TTS demo.</textarea>
    </div>
    
    <div>
        <label for="voice-select">Voice:</label>
        <select id="voice-select">
            <option value="af_bella">Bella (Female, US)</option>
            <option value="am_adam">Adam (Male, US)</option>
            <option value="bf_emma">Emma (Female, UK)</option>
        </select>
    </div>
    
    <div>
        <button id="speak-button">Speak</button>
        <button id="download-button">Download WAV</button>
    </div>
    
    <div>
        <audio id="audio-player" controls></audio>
    </div>
    
    <script>
        document.getElementById('speak-button').addEventListener('click', synthesizeSpeech);
        document.getElementById('download-button').addEventListener('click', downloadSpeech);
        
        async function synthesizeSpeech() {
            const text = document.getElementById('text-input').value;
            const voice = document.getElementById('voice-select').value;
            
            try {
                const response = await fetch('http://localhost:8000/synthesize-file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: text,
                        voice: voice, 
                        language: "en-us"
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                
                const audioElement = document.getElementById('audio-player');
                audioElement.src = audioUrl;
                audioElement.play();
            } catch (error) {
                console.error('Error synthesizing speech:', error);
                alert('Error synthesizing speech. Check console for details.');
            }
        }
        
        async function downloadSpeech() {
            const text = document.getElementById('text-input').value;
            const voice = document.getElementById('voice-select').value;
            
            try {
                const response = await fetch('http://localhost:8000/synthesize-file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: text,
                        voice: voice, 
                        language: "en-us"
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // Create download link
                const downloadLink = document.createElement('a');
                downloadLink.href = audioUrl;
                downloadLink.download = 'speech.wav';
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            } catch (error) {
                console.error('Error downloading speech:', error);
                alert('Error downloading speech. Check console for details.');
            }
        }
    </script>
</body>
</html>
```

#### Python Web Application Integration (Flask)

```python
from flask import Flask, request, jsonify, send_file
import requests
import tempfile
import os

app = Flask(__name__)

TTS_API_URL = "http://localhost:8000"

@app.route('/speak', methods=['POST'])
def speak():
    """Generate speech and send WAV file to client"""
    data = request.json
    
    # Validate input
    if 'text' not in data:
        return jsonify({"error": "Text is required"}), 400
    
    # Create temporary file for the WAV
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    try:
        # Call the TTS API
        response = requests.post(
            f"{TTS_API_URL}/synthesize-file",
            json={
                "text": data['text'],
                "voice": data.get('voice', 'af_bella'),
                "language": data.get('language', 'en-us'),
                "speed": data.get('speed', 1.0)
            },
            stream=True
        )
        
        response.raise_for_status()
        
        # Save the response content to the temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Send the WAV file to the client
        return send_file(
            temp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='speech.wav'
        )
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"TTS API error: {str(e)}"}), 500
    
    finally:
        # Clean up the temporary file (it will be removed after being sent)
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Understanding the Architecture in Depth

### How the Client Exits Immediately

A key feature of this system is that the client (`say.py`) exits immediately after sending a request, but the audio still plays. This works because:

1. The client sends the text and voice selection to the model server
2. The model server acknowledges the request
3. The client exits immediately
4. The model server forwards the request to the appropriate voice server
5. The voice server handles the synthesis and playback, even though the client is gone

This non-blocking design allows for efficient command-line usage, where you can queue up multiple speech requests without waiting for each to complete.

### Different Audio Output Options

The system supports two main audio output methods:

1. **Server-side playback**: Audio plays through the server's speakers
   - Used by the `/synthesize` API endpoint
   - Used by `say.py` without the `--output` flag

2. **File output**: Audio is saved to a WAV file
   - Used by the `/synthesize-file` API endpoint
   - Used by `say.py` with the `--output` flag

This flexibility makes the system suitable for various use cases, from desktop applications to web services.

### Voice Server Lifecycle

Voice servers are created dynamically as needed and run as separate processes:

1. When a request for a specific voice arrives, the model server checks if a voice server for that voice exists
2. If not, it starts a new voice server process for that voice
3. The voice server stays alive for a while after playback finishes
4. After a period of inactivity, the voice server automatically shuts down

This approach optimizes resource usage by only running the voice servers that are needed.

## Troubleshooting

### Common Issues and Solutions

1. **No audio playback**:
   - Check if your system's audio is working properly
   - Try running `python say.py --kill` and then try again
   - Check the log file at `/tmp/kokoro.log` for errors

2. **API requests failing**:
   - Ensure the API server is running: `python api.py`
   - Check if the model server is running
   - Verify the correct port is being used (default: 8000)

3. **"Voice not found" error**:
   - Run `python say.py --update-voices` to refresh the voice list
   - Check if the voice name or number is correct

4. **Hanging processes**:
   - Run `python say.py --kill` to terminate all servers
   - Use `ps aux | grep python` to check for lingering processes

### Checking Logs

The system logs information to several files:

- Main log: `/tmp/kokoro.log`
- Voice server logs: `/tmp/voice_server_[voice]_[lang].log`

Check these logs for detailed information about any issues.

## Advanced Configuration

Configuration constants are defined in `src/constants.py` and can be modified if needed:

- Server host and port settings
- Socket paths
- Cache directories
- Audio processing parameters
- Logging options

## Development Extensions

Potential extensions for developers:

1. **Add streaming audio support**: Implement WebSocket or SSE for real-time streaming
2. **Create language-specific fine-tuning**: Customize voices for specific languages
3. **Implement voice cloning**: Add ability to create custom voices
4. **Add emotion and emphasis control**: Allow controlling the emotion or emphasis in the speech

---

This guide should help you understand and start using the Kokoro Say system quickly. For more detailed information, refer to the source code and comments within each file.
