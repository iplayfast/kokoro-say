# Kokoro TTS System

A high-performance text-to-speech system built around the Kokoro TTS engine. This system maintains persistent TTS models to avoid reloading between invocations, significantly improving efficiency for repeated use.

## Features

- **Persistent daemon** maintains models in memory for fast synthesis
- **Multiple voices** with automatic downloading
- **Multiple languages** support (US English, UK English)
- **Two interfaces**: CLI tool (`say.py`) and REST API (`api.py`)
- **Efficient socket communication** between client and server
- **File output support** for saving speech as WAV files
- **Speed control** for speech rate adjustment

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: `kokoro-onnx`, `sounddevice`, `soundfile`, `numpy`, `requests`, `fastapi`, `uvicorn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kokoro-tts.git
   cd kokoro-tts
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Command Line Interface (say.py)

The `say.py` script provides a convenient command-line interface for text-to-speech synthesis.

### Basic Usage

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

### Interactive Mode

```bash
# Interactive mode for multiple inputs
python say.py --interactive
```

### Additional Options

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

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--voice` | Voice to use (name or number) |
| `--lang` | Language code or number |
| `--speed` | Speech speed multiplier (default: 1.0) |
| `--output` | Save audio to specified WAV file |
| `--interactive` | Read lines from stdin and speak each one |
| `--list` | List available voices and languages |
| `--update-voices` | Force update of available voices |
| `--kill` | Send kill command to servers |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

## REST API Server (api.py)

The `api.py` script provides a RESTful API for text-to-speech synthesis, ideal for integrating TTS capabilities into web applications or services.

### Starting the API Server

```bash
# Start API server on default port (8000)
python api.py

# Specify host and port
python api.py --host 0.0.0.0 --port 8080

# Enable auto-reload for development
python api.py --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Get system information |
| `/health` | GET | Health check |
| `/voices` | GET | Get list of available voices |
| `/languages` | GET | Get list of available languages |
| `/synthesize` | POST | Synthesize speech (audio plays on server) |
| `/synthesize-file` | POST | Synthesize speech and return WAV file |

### API Usage Examples

#### Get System Info
```bash
curl http://localhost:8000/
```

#### Get Available Voices
```bash
curl http://localhost:8000/voices
```

#### Get Available Languages
```bash
curl http://localhost:8000/languages
```

#### Synthesize Speech (Server Playback)
```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella", "language": "en-us", "speed": 1.0}'
```

#### Synthesize Speech (Return WAV File)
```bash
curl -X POST http://localhost:8000/synthesize-file \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella", "language": "en-us", "speed": 1.0}' \
  --output speech.wav
```

### API Request Parameters

For `/synthesize` and `/synthesize-file` endpoints:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `text` | string | Text to synthesize | (required) |
| `voice` | string | Voice to use | "af_bella" |
| `language` | string | Language code | "en-us" |
| `speed` | float | Speech speed multiplier | 1.0 |

## System Architecture

The system consists of several components:

1. **Model Server**: Central server that manages the TTS model
2. **Voice Servers**: Per-voice servers that handle synthesis for specific voices
3. **CLI Client**: Command-line interface (`say.py`)
4. **API Server**: RESTful API interface (`api.py`)
5. **Shared TTS Client Library**: Common functionality (`tts_client.py`)

Communication between components happens via Unix domain sockets and TCP sockets for efficient IPC.

## Voice Management

The system automatically downloads voices as needed. Available voices include:

### American English Voices
- af_heart, af_bella, af_nicole, af_sarah, af_sky (female)
- am_adam, am_michael (male)

### British English Voices
- bf_emma, bf_isabella (female)
- bm_george, bm_lewis (male)

## Advanced Configuration

Configuration constants are defined in `src/constants.py` and can be modified if needed:

- Server host and port
- Socket paths
- Cache directories
- Audio processing parameters
- Logging options

## Troubleshooting

- **Server won't start**: Check log file at `/tmp/tts_daemon.log`
- **No audio**: Ensure your system audio is working properly
- **Voice not found**: Run `python say.py --update-voices`
- **Hanging processes**: Run `python say.py --kill` to terminate all servers

## License

[Apache License](LICENSE.txt)

## Acknowledgments

- Based on the Kokoro TTS engine
- Voices from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
