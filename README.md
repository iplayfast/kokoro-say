# Kokoro-Say

A command-line interface for Kokoro text-to-speech synthesis, supporting multiple voices and languages. This tool provides an easy way to convert text to speech using various voices and language models.

## Overview

This CLI tool is built around the Kokoro text-to-speech engine and uses a daemon architecture for improved performance. When you first run a command, the system starts a voice-specific daemon that keeps the TTS model loaded in memory. Subsequent calls with the same voice reuse this loaded daemon, making the system more efficient for repeated use.

### Architecture

The project consists of two main components:

1. `say.py` - The core Python script that:
   - Manages TTS daemons for each voice/language combination
   - Handles voice and language selection
   - Processes text-to-speech conversion
   - Provides daemon process management

2. `say` - A bash launcher script that:
   - Ensures the Python virtual environment is activated
   - Passes all commands to say.py
   - Maintains the correct working directory context

This separation allows the Python code to run in its proper environment while providing a simple command-line interface to users.

## Features

- Multiple voice options (11 different voices)
- Support for 6 languages: English (US/UK), French, Japanese, Korean, and Mandarin Chinese
- Voice and language selection by name or number
- Pipe support for text input
- Daemon architecture for improved performance
- Process management (ability to kill running daemons)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/kokoro-say
cd kokoro-say
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Make the launcher executable:
```bash
chmod +x say
```

5. Make it available system-wide:
```bash
# Create a symlink in your bin directory
ln -s $(pwd)/say ~/bin/say
```

**Note:** On first run, the script will download all available voices from HuggingFace. This may take several minutes depending on your internet connection. This download only happens once, and the voices are stored locally for future use.

## Usage

Show available voices and languages:
```bash
say --list
```

Basic usage with voice and language selection:
```bash
# Using default voice (voice 1)
say "Hello World"

# Using specific voice and language
say --voice af_sarah --lang en-us "Hello World"

# Using numbers for voice and language
say --voice 2 --lang 1 "Hello World"
```

Using pipes:
```bash
echo "Hello World" | say
echo "Hello World" | say --voice 2
cat message.txt | say --voice 1 --lang 1
```

Managing daemons:
```bash
# Kill all running TTS daemons
say --kill
```

## Dependencies

- kokoro-onnx[gpu]
- sounddevice
- psutil
- portaudio19-dev (Linux only)

On Linux systems, you'll need to install portaudio:
```bash
sudo apt-get install portaudio19-dev  # Debian/Ubuntu
```

## License

```
Copyright 2024 [Crystal Softare (Canada) Inc]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgments

This project uses:
- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) for text-to-speech synthesis
- Voice models from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)

## Notes

- The first time you run a command, `say.py` will download all available voice models from HuggingFace. This initial setup may take several minutes, but is only required once.
- Each voice runs in its own daemon process, which stays resident in memory for improved performance.
- Voice files are stored locally after download and reused for subsequent runs.
- Log files for the daemons can be found in `/tmp/tts_daemon.log`
