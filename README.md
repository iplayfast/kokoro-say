# Kokoro Text-to-Speech CLI

A command-line interface for Kokoro text-to-speech synthesis, supporting multiple voices and languages. This tool provides an easy way to convert text to speech using various voices and language models.

## Features

- Multiple voice options (11 different voices)
- Support for 6 languages: English (US/UK), French, Japanese, Korean, and Mandarin Chinese
- Voice and language selection by name or number
- Pipe support for text input
- Single standalone executable after building

## Installation

1. Clone this repository:
```bash
git clone https://github.com/iplayfast/kokoro-tts-cli
cd kokoro-tts-cli
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Build the executable:
```bash
python build.py
```

4. Install system-wide:

On Linux/macOS:
```bash
sudo cp build/dist/say /usr/local/bin/
```

On Windows:
- Copy `build/dist/say.exe` to your desired location
- Add that location to your system's PATH

## Usage

Show available voices and languages:
```bash
say --help
```

Basic usage with voice and language selection:
```bash
# Using names
say --voice af_sarah --lang en-us "Hello World"

# Using numbers
say --voice 1 --lang 1 "Hello World"
```

Using pipes:
```bash
echo "Hello World" | say --voice af_sarah --lang en-us
cat message.txt | say --voice 1 --lang 1
```

## Dependencies

- kokoro-onnx[gpu]
- sounddevice
- PyInstaller (build only)
- torch (build only)
- requests (build only)

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
