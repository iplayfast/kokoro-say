#!/usr/bin/env python3

"""Constants used throughout the TTS application."""

import os
from pathlib import Path

#
# File and Directory Paths
#

# Base cache directory
CACHE_DIR = Path.home() / ".cache" / "kokoro"

# Model and voices directories
MODEL_DIR = CACHE_DIR / "model"
VOICES_DIR = CACHE_DIR / "voices"

# Specific file paths
MODEL_PATH = MODEL_DIR / "kokoro-v0_19.onnx"
VOICES_CONFIG_PATH = VOICES_DIR / "voices.json"

# Temporary directories
TEMP_DIR = Path("/tmp/kokoro_api")
LOG_DIR = Path("/tmp")
LOG_FILE = LOG_DIR / "kokoro.log"

#
# Socket and Network Configuration
#

# Server addresses
MODEL_SERVER_HOST = "127.0.0.1"
MODEL_SERVER_PORT = 5000
API_SERVER_HOST = "127.0.0.1"
API_SERVER_PORT = 8000

# Socket paths
SOCKET_BASE_PATH = "/tmp/tts_socket"

# Network parameters
MAX_CHUNK_SIZE = 8192
SERVER_TIMEOUT = 30.0  # seconds
REQUEST_TIMEOUT = 30.0  # seconds
SYNTHESIS_TIMEOUT = 60.0  # seconds for longer synthesis jobs
VOICE_SERVER_INACTIVE_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Socket protocol
SOCKET_PROTOCOL_VERSION = b"TTSv1.0\n"
SOCKET_HEADER_SIZE = 8

#
# Voice Definitions
#

AMERICAN_VOICES = {
    "af_heart": "Heart (Female, US)",
    "af_bella": "Bella (Female, US)",
    "af_nicole": "Nicole (Female, US)",
    "af_sarah": "Sarah (Female, US)",
    "af_sky": "Sky (Female, US)",
    "am_adam": "Adam (Male, US)",
    "am_michael": "Michael (Male, US)"
}

BRITISH_VOICES = {
    "bf_emma": "Emma (Female, UK)",
    "bf_isabella": "Isabella (Female, UK)",
    "bm_george": "George (Male, UK)",
    "bm_lewis": "Lewis (Male, UK)"
}

VOICE_PREFIX_MEANINGS = {
    "af": "American Female",
    "am": "American Male",
    "bf": "British Female",
    "bm": "British Male"
}

# Available languages with their display names
LANGUAGES = [
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)"),
    ("fr-fr", "French"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("cmn", "Mandarin Chinese")
]

# Default values
DEFAULT_VOICE = "af_bella"
DEFAULT_LANG = "en-us"

#
# Audio Processing Constants
#

SAMPLE_RATE = 24000
MAX_AUDIO_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB for large audio responses
SENTENCE_END_MARKERS = {'.', '!', '?'}

#
# Multiprocessing and Threading
#

MAX_WORKERS = 4
THREAD_POOL_SIZE = 4

#
# Logging Configuration
#

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = 'DEBUG'

#
# Model Parameters
#

MODEL_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx"
VOICES_BASE_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices"
SPEAKER_EMBEDDING_DIM = 256
SPEAKER_EMBEDDING_CHANNELS = 256

#
# Testing Constants
#

# Delay between tests in seconds
TEST_DELAY = 2.0
# Longer text for testing interruptions and performance
TEST_LONG_TEXT = """This is a very long piece of text that will take some time to speak. 
It contains multiple sentences and should take at least 10 seconds to say. 
Let me tell you about text to speech systems. They convert written text into spoken words. 
This technology has many applications in accessibility, education, and entertainment."""
# Test sample file names
TEST_OUTPUT_PREFIX = "test_output"