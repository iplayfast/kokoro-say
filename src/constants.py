#!/usr/bin/env python3

"""Constants used throughout the TTS application."""

# Available languages with their display names
LANGUAGES = [
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)"),
    ("fr-fr", "French"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("cmn", "Mandarin Chinese")
]

# Server configuration
MODEL_SERVER_HOST = "127.0.0.1"
MODEL_SERVER_PORT = 5000
VOICE_SERVER_BASE_PORT = 5001

# Default values
DEFAULT_VOICE = "1"
DEFAULT_LANG = "en-us"

# Audio processing constants
MAX_CHUNK_SIZE = 500
SENTENCE_END_MARKERS = {'.', '!', '?'}

# Logging configuration
LOG_FILE = '/tmp/tts_daemon.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
LOG_LEVEL = 'WARNING'

# Server configuration
MAX_RETRIES = 5
RETRY_DELAY = 2
SERVER_TIMEOUT = 30