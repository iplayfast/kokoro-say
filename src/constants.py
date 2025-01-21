#!/usr/bin/env python3

"""
Constants used throughout the TTS application.
"""

# Available languages with their display names
LANGUAGES = [
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)"),
    ("fr-fr", "French"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("cmn", "Mandarin Chinese")
]

# Socket and PID file paths
MODEL_SERVER_SOCKET = "/tmp/kokoro_model_server"
SOCKET_BASE_PATH = "/tmp/tts_socket"
PID_BASE_PATH = "/tmp/tts_daemon"
MODEL_PID_FILE = "/tmp/model_server.pid"

# Default values
DEFAULT_VOICE = "1"
DEFAULT_LANG = "en-us"

# Audio processing constants
MAX_CHUNK_SIZE = 500
SENTENCE_END_MARKERS = {'.', '!', '?'}

# Logging configuration
# Logging configuration
LOG_FILE = '/tmp/tts_daemon.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
LOG_LEVEL = 'WARNING'  # Change this to control verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Server configuration
MAX_RETRIES = 5
RETRY_DELAY = 2
SERVER_TIMEOUT = 30