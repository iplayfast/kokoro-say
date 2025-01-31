#!/usr/bin/env python3

"""Constants used throughout the TTS application."""

import os
from pathlib import Path

# Voice definitions
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

# Voice name prefixes and their meanings
VOICE_PREFIX_MEANINGS = {
    "af": "American Female",
    "am": "American Male",
    "bf": "British Female",
    "bm": "British Male"
}

# Available languages with their display names
LANGUAGES = [
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)")
]

# Server configuration
MODEL_SERVER_HOST = "127.0.0.1"
MODEL_SERVER_PORT = 5000
VOICE_SERVER_BASE_PORT = 5001

# Socket paths
SOCKET_BASE_PATH = "/tmp/kokoro_socket"
MODEL_SOCKET_PATH = os.path.join(SOCKET_BASE_PATH, "model")
VOICE_SOCKET_PATH = os.path.join(SOCKET_BASE_PATH, "voice")

# Cache directories
CACHE_DIR = Path.home() / ".cache" / "kokoro"
MODEL_CACHE_DIR = CACHE_DIR / "model"
VOICES_CACHE_DIR = CACHE_DIR / "voices"

# Default values
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "en-us"

# Audio processing constants
MAX_CHUNK_SIZE = 500
SENTENCE_END_MARKERS = {'.', '!', '?'}
SAMPLE_RATE = 24000

# Logging configuration
LOG_FILE = '/tmp/tts_daemon.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
LOG_LEVEL = 'WARNING'

# Server configuration
MAX_RETRIES = 5
RETRY_DELAY = 2
SERVER_TIMEOUT = 30

# Helper functions
def get_all_voices():
    """Return a dictionary of all available voices"""
    return {**AMERICAN_VOICES, **BRITISH_VOICES}

def is_british_voice(voice: str) -> bool:
    """Check if a voice is British based on its prefix"""
    return voice.startswith(("bf_", "bm_"))

def get_voice_type(voice: str) -> str:
    """Get the type description of a voice based on its prefix"""
    prefix = voice[:2]
    return VOICE_PREFIX_MEANINGS.get(prefix, "Unknown")
