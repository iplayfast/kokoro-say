"""
Text-to-Speech system with persistent voice-specific daemons.
"""

import os
import sys
import logging

# Configure logging first, before any other imports
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('/tmp/tts_daemon.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Now import the rest of the package components
from .base_server import UnixSocketServer
from .model_server import ModelServer
from .voice_server import VoiceServer  
from .constants import (
    LANGUAGES,
    DEFAULT_VOICE,
    DEFAULT_LANG,
    SOCKET_BASE_PATH,
    MODEL_SERVER_SOCKET
)
from .utils import (
    ensure_model_and_voices,
    kill_all_daemons,
    get_voice_from_input,
    get_language_from_input,
    ensure_model_server_running
)

__all__ = [
    'UnixSocketServer',
    'ModelServer',
    'VoiceServer',  
    'LANGUAGES',
    'DEFAULT_VOICE',
    'DEFAULT_LANG',
    'SOCKET_BASE_PATH',
    'MODEL_SERVER_SOCKET',
    'ensure_model_and_voices',
    'kill_all_daemons',
    'get_voice_from_input',
    'get_language_from_input', 
    'ensure_model_server_running'
]