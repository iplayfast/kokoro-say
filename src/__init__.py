"""Text-to-Speech system with persistent voice-specific servers."""

import os
import sys
import logging
from .constants import LOG_FILE, LOG_FORMAT, DEFAULT_LOG_LEVEL

# Configure root logger
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# Expose constants
from .constants import *

# Import utility functions
from .utils import VoiceManager, get_voice_from_input, get_language_from_input, show_help

# Import socket protocol
from .socket_protocol import SocketProtocol

# Import model and voice servers last to avoid circular imports
from .model_server import ModelServer
from .voice_server import VoiceServer

# Export public API
__all__ = [
    'ModelServer',
    'VoiceServer',
    'SocketProtocol',
    'VoiceManager',
    'LANGUAGES',
    'DEFAULT_VOICE',
    'DEFAULT_LANG',
    'MODEL_SERVER_HOST',
    'MODEL_SERVER_PORT',
    'get_voice_from_input',
    'get_language_from_input',
    'show_help'
]