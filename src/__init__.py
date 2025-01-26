"""Text-to-Speech system with persistent voice-specific servers."""

import os
import sys
import logging

# Get log level from environment or default to WARNING
LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING').upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler('/tmp/tts_daemon.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import components after logging is configured
from .model_server import ModelServer
from .voice_server import VoiceServer
from .constants import *
from .utils import *

__all__ = [
    'ModelServer',
    'VoiceServer',
    'LANGUAGES',
    'DEFAULT_VOICE',
    'DEFAULT_LANG',
    'MODEL_SERVER_HOST',
    'MODEL_SERVER_PORT',
    'VOICE_SERVER_BASE_PORT',
    'ensure_model_and_voices',
    'kill_all_daemons',
    'get_voice_from_input',
    'get_language_from_input',
    'ensure_model_server_running'
]