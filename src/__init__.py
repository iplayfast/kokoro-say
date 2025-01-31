"""Text-to-Speech system using Kokoro with persistent servers."""

import os
import sys
import logging

# Configure root logger
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/tts_daemon.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import components
from .constants import *
from .fetchers import *
from .model_server import ModelServer
