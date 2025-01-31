#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import warnings
import argparse
from typing import Optional
from pathlib import Path

# Add parent directory to Python path for src imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Filter specific PyTorch warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")
warnings.filterwarnings("ignore", message="Loading bf_* voice into American English pipeline")
warnings.filterwarnings("ignore", message="Loading bm_* voice into American English pipeline")

from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICE_SERVER_BASE_PORT,
    LANGUAGES,
    DEFAULT_VOICE,
    DEFAULT_LANG
)
from src.utils import show_help, get_voice_from_input, get_language_from_input, VoiceManager

logger = logging.getLogger(__name__)

def ensure_model_server() -> bool:
    """Ensure model server is running, start if needed"""
    try:
        # Try connecting to model server
        with socket.create_connection((MODEL_SERVER_HOST, MODEL_SERVER_PORT), timeout=1) as conn:
            # Send ping request
            request = {"command": "ping"}
            data = json.dumps(request).encode()
            length_prefix = f"{len(data):08d}\n".encode()
            conn.sendall(length_prefix + data)
            
            # Get response
            length_data = conn.recv(9)
            expected_length = int(length_data.decode().strip())
            response_data = conn.recv(expected_length)
            response = json.loads(response_data.decode())
            
            return response.get("status") == "success"
    except:
        # Server not running, start it
        logger.info("Starting model server...")
        
        # Construct Python path that includes the project root
        python_path = f"PYTHONPATH={SCRIPT_DIR} "
        
        # Build the complete command
        cmd = (
            f"{python_path} "
            f"python3 {SCRIPT_DIR}/src/model_server.py "
            f"--script-dir {SCRIPT_DIR} "
            f"--log-level INFO "
            f">> /tmp/model_server.log 2>&1 &"
        )
        
        logger.debug(f"Starting model server with command: {cmd}")
        os.system(cmd)
        
        # Wait for server to start
        import time
        for _ in range(5):  # Try for 5 seconds
            time.sleep(1)
            try:
                with socket.create_connection((MODEL_SERVER_HOST, MODEL_SERVER_PORT), timeout=1):
                    return True
            except:
                continue
        return False

def get_voice_server_port(voice: str) -> Optional[int]:
    """Get port for voice server from model server"""
    try:
        with socket.create_connection((MODEL_SERVER_HOST, MODEL_SERVER_PORT), timeout=5) as conn:
            request = {
                "command": "get_voice_server",
                "voice": voice
            }
            data = json.dumps(request).encode()
            length_prefix = f"{len(data):08d}\n".encode()
            conn.sendall(length_prefix + data)
            
            length_data = conn.recv(9)
            expected_length = int(length_data.decode().strip())
            response_data = conn.recv(expected_length)
            response = json.loads(response_data.decode())
            
            if response.get("status") == "success":
                return response.get("port")
            else:
                logger.error(f"Model server returned error: {response}")
                return None
    except ConnectionRefusedError:
        logger.error("Connection to model server refused. Is the server running?")
    except socket.timeout:
        logger.error("Timeout connecting to model server")
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response from model server")
    except Exception as e:
        logger.error(f"Unexpected error getting voice server port: {e}")
    return None

def speak_text(text: str, voice: str, lang: str, speed: float = 1.0, output: Optional[str] = None) -> bool:
    """Send text to appropriate voice server for TTS"""
    # Get voice server port
    port = get_voice_server_port(voice)
    if not port:
        logger.error("Could not get voice server port")
        return False
        
    try:
        # Connect to voice server
        with socket.create_connection((MODEL_SERVER_HOST, port), timeout=30) as conn:
            request = {
                "text": text,
                "speed": speed,
                "play": output is None,  # Don't play if saving to file
                "output": output
            }
            data = json.dumps(request).encode()
            length_prefix = f"{len(data):08d}\n".encode()
            conn.sendall(length_prefix + data)
            
            # Get response
            length_data = conn.recv(9)
            expected_length = int(length_data.decode().strip())
            response_data = conn.recv(expected_length)
            response = json.loads(response_data.decode())
            
            return response.get("status") == "success"
            
    except Exception as e:
        logger.error(f"Error communicating with voice server: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-to-Speech using Kokoro with voice persistence"
    )
    parser.add_argument('--voice', default=DEFAULT_VOICE, help='Voice to use (name or number)')
    parser.add_argument('--lang', default=DEFAULT_LANG, help='Language code or number')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--output', help='Output WAV file path')
    parser.add_argument('--list', action='store_true', help='List available voices and languages')
    parser.add_argument('--log', action='store_true', help='Enable detailed logging')
    parser.add_argument('text', nargs='*', help='Text to speak')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.log else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/tmp/tts_daemon.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        # Initialize voice manager
        voice_manager = VoiceManager()
        
        # Handle list command
        if args.list or len(sys.argv) == 1:
            show_help(voice_manager, LANGUAGES)
            return
            
        # Get input text
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text)
            
        if not text:
            logger.error("No text provided")
            show_help(voice_manager, LANGUAGES)
            sys.exit(1)

        # Validate voice and language
        try:
            voice = get_voice_from_input(args.voice, voice_manager)
            lang = get_language_from_input(args.lang, LANGUAGES)
        except ValueError as e:
            logger.error(str(e))
            show_help(voice_manager, LANGUAGES)
            sys.exit(1)
            
        # Ensure model server is running
        if not ensure_model_server():
            logger.error("Failed to start model server")
            sys.exit(1)
            
        # Process text
        logger.info(f"Processing text: {text[:50]}...")
        if not speak_text(text, voice, lang, args.speed, args.output):
            logger.error("Failed to process text")
            sys.exit(1)
            
        if args.output:
            logger.info(f"Audio saved to {args.output}")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
