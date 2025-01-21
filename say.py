#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
import socket
import time
import multiprocessing
import threading
from typing import List
from pathlib import Path

from src.voice_server import VoiceServer
from src.fetchers import VoiceFetcher
from src.model_server import ModelServer
from src.constants import (
    LANGUAGES,
    DEFAULT_VOICE,
    DEFAULT_LANG,
    LOG_FILE,
    LOG_FORMAT,
    SOCKET_BASE_PATH,
    MODEL_SERVER_SOCKET,    
    LOG_LEVEL
)
from src.utils import (
    ensure_model_and_voices,
    kill_all_daemons,
    get_voice_from_input,
    get_language_from_input,
    ensure_model_server_running
)

def setup_logging(verbose: bool = False):
    """Configure logging with optional verbosity"""
    log_level = logging.DEBUG if verbose else logging.ERROR
    
    # Remove any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    if verbose:
        # When verbose, log to file and stdout with full details
        logging.basicConfig(
            level=log_level,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        # When not verbose, only critical errors go to stdout
        logging.basicConfig(
            level=logging.CRITICAL,
            format='%(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # Set component log levels
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
    
    # Always set third-party loggers to WARNING or higher
    logging.getLogger('sounddevice').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)


# Initialize logging early
setup_logging()
logger = logging.getLogger(__name__)

def show_help(voices: List[str]) -> None:
    """
    Display help message showing available voices and languages
    
    Args:
        voices: List of available voice names
    """
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"  {i:2d}. {voice}")
    
    print("\nAvailable languages:")
    for i, (code, name) in enumerate(LANGUAGES, 1):
        print(f"  {i:2d}. {code:6s} - {name}")
    
    print("\nUsage examples:")
    cmd = os.path.basename(sys.argv[0])
    print(f"  1. List available voices and languages:")
    print(f"     {cmd} --help")
    print(f"\n  2. Using voice and language by name/code:")
    print(f"     {cmd} --voice {voices[0]} --lang en-us \"Hello World\"")
    print(f"\n  3. Using voice and language by number:")
    print(f"     {cmd} --voice 1 --lang 3 \"Bonjour le monde\"")
    print(f"\n  4. Using pipe with mixed selection:")
    print(f"     echo \"こんにちは\" | {cmd} --voice 1 --lang ja")

def start_server_process(voice: str, lang: str) -> None:
    """Start the VoiceServer in a new process"""
    try:
        logger.debug(f"Starting VoiceServer process for voice {voice} and language {lang}")
        server = VoiceServer(voice, lang)
        logger.debug("VoiceServer instance created")        
        server.start()
    except Exception as e:
        logger.error(f"Failed to start VoiceServer: {e}", exc_info=True)
        sys.exit(1)

def send_request(text: str, voice: str, lang: str) -> None:
    """Send text to existing daemon or start a new one"""
    socket_path = f"{SOCKET_BASE_PATH}_{voice}_{lang}"
    logger.debug(f"Using socket path: {socket_path}")
    
    # Ensure model server is running
    logger.debug("Ensuring model server is running")
    if not ensure_model_server_running():
        logger.error("Failed to start model server")
        sys.exit(1)
    
    # If voice server not running, start it
    if not os.path.exists(socket_path):
        logger.debug(f"Starting new voice server for {voice}_{lang}")
        process = multiprocessing.Process(
            target=start_server_process,
            args=(voice, lang)
        )
        process.start()
        
        # Wait for socket to appear
        start_time = time.time()
        while not os.path.exists(socket_path):
            if time.time() - start_time > 30:
                logger.error("Timeout waiting for voice server to start")
                sys.exit(1)
            time.sleep(0.1)
    
    try:
        # Connect to voice server
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(30)
        
        # Try to connect multiple times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.connect(socket_path)
                logger.debug("Connected to voice server")
                break
            except socket.error as e:
                if attempt == max_retries - 1:
                    raise
                logger.debug(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
        
        # Prepare and send request
        request = {
            "text": text,
            "voice": voice,
            "lang": lang
        }
        request_data = json.dumps(request).encode('utf-8')
        length_prefix = f"{len(request_data):08d}\n".encode('utf-8')
        client.sendall(length_prefix + request_data)
        logger.debug("Request sent, waiting for response")
        
        # Read response length
        length_data = client.recv(9)
        if not length_data:
            raise RuntimeError("Server closed connection while reading length")
        
        expected_length = int(length_data.decode().strip())
        logger.debug(f"Expecting response of length {expected_length}")
        
        # Read response data with timeout tracking
        response_data = bytearray()
        remaining = expected_length
        start_time = time.time()
        
        while remaining > 0:
            if time.time() - start_time > 30:
                raise socket.timeout("Timeout while reading response data")
            
            chunk = client.recv(min(8192, remaining))
            if not chunk:
                raise RuntimeError("Server closed connection while reading response")
            response_data.extend(chunk)
            remaining -= len(chunk)
        
        # Parse and handle response
        response = json.loads(response_data.decode())
        logger.debug(f"Received response: (not shown because it's too long)") #{response}")
        
        if response.get("status") == "error":
            logger.error(f"Server error: {response.get('message')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error communicating with voice server: {e}")
        sys.exit(1)
    finally:
        try:
            client.close()
        except:
            pass
        
def main():
    try:
        logger.debug("Starting main function")
        if sys.platform != 'win32':
            multiprocessing.set_start_method('spawn')
        
        parser = argparse.ArgumentParser(
            description="Text-to-Speech with persistent voice-specific daemons",
            add_help=True
        )
        parser.add_argument('--voice', help='Voice to use (name or number)', default=DEFAULT_VOICE)
        parser.add_argument('--lang', help='Language to use (code or number)', default=DEFAULT_LANG)
        parser.add_argument('--list', action='store_true', help='List available voices and languages')
        parser.add_argument('--kill', action='store_true', help='Kill all running TTS daemons')
        parser.add_argument('--log', action='store_true', help='Enable detailed logging')
        parser.add_argument('text', nargs='*', help='Text to speak')
        
        args = parser.parse_args()
        setup_logging(args.log)
        logger.debug(f"Parsed arguments: {args}")

        if args.kill:
            kill_all_daemons()
            sys.exit(0)

        script_dir = Path(__file__).parent.absolute()
        model_path, voices_path = ensure_model_and_voices(script_dir)
        logger.debug(f"Model path: {model_path}, Voices path: {voices_path}")
        
        try:                            
            logger.debug("Creating voice fetcher to get available voices")
            voice_fetcher = VoiceFetcher()
            available_voices = sorted(voice_fetcher.get_available_voices())
            logger.debug(f"Available voices: {available_voices}")
        except Exception as e:
            logger.error(f"Error initializing TTS system: {e}", exc_info=True)
            sys.exit(1)
        
        if args.list or len(sys.argv) == 1:
            show_help(available_voices)
            sys.exit(0)
        
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text)
        
        if not text:
            logger.error("No text provided")
            show_help(available_voices)
            sys.exit(1)
        
        voice = get_voice_from_input(args.voice, available_voices)
        if not voice:
            logger.error(f"Invalid voice '{args.voice}'")
            show_help(available_voices)
            sys.exit(1)
        
        lang = get_language_from_input(args.lang)
        if not lang:
            logger.error(f"Invalid language '{args.lang}'")
            show_help(available_voices)
            sys.exit(1)
        
        logger.debug(f"Processing text: '{text}' with voice: {voice}, language: {lang}")
        send_request(text, voice, lang)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass
        kill_all_daemons()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()