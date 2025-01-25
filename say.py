#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
import socket
import time
import multiprocessing
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, List

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

def main():
    try:
        # Uncomment the following lines for default debugging
        # sys.argv = [
        #     'say.py', 
        #     '--voice', '3',   # Default voice
        #     '--lang', 'en-gb', # Default language 
        #     '--log',           # Enable detailed logging
        #     'hello world'      # Default text
        # ]

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
        parser.add_argument('--output', help='Output WAV file path')
        parser.add_argument('text', nargs='*', help='Text to speak', default='Hello, world!')
        
        args = parser.parse_args()
        
        # Uncomment to force default debugging parameters
        # args.voice = '3'
        # args.lang = 'en-gb'
        # args.log = True
        args.text = ['hello world']
        
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
            
            # Uncomment for full voice details debugging
            # print("Full Voice Details:")
            # for idx, voice in enumerate(available_voices):
            #     print(f"{idx+1}: {voice}")
            
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
        
        # Uncomment for additional debugging output
        # print(f"Debug: Selected Voice '{voice}', Language '{lang}'")
        # print(f"Debug: Text to process: '{text}'")
        
        logger.debug(f"Processing text: '{text}' with voice: {voice}, language: {lang}")
        
        # Ensure model server is running
        logger.debug("Ensuring model server is running")
        if not ensure_model_server_running():
            logger.error("Failed to start model server")
            sys.exit(1)
            
        socket_path = f"{SOCKET_BASE_PATH}_{voice}_{lang}"
        logger.debug(f"Using socket path: {socket_path}")
        
        if not os.path.exists(socket_path):
            logger.debug(f"Starting new voice server for {voice}_{lang}")
            process = multiprocessing.Process(
                target=start_server_process,
                args=(voice, lang)
            )
            process.start()
            logger.debug(f"Started voice server process {process.pid}")
            
            # Wait for socket to appear
            start_time = time.time()
            while not os.path.exists(socket_path):
                if time.time() - start_time > 30:
                    logger.error("Timeout waiting for voice server")
                    sys.exit(1)
                time.sleep(0.1)
            logger.debug("Voice server socket is ready")
        
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(30)
            client.connect(socket_path)
            logger.debug("Connected to voice server")

            request = {
                "text": text,
                "voice": voice,
                "lang": lang,
                "format": "wav" if args.output else "audio"
            }
            
            # Uncomment to inspect request details before sending
            # print("Debug Request Details:")
            # import json
            # print(json.dumps(request, indent=2))
            
            request_data = json.dumps(request).encode('utf-8')
            length_prefix = f"{len(request_data):08d}\n".encode('utf-8')
            client.sendall(length_prefix + request_data)
            
            length_data = client.recv(9)
            if not length_data:
                raise RuntimeError("Server closed connection while reading length")
            
            response = receive_server_response(client, length_data)
            if response.get("status") != "success":
                raise RuntimeError(response.get("message", "Unknown error"))
                
            if args.output:
                if "wav_data" in response:
                    with open(args.output, 'wb') as f:
                        f.write(response["wav_data"])
                    logger.info(f"Audio saved to {args.output}")
                else:
                    raise RuntimeError("No WAV data received")
            elif response.get("message")=="Audio playback started":
                # Audio is being handled by voice server
                time.sleep(0.1) # small delay to ensure playback starts
            else:
                raise RuntimeError("No audio data received")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
        finally:
            try:
                client.close()
            except:
                pass
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        try:
            sd.stop()
        except:
            pass
        kill_all_daemons()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

# Add other functions (start_server_process, show_help, receive_server_response) 
# as they were in the original script...

if __name__ == "__main__":
    main()
