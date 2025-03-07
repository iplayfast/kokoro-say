#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import time
from pathlib import Path

from src import constants
from src.utils import VoiceManager, show_help
from src.tts_client import (
    ensure_model_server,
    send_text,
    validate_voice_and_language,
    kill_server
)

# Configure logging
logging.basicConfig(
    level=constants.DEFAULT_LOG_LEVEL,
    format=constants.LOG_FORMAT,
    handlers=[
        logging.FileHandler(constants.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Speech using Kokoro with voice persistence"
    )
    parser.add_argument('--voice', default='af_bella', help='Voice to use (name or number)')
    parser.add_argument('--lang', default='en-us', help='Language code or number')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--kill', action='store_true', help='Send kill command to servers')
    parser.add_argument('--list', action='store_true', help='List available voices and languages')
    parser.add_argument('--update-voices', action='store_true', help='Force update of available voices list')
    parser.add_argument('--interactive', action='store_true', 
                        help='Start interactive mode: read multiple lines from stdin and speak each one')
    parser.add_argument('--log-level', default=constants.DEFAULT_LOG_LEVEL, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--output', help='Save audio to specified WAV file path')
    parser.add_argument('text', nargs='*', help='Text to speak')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(args.log_level)
    
    voice_manager = VoiceManager()
    
    try:
        # Handle voice list update
        if args.update_voices:
            logger.info("Forcing update of voice list...")
            from src.fetchers import VoiceFetcher
            voice_fetcher = VoiceFetcher()
            voices = voice_fetcher.get_available_voices(force_update=True)
            print("\nAvailable voices updated:")
            for i, voice in enumerate(sorted(voices), 1):
                print(f"  {i:2d}. {voice}")
            return

        # Handle list command
        if args.list or len(sys.argv) == 1:
            show_help(voice_manager, constants.LANGUAGES)
            return

        # Handle kill command
        if args.kill:
            kill_server()
            # Wait briefly to allow server to fully shut down
            time.sleep(0.5)
            return

        # Get input text
        if not sys.stdin.isatty() and not args.interactive:
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text)
            
        if not text and not args.interactive:
            logger.error("No text provided")
            parser.print_help()            
            sys.exit(1)

        # Validate voice and language
        try:
            voice, lang = validate_voice_and_language(args.voice, args.lang)
        except ValueError as e:
            logger.error(str(e))
            parser.print_help()            
            sys.exit(1)
        
        # Ensure model server is running
        if not ensure_model_server():
            logger.error("Failed to start model server")            
            sys.exit(1)
            
        # Wait briefly to ensure server is fully initialized 
        time.sleep(0.5)
        
        # Interactive mode: continuously read from stdin and speak each line
        if args.interactive:
            print(f"Interactive mode enabled. Using voice: {voice}, language: {lang}, speed: {args.speed}")
            print("Enter text to speak (Ctrl+D or Ctrl+C to exit):")
            
            line_num = 1
            try:
                while True:
                    try:
                        # Print a prompt
                        print(f"\n[{line_num}] > ", end="", flush=True)
                        line = input()
                        
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        print(f"Speaking: {line}")
                        
                        # Generate output filename if --output is specified
                        output_file = None
                        if args.output:
                            # Create sequentially numbered files
                            base, ext = os.path.splitext(args.output)
                            if not ext:
                                ext = ".wav"
                            output_file = f"{base}_{line_num}{ext}"
                            
                        # Send text for synthesis
                        if not send_text(line, voice, lang, args.speed, output_file):
                            logger.error("Failed to send text for synthesis")
                            continue
                            
                        if output_file:
                            print(f"Audio saved to {output_file}")
                        else:
                            # Just a small delay to ensure audio starts playing
                            time.sleep(0.2)
                            
                        line_num += 1
                            
                    except EOFError:
                        # Exit on Ctrl+D
                        print("\nExiting interactive mode.")
                        break
            except KeyboardInterrupt:
                # Exit on Ctrl+C
                print("\nInterrupted. Exiting interactive mode.")
                
        else:
            # Normal mode: process single text
            if not send_text(text, voice, lang, args.speed, args.output):
                logger.error("Failed to send text for synthesis")
                sys.exit(1)
                
            if args.output:
                print(f"Audio saved to {args.output}")
            else:
                # Wait briefly for audio to start playing before exiting
                #print("Request sent successfully. Audio synthesis in progress...")            
                time.sleep(0.2)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    try:
        main()
    finally:
        logger.info("Exiting main program")
        sys.exit(0)
