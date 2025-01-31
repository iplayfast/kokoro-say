#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import List

# Filter specific PyTorch warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")
warnings.filterwarnings("ignore", message="Loading bf_* voice into American English pipeline")
warnings.filterwarnings("ignore", message="Loading bm_* voice into American English pipeline")

from src.tts_manager import TTSManager
from src.utils import show_help, get_voice_from_input, get_language_from_input, play_audio, VoiceManager
from src.constants import (
    LANGUAGES, DEFAULT_VOICE, DEFAULT_LANG,
    AMERICAN_VOICES, BRITISH_VOICES
)

logger = logging.getLogger(__name__)

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
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('text', nargs='*', help='Text to speak')
    return parser.parse_args()

def determine_lang_code(voice: str) -> str:
    """
    Determine the appropriate language code based on voice prefix
    
    Args:
        voice: Voice identifier (e.g., af_heart, bf_emma)
        
    Returns:
        str: Language code ('a' for American, 'b' for British)
    """
    return voice[0]  # First character determines accent ('a' or 'b')

def handle_voice_selection(voice_input: str, voice_manager: VoiceManager) -> tuple[str, str]:
    """
    Handle voice selection and downloading
    
    Args:
        voice_input: Voice name or number
        voice_manager: VoiceManager instance
        
    Returns:
        tuple[str, str]: Selected voice name and language code
    """
    # Define default voice if none specified
    voice = voice_input or DEFAULT_VOICE
    
    # Get actual voice name from input
    try:
        voice = get_voice_from_input(voice, voice_manager)
    except ValueError:
        # If voice doesn't exist yet, try to download the requested voice
        all_voices = {**AMERICAN_VOICES, **BRITISH_VOICES}
        if voice in all_voices:
            logger.info(f"Voice {voice} not found locally, attempting to download...")
            voice_manager.ensure_voice_available(voice)
            voice = get_voice_from_input(voice, voice_manager)
        else:
            raise
    
    # Determine language code from voice prefix
    lang_code = determine_lang_code(voice)
            
    return voice, lang_code

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
    
    # Set logging level for specific modules
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    
    try:
        # Initialize TTS manager with CPU preference
        tts = TTSManager(force_cpu=args.cpu)
        
        # Handle list command
        if args.list or len(sys.argv) == 1:
            show_help(tts.voice_manager, LANGUAGES)
            return

        # Get input text
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text)
            
        if not text:
            logger.error("No text provided")
            show_help(tts.voice_manager, LANGUAGES)
            sys.exit(1)

        # Validate voice and language
        try:
            voice, lang_code = handle_voice_selection(args.voice, tts.voice_manager)
            # Note: get_language_from_input is used for display/logging purposes
            display_lang = get_language_from_input(args.lang, LANGUAGES)
        except ValueError as e:
            logger.error(str(e))
            show_help(tts.voice_manager, LANGUAGES)
            sys.exit(1)
            
        # Process text
        logger.info(f"Processing text with voice {voice} (lang_code={lang_code}): {text[:50]}...")
        for graphemes, phonemes, audio in tts.process_text(
            text=text,
            voice=voice,
            lang=lang_code,  # Now using the actual language code for the pipeline
            speed=args.speed,
            output_path=args.output
        ):
            if args.log:
                logger.debug(f"Processing: {graphemes}")
                logger.debug(f"Phonemes: {phonemes}")
            
            if not args.output:  # Play audio if not saving to file
                logger.debug("Playing audio segment")
                play_audio(audio)
        
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
