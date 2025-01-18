#!/usr/bin/env python3
"""
Text-to-speech script using kokoro-onnx with support for voice and language selection.
"""
import sys
import os
import argparse
import sounddevice as sd
from kokoro_onnx import Kokoro

# Available languages with descriptions
LANGUAGES = [
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)"),
    ("fr-fr", "French"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("cmn", "Mandarin Chinese")
]

def get_voice_from_input(voice_input, voices):
    """Convert voice input (name or number) to voice name"""
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(voices):
            return voices[index]
    elif voice_input in voices:
        return voice_input
    return None

def get_language_from_input(lang_input):
    """Convert language input (code or number) to language code"""
    if lang_input.isdigit():
        index = int(lang_input) - 1
        if 0 <= index < len(LANGUAGES):
            return LANGUAGES[index][0]
    else:
        for code, _ in LANGUAGES:
            if code == lang_input:
                return code
    return None

def show_help(kokoro):
    """Display help message with available voices and languages"""
    voices = sorted(list(kokoro.get_voices()))
    
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"  {i:2d}. {voice}")
    
    print("\nAvailable languages:")
    for i, (code, name) in enumerate(LANGUAGES, 1):
        print(f"  {i:2d}. {code:6s} - {name}")
    
    print("\nUsage examples:")
    print(f"  1. List options:")
    print(f"     {sys.argv[0]} --help")
    print(f"\n  2. Using voice and language by name/code:")
    print(f"     {sys.argv[0]} --voice {voices[0]} --lang en-us \"Hello World\"")
    print(f"\n  3. Using voice and language by number:")
    print(f"     {sys.argv[0]} --voice 1 --lang 3 \"Bonjour le monde\"")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action='store_true')
    parser.add_argument('--voice', help='voice name or number')
    parser.add_argument('--lang', help='language code or number', default='en-us')
    parser.add_argument('text', nargs='*', help='text to speak')
    args = parser.parse_args()

    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
    voices = sorted(list(kokoro.get_voices()))
    
    if args.help:
        show_help(kokoro)
        return

    # Get input text from pipe or arguments
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = ' '.join(args.text) if args.text else None

    if not text:
        print("Error: No text provided")
        show_help(kokoro)
        return

    # Process voice selection
    selected_voice = voices[0]
    if args.voice:
        voice = get_voice_from_input(args.voice, voices)
        if not voice:
            print(f"Error: Invalid voice '{args.voice}'")
            show_help(kokoro)
            return
        selected_voice = voice

    # Process language selection
    selected_lang = get_language_from_input(args.lang)
    if not selected_lang:
        print(f"Error: Invalid language '{args.lang}'")
        show_help(kokoro)
        return

    # Get language name for display
    lang_name = next(name for code, name in LANGUAGES if code == selected_lang)
    
    print(f"Playing audio using voice: {selected_voice}, language: {lang_name}...")
    samples, sample_rate = kokoro.create(text, voice=selected_voice, speed=1.0, lang=selected_lang)
    sd.play(samples, sample_rate)
    sd.wait()

if __name__ == "__main__":
    main()
