#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple

import sounddevice as sd
import numpy as np

from src.constants import (
    CACHE_DIR, SAMPLE_RATE, 
    AMERICAN_VOICES, BRITISH_VOICES,
    VOICE_PREFIX_MEANINGS
)

logger = logging.getLogger(__name__)

class VoiceManager:
    def __init__(self):
        self.american_voices: Set[str] = set()
        self.british_voices: Set[str] = set()
        self.voices_dir = CACHE_DIR / "voices"
        self._discover_voices()
    
    def _discover_voices(self) -> None:
        """Discover available voices by checking the cache directory"""
        # Add all known voices from constants
        self.american_voices.update(AMERICAN_VOICES.keys())
        self.british_voices.update(BRITISH_VOICES.keys())
        
        if not self.voices_dir.exists():
            return
            
        # Update with any additional voices found in cache
        for voice_file in self.voices_dir.glob("*.pt"):
            voice_name = voice_file.stem  # Remove .pt extension
            if voice_name.startswith(("af_", "am_")):
                self.american_voices.add(voice_name)
            elif voice_name.startswith(("bf_", "bm_")):
                self.british_voices.add(voice_name)
    
    @property
    def all_voices(self) -> List[str]:
        """Get list of all available voices"""
        return sorted(list(self.american_voices | self.british_voices))
    
    def is_british_voice(self, voice: str) -> bool:
        """Check if a voice is British"""
        return voice in self.british_voices

    def is_available(self, voice: str) -> bool:
        """Check if a voice is available"""
        return voice in (self.american_voices | self.british_voices)
    
    def get_voice_info(self, voice: str) -> str:
        """Get descriptive information for a voice"""
        if voice in AMERICAN_VOICES:
            return AMERICAN_VOICES[voice]
        if voice in BRITISH_VOICES:
            return BRITISH_VOICES[voice]
        return voice  # Fallback to voice name if not found in descriptions

    def ensure_voice_available(self, voice: str) -> None:
        """Ensure a voice is available, downloading if necessary"""
        from src.fetchers import VoiceFetcher
        fetcher = VoiceFetcher()
        fetcher.fetch_voices(voice)
        self._discover_voices()  # Refresh available voices

def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play audio using sounddevice"""
    sd.play(audio, sample_rate)
    sd.wait()

def get_voice_from_input(voice_input: str, voice_manager: VoiceManager) -> str:
    """Convert voice input (name or number) to voice name"""
    all_voices = voice_manager.all_voices
    
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(all_voices):
            return all_voices[index]
    elif voice_input in all_voices:
        return voice_input
    raise ValueError(f"Invalid voice: {voice_input}")

def get_language_from_input(lang_input: str, languages: List[tuple[str, str]]) -> str:
    """Convert language input (code or number) to language code"""
    if lang_input.isdigit():
        index = int(lang_input) - 1
        if 0 <= index < len(languages):
            return languages[index][0]
    else:
        for code, _ in languages:
            if code == lang_input:
                return code
    raise ValueError(f"Invalid language: {lang_input}")

def show_help(voice_manager: VoiceManager, languages: List[Tuple[str, str]]) -> None:
    """Display available voices and languages"""
    if voice_manager.american_voices:
        print("\nAmerican English Voices:")
        for i, voice in enumerate(sorted(voice_manager.american_voices), 1):
            print(f"  {i:2d}. {voice} - {voice_manager.get_voice_info(voice)}")
    
    if voice_manager.british_voices:
        base_index = len(voice_manager.american_voices) + 1
        print("\nBritish English Voices:")
        for i, voice in enumerate(sorted(voice_manager.british_voices), base_index):
            print(f"  {i:2d}. {voice} - {voice_manager.get_voice_info(voice)}")
    
    if not (voice_manager.american_voices or voice_manager.british_voices):
        print("\nNo voices found. Run a TTS command first to download voices.")
    
    print("\nLanguages:")
    for i, (code, name) in enumerate(languages, 1):
        print(f"  {i:2d}. {code:6s} - {name}")
    
    print("\nVoice naming convention:")
    for prefix, meaning in VOICE_PREFIX_MEANINGS.items():
        print(f"  {prefix}_* - {meaning} voices")
    
    print("\nUsage examples:")
    cmd = os.path.basename(sys.argv[0])
    if voice_manager.american_voices:
        example_voice = next(iter(sorted(voice_manager.american_voices)))
    elif voice_manager.british_voices:
        example_voice = next(iter(sorted(voice_manager.british_voices)))
    else:
        example_voice = "af_heart"
        
    print(f"  1. Basic usage:")
    print(f"     {cmd} --voice {example_voice} \"Hello World\"")
    print(f"\n  2. Using voice by number:")
    print(f"     {cmd} --voice 1 \"Hello World\"")
    print(f"\n  3. Adjusting speech speed:")
    print(f"     {cmd} --speed 1.2 \"Hello World\"")
    print(f"\n  4. Using pipe input:")
    print(f"     echo \"Hello World\" | {cmd}")
    print(f"\n  5. Saving to WAV file:")
    print(f"     {cmd} --output speech.wav \"Hello World\"")
