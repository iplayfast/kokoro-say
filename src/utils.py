#!/usr/bin/env python3

import os
import sys
import json
import logging
import warnings
import requests
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, List, Set, Tuple

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
        self.japanese_voices: Set[str] = set()
        self.chinese_voices: Set[str] = set()
        self.other_voices: Set[str] = set()
        self.cache_dir = Path.home() / ".cache" / "kokoro" / "voices"
        self.voices_config_path = self.cache_dir / "available_voices.json"
        self._discover_voices()
    
    def _discover_voices(self) -> None:
        """Discover available voices from cache and config files"""
        # First check for cached list from VoiceFetcher
        if self.voices_config_path.exists():
            try:
                with open(self.voices_config_path, 'r') as f:
                    all_voices = json.load(f)
                #logger.debug(f"Found {len(all_voices)} voices in cache")
                
                # Categorize by prefix
                for voice in all_voices:
                    if voice.startswith(("af_", "am_")):
                        self.american_voices.add(voice)
                    elif voice.startswith(("bf_", "bm_")):
                        self.british_voices.add(voice)
                    elif voice.startswith(("jf_", "jm_")):
                        self.japanese_voices.add(voice)
                    elif voice.startswith(("zf_", "zm_")):
                        self.chinese_voices.add(voice)
                    else:
                        self.other_voices.add(voice)
                
                # Return early if we found voices in the cache
                if self.american_voices or self.british_voices or self.japanese_voices or self.chinese_voices or self.other_voices:
                    #logger.info(f"Loaded {len(self.american_voices)} American, {len(self.british_voices)} British, "
                    #           f"{len(self.japanese_voices)} Japanese, {len(self.chinese_voices)} Chinese, "
                    #           f"and {len(self.other_voices)} other voices from cached list")
                    return
            except Exception as e:
                logger.error(f"Error reading cached voice list: {e}")
        
        # Fallback: Process all .pt files in the cache directory
        if not self.cache_dir.exists():
            # Add default voices to ensure we have something
            self.american_voices.update([
                "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                "am_adam", "am_michael"
            ])
            self.british_voices.update([
                "bf_emma", "bf_isabella", "bm_george", "bm_lewis"
            ])
            return
            
        # Look for voice files directly
        for voice_file in self.cache_dir.glob("*.pt"):
            voice_name = voice_file.stem  # Remove .pt extension
            if voice_name.startswith(("af_", "am_")):
                self.american_voices.add(voice_name)
            elif voice_name.startswith(("bf_", "bm_")):
                self.british_voices.add(voice_name)
            elif voice_name.startswith(("jf_", "jm_")):
                self.japanese_voices.add(voice_name)
            elif voice_name.startswith(("zf_", "zm_")):
                self.chinese_voices.add(voice_name)
            else:
                self.other_voices.add(voice_name)
                
        logger.info(f"Found {len(self.american_voices)} American, {len(self.british_voices)} British, "
                  f"{len(self.japanese_voices)} Japanese, {len(self.chinese_voices)} Chinese, "
                  f"and {len(self.other_voices)} other voices from voice files")
    
    @property
    def all_voices(self) -> List[str]:
        """Get list of all available voices"""
        return sorted(list(self.american_voices | self.british_voices | 
                          self.japanese_voices | self.chinese_voices | 
                          self.other_voices))
    
    def is_available(self, voice: str) -> bool:
        """Check if a voice is known (whether downloaded or not)"""
        return voice in self.all_voices
    
    def is_downloaded(self, voice: str) -> bool:
        """Check if a voice file exists in the cache"""
        voice_path = self.cache_dir / f"{voice}.pt"
        return voice_path.exists()
    
    def is_british_voice(self, voice: str) -> bool:
        """Check if a voice is British"""
        return voice in self.british_voices

    def ensure_voice_available(self, voice: str) -> bool:
        """
        Ensure a voice is available, downloading if necessary.
        Args:
            voice: Name of the voice to download
        Returns:
            bool: True if voice is available
        Raises:
            ValueError: If voice cannot be downloaded
        """
        # Check if this is a known voice
        if voice not in self.all_voices:
            raise ValueError(f"Unknown voice: {voice}")
        
        # Check if already downloaded
        voice_path = self.cache_dir / f"{voice}.pt"
        if voice_path.exists():
            return True
            
        # Need to download the voice
        from src.fetchers import VoiceFetcher
        fetcher = VoiceFetcher()
        try:
            fetcher.fetch_voice(voice)
            return True
        except Exception as e:
            raise ValueError(f"Failed to download voice {voice}: {str(e)}")

def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Play audio using sounddevice"""
    sd.play(audio, sample_rate)
    sd.wait()

def get_voice_from_input(voice_input: str, voice_manager: VoiceManager) -> str:
    """Convert voice input (name or number) to voice name"""
    all_voices = voice_manager.all_voices
    
    # Handle numeric input
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(all_voices):
            return all_voices[index]
        else:
            raise ValueError(f"Invalid voice number: {voice_input}. Must be between 1 and {len(all_voices)}")
    # Handle voice name input
    elif voice_input in all_voices:
        return voice_input
    # Try checking if it's a known voice prefix
    elif voice_input.startswith(("af_", "am_", "bf_", "bm_", "jf_", "jm_", "zf_", "zm_")):
        # This might be a voice that's in our list but not yet downloaded
        # First check if it's in our all_voices list
        if voice_input in all_voices:
            return voice_input
        # If not, raise an error
        raise ValueError(f"Voice '{voice_input}' is not in the available voice list. Run --update-voices to refresh the list.")
    else:
        # Unknown voice
        if len(all_voices) > 0:
            voice_examples = ", ".join(all_voices[:5])
            raise ValueError(f"Unknown voice: {voice_input}. Available voices include: {voice_examples}, etc.")
        else:
            raise ValueError(f"Unknown voice: {voice_input}. No voices available. Run --update-voices to fetch the voice list.")

def get_language_from_input(lang_input: str, languages: List[Tuple[str, str]]) -> str:
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
    american_voices = sorted(voice_manager.american_voices)
    british_voices = sorted(voice_manager.british_voices)
    japanese_voices = sorted(voice_manager.japanese_voices)
    chinese_voices = sorted(voice_manager.chinese_voices)
    other_voices = sorted(voice_manager.other_voices)
    
    index = 1
    
    if american_voices:
        print("\nAmerican English Voices:")
        for i, voice in enumerate(american_voices, index):
            print(f"  {i:2d}. {voice}")
        index += len(american_voices)
    
    if british_voices:
        print("\nBritish English Voices:")
        for i, voice in enumerate(british_voices, index):
            print(f"  {i:2d}. {voice}")
        index += len(british_voices)
    
    if japanese_voices:
        print("\nJapanese Voices:")
        for i, voice in enumerate(japanese_voices, index):
            print(f"  {i:2d}. {voice}")
        index += len(japanese_voices)
    
    if chinese_voices:
        print("\nMandarin Chinese Voices:")
        for i, voice in enumerate(chinese_voices, index):
            print(f"  {i:2d}. {voice}")
        index += len(chinese_voices)
    
    if other_voices:
        print("\nOther Voices:")
        for i, voice in enumerate(other_voices, index):
            print(f"  {i:2d}. {voice}")
    
    if not voice_manager.all_voices:
        print("\nNo voices found. Run a TTS command first to download voices.")
    
    print("\nLanguages:")
    for i, (code, name) in enumerate(languages, 1):
        print(f"  {i:2d}. {code:6s} - {name}")
    
    print("\nUsage examples:")
    cmd = os.path.basename(sys.argv[0])
    if american_voices:
        example_voice = american_voices[0]
    elif british_voices:
        example_voice = british_voices[0]
    else:
        example_voice = "af_bella"
        
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