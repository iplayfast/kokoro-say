#!/usr/bin/env python3

"""
Fetcher classes for downloading and managing voice models from Hugging Face.
Modified to remove model download functionality.
"""

import os
import sys
import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import constants directly
from src.constants import (
    CACHE_DIR,
    VOICES_DIR,
    MODEL_DIR,
    VOICES_CONFIG_PATH,
    VOICES_BASE_URL,
    SPEAKER_EMBEDDING_DIM,
    SPEAKER_EMBEDDING_CHANNELS
)

logger = logging.getLogger(__name__)

class VoiceFetcher:
    def __init__(self):
        self.base_url = VOICES_BASE_URL
        self.voices_dir = VOICES_DIR
        
        # Default voice list as fallback
        self.default_voices = [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
        
        # Ensure voices directory exists
        self.voices_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_voices(self, force_update: bool = False) -> List[str]:
        """
        Get list of available voices from Hugging Face repository
        
        Args:
            force_update: Force update the voice list from the repository
            
        Returns:
            List of available voice names
        """
        cache_file = self.voices_dir / "available_voices.json"
        
        # If force update is requested or the cache doesn't exist or is older than 1 day
        if (force_update or 
            not cache_file.exists() or 
            (time.time() - cache_file.stat().st_mtime) > 86400):  # 1 day in seconds
            
            try:
                logger.info("Fetching list of available voices from Hugging Face...")
                
                # First attempt: try to get VOICES.md which has the comprehensive list
                voices_from_md = self._parse_voices_from_md()
                
                # Second attempt: try to list contents of /voices directory
                voices_from_listing = self._list_repository_voices()
                
                # Combine both methods for comprehensive coverage
                all_voices = set(voices_from_md + voices_from_listing)
                
                # Add our default voices to ensure they're included
                all_voices.update(self.default_voices)
                
                # Convert to sorted list
                voices = sorted(list(all_voices))
                
                if voices:
                    # Make sure directory exists
                    self.voices_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save the voices to cache
                    with cache_file.open('w') as f:
                        json.dump(voices, f)
                    
                    # Print the found voices for verification
                    logger.info(f"Found and cached {len(voices)} available voices")
                    logger.info(f"Sample voices: {', '.join(voices[:10])}")
                    
                    return voices
                else:
                    logger.warning("Failed to fetch voices from repository, using default list")
                    return self.default_voices
                    
            except Exception as e:
                logger.error(f"Error fetching available voices: {e}")
                if cache_file.exists():
                    try:
                        with cache_file.open('r') as f:
                            return json.load(f)
                    except:
                        pass
                return self.default_voices
        else:
            # Use the cached file
            try:
                with cache_file.open('r') as f:
                    voices = json.load(f)
                logger.debug(f"Using cached list of {len(voices)} voices")
                return voices
            except Exception as e:
                logger.error(f"Error reading cached voices: {e}")
                return self.default_voices
    
    def _parse_voices_from_md(self) -> List[str]:
        """Parse the VOICES.md file to extract all voice names"""
        try:
            logger.info("Attempting to fetch VOICES.md from Hugging Face...")
            response = requests.get("https://huggingface.co/hexgrad/Kokoro-82M/raw/main/VOICES.md", timeout=10)
            response.raise_for_status()
            
            voices = []
            lines = response.text.split("\n")
            
            logger.info(f"VOICES.md has {len(lines)} lines")
            
            # More straightforward approach: look for lines with voice IDs
            for line in lines:
                line = line.strip()
                
                # Skip lines that don't look like table rows
                if not line.startswith('|'):
                    continue
                    
                # Skip header and separator rows
                if "Name" in line or "----" in line:
                    continue
                    
                # Split into cells and clean up
                cells = [cell.strip() for cell in line.split('|')]
                
                # Check if we have enough cells and the first non-empty one might be a voice name
                if len(cells) > 1:
                    # The voice name should be in the first cell after the initial pipe
                    voice_cell = cells[1] if len(cells) > 1 else ""
                    
                    # Clean up voice name - remove formatting like ** and whitespace
                    voice_name = voice_cell.replace('*', '').replace('**', '').strip()
                    
                    # Handle escape sequences properly (use a regular expression instead)
                    import re
                    voice_name = re.sub(r'\\\_', '_', voice_name)
                    
                    # Look for names that match the pattern like af_heart, am_adam, etc.
                    if '_' in voice_name and len(voice_name.split('_')) == 2:
                        prefix = voice_name.split('_')[0]
                        
                        # Only accept voices with valid prefixes (2 chars, letters only)
                        if len(prefix) == 2 and prefix.isalpha():
                            logger.debug(f"Found voice: {voice_name}")
                            voices.append(voice_name)
            
            logger.info(f"Extracted {len(voices)} voices from VOICES.md")
            
            # Print the found voices for verification
            if voices:
                logger.info(f"First 10 found voices: {', '.join(voices[:10])}")
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to parse VOICES.md: {str(e)}")
            logger.exception("Detailed error:")
            return []
    
    def _list_repository_voices(self) -> List[str]:
        """List voices by querying the repository API"""
        logger.info("Attempting to list voices directory contents...")
        try:
            # Direct API request to list files
            response = requests.get(
                "https://huggingface.co/api/repos/hexgrad/Kokoro-82M/list/main/voices",
                timeout=10
            )
            
            # Handle HTTP errors explicitly
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return []
                
            # Save the raw content for debugging
            with open("/tmp/voices_list_debug.txt", "wb") as f:
                f.write(response.content)
            logger.info(f"Saved API response to /tmp/voices_list_debug.txt")
            
            # Parse the JSON response
            try:
                files = response.json()
                logger.info(f"Got {len(files)} items from repository API")
                
                voices = []
                for item in files:
                    path = item.get("path", "")
                    if path.endswith(".pt"):
                        voice_name = Path(path).stem
                        logger.debug(f"Found voice file: {path} -> {voice_name}")
                        voices.append(voice_name)
                
                logger.info(f"Extracted {len(voices)} voices from repository listing")
                return voices
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response.text[:200]}...")
                return []
                
        except Exception as e:
            logger.error(f"Failed to list repository voices: {str(e)}")
            logger.exception("Detailed error:")
            return []
    
    def fetch_voice(self, voice: str, cache_dir: Optional[Path] = None) -> Path:
        """Fetch a single voice file from Hugging Face."""
        if cache_dir is None:
            cache_dir = self.voices_dir
            
        # Get the list of available voices
        available_voices = self.get_available_voices()
            
        # Verify voice exists in known list
        if voice not in available_voices and voice not in self.default_voices:
            available = self.default_voices + [v for v in available_voices if v not in self.default_voices]
            raise ValueError(f"Voice '{voice}' not found. Available voices: {', '.join(available[:10])}...")
        
        voice_path = cache_dir / f"{voice}.pt"
        
        # Return cached file if it exists and not forcing update
        if voice_path.exists():
            logger.debug(f"Using cached voice file: {voice_path}")
            return voice_path
            
        # Download voice file using direct download URL
        url = f"{self.base_url}/{voice}.pt"
        logger.info(f"Downloading voice {voice} from {url}")
        
        try:
            # Create parent directories if they don't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Simulate download for development/testing
            if os.environ.get("KOKORO_DEV_MODE") == "1":
                print(f"\nDEV MODE: Simulating download of voice model {voice}...")
                # Create empty file for testing
                with voice_path.open('wb') as f:
                    f.write(b"SIMULATED VOICE MODEL")
                time.sleep(1)  # Simulate download time
                return voice_path
            
            # Real download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save with progress tracking
            total_size = int(response.headers.get('content-length', 0))
            
            print(f"\nDownloading voice model {voice}...")
            with voice_path.open('wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for data in response.iter_content(chunk_size=8192):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r{voice}.pt: |{'█' * done}{'-' * (50-done)}| {int(100 * downloaded / total_size)}%")
                        sys.stdout.flush()
                    print()
                    
            logger.info(f"Downloaded voice file to {voice_path}")
            return voice_path
            
        except Exception as e:
            if voice_path.exists():
                voice_path.unlink()
            logger.error(f"Failed to download voice {voice}: {str(e)}")
            raise RuntimeError(f"Failed to download voice {voice}: {str(e)}")

def ensure_voice(voice: str) -> Path:
    """
    Ensure the specified voice exists
    
    Args:
        voice: Name of the voice to ensure exists
        
    Returns:
        Path: Path to the voice file
    """
    try:
        # Download/verify the voice file
        voice_fetcher = VoiceFetcher()
        voice_path = voice_fetcher.fetch_voice(voice)
        
        # Verify the file exists
        if not voice_path.exists():
            raise RuntimeError(f"Voice file not found at {voice_path}")
            
        return voice_path
        
    except Exception as e:
        logger.error(f"Voice setup failed: {e}")
        raise RuntimeError(f"Failed to ensure voice exists: {e}")