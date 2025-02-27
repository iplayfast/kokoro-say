#!/usr/bin/env python3

"""
Fetcher classes for downloading and managing voice models from Hugging Face.
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
    MODEL_PATH,
    VOICES_CONFIG_PATH,
    MODEL_URL,
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

class ModelFetcher:
    def __init__(self):
        self.model_url = MODEL_URL
        self.model_dir = MODEL_DIR
    
    def _create_voices_json(self, path: Path) -> None:
        """Create voices.json configuration file"""
        # Create base configuration
        config = {
            "speakers": {},
            "multispeaker": True,
            "speaker_embedding_channels": SPEAKER_EMBEDDING_CHANNELS,
            "speaker_embeddings_path": None,
            "speaker_embedding_dim": SPEAKER_EMBEDDING_DIM
        }
        
        # Add configurations for all known voices
        voice_fetcher = VoiceFetcher()
        for voice in voice_fetcher.get_available_voices():
            # Determine language based on prefix
            prefix = voice[:2]  # af, am, bf, bm, etc.
            
            # Determine if British
            is_british = prefix in ("bf", "bm")
            
            # Determine gender
            is_male = prefix[1] == "m"
            
            # Determine language code based on first letter of prefix
            language_map = {
                "a": "en-us",    # American English
                "b": "en-gb",    # British English
                "j": "ja",       # Japanese
                "z": "cmn",      # Mandarin Chinese
                "e": "es",       # Spanish
                "f": "fr-fr",    # French
                "h": "hi",       # Hindi
                "i": "it",       # Italian
                "p": "pt-br"     # Brazilian Portuguese
            }
            
            language = language_map.get(prefix[0], "en-us")
            
            config["speakers"][voice] = {
                "name": voice,
                "language": language,
                "gender": "male" if is_male else "female",
                "description": f"Voice model for {voice}",
                "embedding_path": str(voice_fetcher.voices_dir / f"{voice}.pt"),
                "symbols": None
            }
        
        # Write configuration file with explicit encoding
        with path.open('w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Verify the file was written correctly
        try:
            with path.open('r', encoding='utf-8') as f:
                verification = json.load(f)
            if len(verification["speakers"]) != len(config["speakers"]):
                raise ValueError("Configuration verification failed - speaker count mismatch")
            logger.info(f"Successfully created and verified voices configuration at {path}")
        except Exception as e:
            logger.error(f"Failed to verify voices configuration: {e}")
            if path.exists():
                path.unlink()
            raise RuntimeError(f"Failed to create valid voices configuration: {e}")
    
    def fetch_model(self) -> Tuple[Path, Path]:
        """
        Fetch model and create voices.json if needed
        
        Returns:
            Tuple[Path, Path]: Paths to the model file and voices.json file
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_PATH
        voices_json_path = VOICES_CONFIG_PATH

        # Simulate download for development/testing
        if os.environ.get("KOKORO_DEV_MODE") == "1":
            print(f"\nDEV MODE: Simulating download of Kokoro model...")
            # Create empty file for testing
            with model_path.open('wb') as f:
                f.write(b"SIMULATED MODEL FILE")
            time.sleep(1)  # Simulate download time
        # Download model if needed
        elif not model_path.exists():
            print(f"\nDownloading Kokoro model...")
            try:
                response = requests.get(self.model_url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with model_path.open('wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        downloaded = 0
                        for data in response.iter_content(chunk_size=8192):
                            downloaded += len(data)
                            f.write(data)
                            done = int(50 * downloaded / total_size)
                            sys.stdout.write(f"\rModel: |{'█' * done}{'-' * (50-done)}| {int(100 * downloaded / total_size)}%")
                            sys.stdout.flush()
                        print()
                        
            except Exception as e:
                if model_path.exists():
                    model_path.unlink()
                logger.error(f"Failed to download model: {str(e)}")
                raise RuntimeError(f"Failed to download model: {str(e)}")

        # Create voices.json if needed
        if not voices_json_path.exists():
            print(f"Creating voices configuration...")
            self._create_voices_json(voices_json_path)
        else:
            # Verify existing voices.json is valid and update with new voices if needed
            try:
                with voices_json_path.open('r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # Check if config has speakers
                if "speakers" not in config or not config["speakers"]:
                    logger.warning("Existing voices.json is invalid, recreating")
                    self._create_voices_json(voices_json_path)
                else:
                    # Get all available voices
                    voice_fetcher = VoiceFetcher()
                    available_voices = voice_fetcher.get_available_voices()
                    
                    # Check if any new voices are missing from the config
                    existing_voices = set(config["speakers"].keys())
                    missing_voices = set(available_voices) - existing_voices
                    
                    if missing_voices:
                        logger.info(f"Updating voices.json with {len(missing_voices)} new voices")
                        self._create_voices_json(voices_json_path)
                    
            except Exception as e:
                logger.warning(f"Existing voices.json is invalid, recreating: {e}")
                self._create_voices_json(voices_json_path)
        
        return model_path, voices_json_path

def ensure_model_and_voices(voice: str) -> Tuple[Path, Path]:
    """
    Ensure both model and specified voice exist
    
    Args:
        voice: Name of the voice to ensure exists
        
    Returns:
        tuple[Path, Path]: Paths to the model and voice files
    """
    try:
        # First get model and voices.json
        model_fetcher = ModelFetcher()
        model_path, voices_json_path = model_fetcher.fetch_model()
        
        # Then get the specific voice file
        voice_fetcher = VoiceFetcher()
        voice_path = voice_fetcher.fetch_voice(voice)
        
        # Verify both files exist
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at {model_path}")
        if not voices_json_path.exists():
            raise RuntimeError(f"Voices configuration not found at {voices_json_path}")
        if not voice_path.exists():
            raise RuntimeError(f"Voice file not found at {voice_path}")
            
        return model_path, voices_json_path
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise RuntimeError(f"Failed to ensure model and voices: {e}")