#!/usr/bin/env python3

"""
Fetcher classes for downloading and managing voice models and the TTS model.
"""

import os
import sys
import json
import io
import logging
import numpy as np
import requests
import torch
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceFetcher:
    def __init__(self):
        self.voices = [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
        # Use correct HuggingFace repository
        self.pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt"
        self.voices_dir = Path.home() / ".cache" / "kokoro" / "voices"
    
    def fetch_voices(self, voice: str) -> Path:
        """
        Fetch a specific voice if it doesn't exist
        
        Args:
            voice: Name of the voice to fetch
            
        Returns:
            Path to the voice file
        """
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        voice_path = self.voices_dir / f"{voice}.pt"
        
        if voice_path.exists():
            return voice_path
        
        print(f"\nDownloading voice model {voice}...")
        url = self.pattern.format(voice=voice)
        
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            
            # Show download progress
            total_size = int(r.headers.get('content-length', 0))
            with voice_path.open('wb') as f:
                if total_size == 0:
                    f.write(r.content)
                else:
                    downloaded = 0
                    for data in r.iter_content(chunk_size=8192):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r{voice}.pt: |{'█' * done}{'-' * (50-done)}| {int(100 * downloaded / total_size)}%")
                        sys.stdout.flush()
                    print()
                    
            return voice_path
            
        except Exception as e:
            if voice_path.exists():
                voice_path.unlink()  # Remove partial download
            raise RuntimeError(f"Failed to download voice {voice}: {str(e)}")

    def get_available_voices(self) -> list[str]:
        """Return list of available voice names"""
        return self.voices.copy()


class ModelFetcher:
    def __init__(self):
        # Correct URL from kokoro-cli.py
        self.model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
        self.model_dir = Path.home() / ".cache" / "kokoro" / "model"
    
    def fetch_model(self) -> Path:
        """
        Fetch model if it doesn't exist
        
        Returns:
            Path to the model file
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "kokoro-v0_19.onnx"

        if model_path.exists():
            return model_path
        
        print(f"\nDownloading Kokoro model...")
        
        try:
            r = requests.get(self.model_url, stream=True)
            r.raise_for_status()
            
            # Show download progress
            total_size = int(r.headers.get('content-length', 0))
            with model_path.open('wb') as f:
                if total_size == 0:
                    f.write(r.content)
                else:
                    downloaded = 0
                    for data in r.iter_content(chunk_size=8192):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\rModel: |{'█' * done}{'-' * (50-done)}| {int(100 * downloaded / total_size)}%")
                        sys.stdout.flush()
                    print()
            
            return model_path
            
        except Exception as e:
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            raise RuntimeError(f"Failed to download model: {str(e)}")


def ensure_model_and_voices(voice: str) -> tuple[Path, Path]:
    """
    Ensure both model and specified voice exist
    
    Args:
        voice: Name of the voice to ensure exists
        
    Returns:
        tuple[Path, Path]: Paths to the model and voice files
    
    Raises:
        RuntimeError: If downloads fail
    """
    try:
        model_fetcher = ModelFetcher()
        voice_fetcher = VoiceFetcher()
        
        model_path = model_fetcher.fetch_model()
        voice_path = voice_fetcher.fetch_voices(voice)
        
        return model_path, voice_path
        
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        raise RuntimeError(f"Failed to ensure model and voices: {e}")
