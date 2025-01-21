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

class VoiceFetcher:
    def __init__(self):
        self.voices = [
            "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
        self.pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt"
    
    def fetch_voices(self, path: str = "voices.json") -> None:
        """Fetch voices if voices.json doesn't exist"""
        
        if os.path.exists(path):
            return
        
        print("\nFirst-time setup: Downloading voice models...")
        print("This may take several minutes but only needs to be done once.\n")
        
        voices_json: Dict[str, Any] = {}
        for i, voice in enumerate(self.voices, 1):
            url = self.pattern.format(voice=voice)
            print(f"Downloading voice {i}/{len(self.voices)}: {voice}")
            try:
                r = requests.get(url)
                r.raise_for_status()
                content = io.BytesIO(r.content)
                voice_data: np.ndarray = torch.load(content).numpy()
                voices_json[voice] = voice_data.tolist()
            except Exception as e:
                print(f"Error downloading voice {voice}: {str(e)}")
                if not voices_json:  # If no voices downloaded yet
                    raise RuntimeError("Failed to download any voices. Please check your internet connection and try again.")
                print("Continuing with downloaded voices...")
                break
        
        if not voices_json:
            raise RuntimeError("No voices were successfully downloaded")
        
        print("\nSaving voice data...")
        with open(path, "w") as f:
            json.dump(voices_json, f, indent=4)
        print(f"Voice setup complete! Saved to {path}\n")

    def get_available_voices(self) -> list[str]:
        """Return list of available voice names"""
        return self.voices.copy()


class ModelFetcher:
    def __init__(self):
        self.model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
    
    def fetch_model(self, path: str = "kokoro-v0_19.onnx") -> None:
        """Fetch model if it doesn't exist"""
        

        if os.path.exists(path):
            return
        
        print(f"\nDownloading Kokoro model to {path}")
        print("This may take several minutes but only needs to be done once.\n")
        
        try:
            r = requests.get(self.model_url)
            r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)
            print(f"Model download complete! Saved to {path}\n")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}\nURL: {self.model_url}")


def ensure_model_and_voices(script_dir: str | os.PathLike) -> tuple[str, str]:
    """
    Ensure both model and voices exist before starting
    
    Args:
        script_dir: Directory where models should be stored
        
    Returns:
        tuple[str, str]: Paths to the model and voices files
    
    Raises:
        RuntimeError: If downloads fail
    """
    script_dir = os.path.expanduser(script_dir)
    model_path = os.path.join(script_dir, "kokoro-v0_19.onnx")
    voices_path = os.path.join(script_dir, "voices.json")
    
    model_fetcher = ModelFetcher()
    voice_fetcher = VoiceFetcher()
    
    try:
        model_fetcher.fetch_model(model_path)
        voice_fetcher.fetch_voices(voices_path)
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        raise RuntimeError(f"Failed to ensure model and voices: {e}")
    
    return model_path, voices_path
