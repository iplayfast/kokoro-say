#!/usr/bin/env python3

import os
import sys
import json
import logging
import warnings
import threading
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, Any, List, Union
from kokoro import KModel, KPipeline

from src.constants import CACHE_DIR, SAMPLE_RATE
from src.utils import VoiceManager
from src.fetchers import ensure_model_and_voices

logger = logging.getLogger(__name__)

class TTSManager:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.current_voice = None
        self.current_lang = None
        self.voice_manager = VoiceManager()
        self.voice_cache: Dict[str, Any] = {}  # Cache for loaded voice packs
        self.voice_cache_lock = threading.Lock()
        
        # Initialize model and pipeline
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize Kokoro TTS model and pipeline"""
        try:
            # Initialize model with GPU if available
            use_gpu = torch.cuda.is_available()
            device = 'cuda' if use_gpu else 'cpu'
            self.model = KModel().to(device).eval()
            
            # Initialize pipeline (will load voices as needed)
            self.pipeline = KPipeline(lang_code='a', model=False)
            
            # Add custom pronunciations
            self.pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _load_voice(self, voice: str) -> Any:
        """Load a voice pack with caching"""
        with self.voice_cache_lock:
            if voice in self.voice_cache:
                logger.debug(f"Using cached voice pack for {voice}")
                return self.voice_cache[voice]
            
            logger.debug(f"Loading voice pack for {voice}")
            pack = self.pipeline.load_voice(voice)
            self.voice_cache[voice] = pack
            return pack
    
    def process_text(
        self, 
        text: str, 
        voice: str, 
        lang: str,
        speed: float = 1.0,
        output_path: Optional[str] = None
    ) -> Generator[Tuple[str, str, np.ndarray], None, None]:
        """Process text and generate audio"""
        try:
            # Determine pipeline language code
            is_british_voice = self.voice_manager.is_british_voice(voice)
            requested_british = lang == "en-gb"
            lang_code = 'b' if (is_british_voice or requested_british) else 'a'
            
            # Get voice pack from cache
            pack = self._load_voice(voice)
            
            # Generate audio
            audio_segments = []
            for graphemes, ps, _ in self.pipeline(text, voice, speed):
                ref_s = pack[len(ps)-1]
                try:
                    audio = self.model(ps, ref_s, speed)
                    audio_data = audio.cpu().numpy()
                except Exception as e:
                    if torch.cuda.is_available():
                        logger.warning("GPU error, falling back to CPU")
                        self.model = self.model.to('cpu')
                        audio = self.model(ps, ref_s, speed)
                        audio_data = audio.cpu().numpy()
                    else:
                        raise
                
                audio_segments.append(audio_data)
                yield graphemes, ps, audio_data
            
            # Save to file if requested
            if output_path and audio_segments:
                logger.debug(f"Saving audio to {output_path}")
                final_audio = np.concatenate(audio_segments)
                sf.write(output_path, final_audio, SAMPLE_RATE)
                
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise

    def cleanup(self):
        """Clean up resources"""
        # Clear voice cache
        with self.voice_cache_lock:
            self.voice_cache.clear()
