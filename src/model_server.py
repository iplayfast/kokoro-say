#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import time
import signal
import threading
import queue
from pathlib import Path
from typing import Dict, Set, Optional, Any, Union
import psutil
import torch
from kokoro import KModel, KPipeline

from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    SOCKET_BASE_PATH,
    SERVER_TIMEOUT,
    MAX_RETRIES
)
from src.fetchers import ensure_model_and_voices

logger = logging.getLogger(__name__)


class ModelServer:
    """Main model server that handles client connections and manages voice servers"""
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.voice_servers: Dict[str, dict] = {}  # {voice_id: {process, last_used}}
        self.running = True
        self.model = None
        self.pipeline = None
        self.voice_cache: Dict[str, Any] = {}  # Cache for loaded voice packs
        self.voice_cache_lock = threading.Lock()  # Thread safety for cache

    def initialize_model(self, voice: str = "af_heart"):
        """Initialize Kokoro TTS model"""
        try:
            # Ensure model and voice files exist
            model_path, voice_path = ensure_model_and_voices(voice)
            
            # Load model
            use_gpu = torch.cuda.is_available()
            device = 'cuda' if use_gpu else 'cpu'
            self.model = KModel().to(device).eval()
            
            # Initialize pipeline
            self.pipeline = KPipeline(lang_code='a', model=False)
            
            # Add custom pronunciations
            self.pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            
            # Pre-load default voice
            self._load_voice(voice)
            
            logger.info("Kokoro model and pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
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

    def process_request(self, request: dict) -> dict:
        """Process client request"""
        try:
            # Handle exit command
            if request.get("command") == "exit":
                self.running = False
                return {"status": "success", "message": "Server shutting down"}

            # Handle ping command
            if request.get("command") == "ping":
                return {"status": "success", "message": "pong"}

            # Handle TTS request
            if "text" in request:
                voice = request["voice"]
                speed = request.get("speed", 1.0)
                
                try:
                    # Get voice pack from cache
                    pack = self._load_voice(voice)
                    
                    # Generate audio
                    for _, ps, _ in self.pipeline(request["text"], voice, speed):
                        ref_s = pack[len(ps)-1]
                        try:
                            audio = self.model(ps, ref_s, speed)
                            audio_data = audio.cpu().numpy()
                            
                            return {
                                "status": "success",
                                "samples": audio_data.tolist(),
                                "sample_rate": 24000
                            }
                        except Exception as e:
                            if torch.cuda.is_available():
                                logger.warning("GPU error, falling back to CPU")
                                self.model = self.model.to('cpu')
                                audio = self.model(ps, ref_s, speed)
                                audio_data = audio.cpu().numpy()
                                return {
                                    "status": "success",
                                    "samples": audio_data.tolist(),
                                    "sample_rate": 24000
                                }
                            raise
                            
                except Exception as e:
                    raise RuntimeError(f"TTS generation failed: {e}")

            raise ValueError("Invalid request")

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def handle_shutdown(self):
        """Handle shutdown signal"""
        logger.info("Shutdown signal received")
        self.running = False

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        # Clear voice cache
        with self.voice_cache_lock:
            self.voice_cache.clear()
        try:
            self.socket.close()
        except:
            pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--script-dir", required=True, type=Path)
    parser.add_argument("--log-level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/tmp/tts_daemon.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:
        logger.info("Starting model server...")
        server = ModelServer(args.script_dir)
        server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        server.handle_shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        try:
            server.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        sys.exit(0)
