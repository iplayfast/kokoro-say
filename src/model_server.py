#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import time
import signal
import threading
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Union
from kokoro import KModel, KPipeline

from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICE_SERVER_BASE_PORT,
    AMERICAN_VOICES,
    BRITISH_VOICES
)
from src.fetchers import ensure_model_and_voices

logger = logging.getLogger(__name__)

class ModelServer:
    """Main model server that handles client connections and manages voice servers"""
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.voice_servers: Dict[str, dict] = {}  # {voice_id: {process, last_used}}
        self.running = True
        self.model = None
        self.pipeline = None
        self.voice_cache: Dict[str, Any] = {}  # Cache for loaded voice packs
        self.voice_cache_lock = threading.Lock()  # Thread safety for cache

    def start(self):
        """Start the model server by binding to the specified host and port"""
        try:
            # Initialize the first voice (default)
            self.initialize_model()
            
            # Bind to the specified host and port
            self.socket.bind((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
            self.socket.listen(5)
            
            logger.info(f"Model server listening on {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
            
            while self.running:
                try:
                    # Set a timeout for accept to allow checking self.running periodically
                    self.socket.settimeout(1.0)
                    try:
                        conn, addr = self.socket.accept()
                        conn.setblocking(True)
                        
                        # Process request in a new thread
                        request_thread = threading.Thread(target=self.handle_request, args=(conn,))
                        request_thread.daemon = True
                        request_thread.start()
                        
                    except socket.timeout:
                        # This allows us to check self.running periodically
                        continue
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Server error: {e}")
                        time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal server error: {e}")
            self.cleanup()
            raise

    def handle_request(self, conn):
        """Handle an individual client request"""
        try:
            # Receive request
            request = self.receive_request(conn)
            
            # Process request
            response = self.process_request(request)
            
            # Send response
            self.send_response(conn, response)
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            error_response = {
                "status": "error",
                "message": str(e)
            }
            self.send_response(conn, error_response)
        finally:
            conn.close()

    def send_response(self, conn, response):
        """Send a response with length prefix"""
        try:
            response_data = json.dumps(response).encode('utf-8')
            length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
            conn.sendall(length_prefix + response_data)
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    def receive_request(self, conn):
        """Receive a length-prefixed request"""
        try:
            length_data = conn.recv(9)
            if not length_data:
                raise RuntimeError("Empty request")
            
            try:
                expected_length = int(length_data.decode().strip())
            except ValueError as e:
                raise RuntimeError(f"Invalid length prefix: {e}")
            
            data = bytearray()
            remaining = expected_length
            
            while remaining > 0:
                chunk = conn.recv(min(8192, remaining))
                if not chunk:
                    raise RuntimeError("Connection closed while reading data")
                data.extend(chunk)
                remaining -= len(chunk)
            
            try:
                request = json.loads(data.decode())
                return request
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON: {e}")
                
        except Exception as e:
            logger.error(f"Error receiving request: {e}")
            raise

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

            # Handle get_voice_server command
            if request.get("command") == "get_voice_server":
                voice = request.get("voice")
                if not voice:
                    return {"status": "error", "message": "No voice specified"}
                
                # Combine both American and British voices
                all_voices = list(AMERICAN_VOICES.keys()) + list(BRITISH_VOICES.keys())
                
                try:
                    voice_index = all_voices.index(voice)
                    port = VOICE_SERVER_BASE_PORT + voice_index + 1
                    
                    return {
                        "status": "success", 
                        "port": port
                    }
                except ValueError:
                    return {"status": "error", "message": f"Voice {voice} not found"}

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
