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
            self.pipeline = KPipeline(lang_code='a', model=False)  # Start with English US
            
            # Add custom pronunciations
            self.pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            
            logger.info("Kokoro model and pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def start(self):
        """Start the model server"""
        try:
            # Initialize model
            self.initialize_model()

            # Set up signal handlers
            signal.signal(signal.SIGTERM, lambda signo, frame: self.handle_shutdown())
            signal.signal(signal.SIGINT, lambda signo, frame: self.handle_shutdown())

            # Set up socket
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
            self.socket.listen(5)
            self.socket.settimeout(1.0)  # Allow interruption for shutdown
            
            logger.info(f"Model server listening on {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")

            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    # Handle client in new thread
                    threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Accept error: {e}")

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.cleanup()

    def handle_client(self, conn: socket.socket):
        """Handle client connection"""
        try:
            # Set timeout for client operations
            conn.settimeout(SERVER_TIMEOUT)

            # Read length prefix
            length_data = conn.recv(9).decode().strip()
            if not length_data:
                raise ConnectionError("Connection closed while reading length prefix")

            expected_length = int(length_data)

            # Read message data
            data = bytearray()
            remaining = expected_length
            while remaining > 0:
                chunk = conn.recv(min(8192, remaining))
                if not chunk:
                    raise ConnectionError(f"Connection closed with {remaining} bytes remaining")
                data.extend(chunk)
                remaining -= len(chunk)

            # Send acknowledgment
            conn.send(b'ACK')

            # Process request
            request = json.loads(data.decode())
            response = self.process_request(request)

            # Send response
            response_data = json.dumps(response).encode()
            conn.send(f"{len(response_data):08d}\n".encode())
            conn.send(response_data)

            # Wait for final confirmation
            try:
                fin = conn.recv(3)
                if fin != b'FIN':
                    logger.warning(f"Client sent invalid final confirmation: {fin!r}")
            except socket.timeout:
                logger.warning("Timeout waiting for client's final confirmation")

        except Exception as e:
            logger.error(f"Error handling client: {e}")
            try:
                error_response = {
                    "status": "error",
                    "message": str(e)
                }
                response_data = json.dumps(error_response).encode()
                conn.send(f"{len(response_data):08d}\n".encode())
                conn.send(response_data)
            except:
                pass
        finally:
            try:
                conn.close()
            except:
                pass

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
                    # Ensure we have the voice
                    _, voice_path = ensure_model_and_voices(voice)
                    
                    # Load voice
                    pack = self.pipeline.load_voice(voice)
                    
                    # Generate audio
                    for _, ps, _ in self.pipeline(request["text"], voice, speed):
                        ref_s = pack[len(ps)-1]
                        try:
                            audio = self.model(ps, ref_s, speed)
                            # Convert to numpy array
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
