#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import signal
import threading
import queue
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.constants import (
    MODEL_PATH,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICES_CONFIG_PATH
)

from src.socket_protocol import SocketProtocol

logger = logging.getLogger(__name__)

class ModelServer:
    """Central server that maintains single KModel instance and handles synthesis requests."""
    
    def __init__(self):
        self.running = True
        self.model = None
        self.pipeline = None
        self.model_lock = threading.Lock()
        self.request_queue = queue.Queue()
        self.synthesis_thread = None
        
        # Thread pool for handling synthesis requests
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # TCP socket for voice server communication
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        
    # Add to model_server.py initialization
    def _initialize(self):
        """Custom initialization."""
        # Ensure socket directory exists
        socket_dir = os.path.dirname(SOCKET_BASE_PATH)
        os.makedirs(socket_dir, exist_ok=True)
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
            
    def initialize_model(self):
        """Initialize the central KModel instance."""
        from kokoro import KPipeline
        import torch
        from src.fetchers import ensure_model_and_voices  # Explicit import

        try:
            # Ensure model and voices exist (using default voice)
            logger.info("Checking for model and voice files...")
            model_path, voices_json_path = ensure_model_and_voices("af_bella")
            
            # Initialize pipeline with model
            use_gpu = torch.cuda.is_available()
            device = 'cuda' if use_gpu else 'cpu'
            logger.info(f"Initializing KPipeline on {device}")
            
            # Initialize the pipeline with correct parameters
            self.pipeline = KPipeline(lang_code='a')
            self.model = self.pipeline.model
            
            # Move model to appropriate device
            if self.model is not None:
                self.model = self.model.to(device).eval()
                logger.info(f"Model initialization complete on {device}")
            else:
                logger.error("Failed to initialize model: model is None")
                raise RuntimeError("Model initialization failed - model is None")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
                
    def start(self):
        """Start the model server."""
        try:
            logger.info("Starting model server")
            # Initialize model if not already done
            if self.model is None:
                self.initialize_model()
                if self.model is None:
                    raise RuntimeError("Model initialization failed")
            
            # Start synthesis worker thread if not running
            if self.synthesis_thread is None:
                self.synthesis_thread = threading.Thread(target=self.synthesis_worker)
                self.synthesis_thread.daemon = True
                self.synthesis_thread.start()
            
            # Bind TCP socket if not already bound
            try:
                self.tcp_sock.bind((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
                self.tcp_sock.listen(5)
                self.tcp_sock.settimeout(1.0)  # Allow for clean shutdown
                logger.info(f"Model server listening on {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    logger.warning("Socket already bound, continuing...")
                else:
                    raise
            
            # Main server loop
            while self.running:
                try:
                    conn, addr = self.tcp_sock.accept()
                    logger.info(f"Accepted connection from {addr}")
                    self.thread_pool.submit(self.handle_client, conn)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Accept error: {e}")
                        
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            if not self.running:
                self.cleanup()

    def handle_client(self, conn: socket.socket):
        """Handle client connection and process requests."""
        try:
            # Receive data using standard protocol
            raw_data = conn.recv(8192)
            logger.debug(f"Received {len(raw_data)} bytes")
            
            try:
                # Decode the bytes to string
                data = raw_data.decode('utf-8')
                logger.debug(f"Decoded data: {data}")
            except UnicodeDecodeError as e:
                logger.error(f"Decoding error: {e}")
                return
            
            try:
                # Try to parse as JSON directly
                request = json.loads(data)
                logger.debug(f"Parsed request: {request}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Problematic data: {data}")
                return
                    
            # Process request
            if "command" in request and request["command"] == "exit":
                logger.info("Received exit command")
                self.running = False
                response = {"status": "shutdown_initiated"}
                conn.sendall(json.dumps(response).encode())
                return
                
            # Handle synthesis request
            if "text" in request:
                logger.info(f"Queueing synthesis request: {request['text'][:30]}...")
                self.request_queue.put((request, conn))
                    
        except Exception as e:
            logger.error(f"Error handling client request: {e}")
        finally:
            # Don't close the connection here - it will be closed after sending the response
            pass
    
    def synthesis_worker(self):
        """Worker thread that processes synthesis requests."""
        while self.running:
            try:
                # Get next request from queue with timeout
                try:
                    request, conn = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Process synthesis request
                try:
                    with self.model_lock:
                        if self.pipeline is None:
                            error_response = {
                                "status": "error",
                                "error": "Pipeline not initialized"
                            }
                            conn.sendall(json.dumps(error_response).encode())
                            continue
                            
                        # Set default values if not provided
                        voice = request.get("voice", "af_bella")
                        speed = request.get("speed", 1.0)
                        
                        logger.info(f"Synthesizing text with voice {voice} at speed {speed}")
                        
                        # Use pipeline to generate audio
                        audio_chunks = []
                        
                        # Process text through pipeline
                        try:
                            for graphemes, phonemes, audio in self.pipeline(
                                text=request["text"],
                                voice=voice,
                                speed=speed
                            ):
                                logger.debug(f"Generated audio chunk: {len(audio)} samples")
                                audio_chunks.append(audio)
                                
                            if not audio_chunks:
                                raise ValueError("No audio chunks generated")
                                
                            # Concatenate all chunks
                            final_audio = np.concatenate(audio_chunks)
                            
                            # Play the audio directly
                            import sounddevice as sd
                            sd.play(final_audio, 24000)
                            
                            # Send simplified response (without the full audio data)
                            response = {
                                "status": "success",
                                "message": "Audio is playing",
                                "audio_length": len(final_audio)
                            }
                            
                            logger.info(f"Playing audio: {len(final_audio)} samples")
                            conn.sendall(json.dumps(response).encode())
                            
                        except Exception as e:
                            logger.error(f"Pipeline execution error: {e}")
                            error_response = {
                                "status": "error",
                                "error": f"Pipeline execution error: {str(e)}"
                            }
                            conn.sendall(json.dumps(error_response).encode())
                        
                except Exception as e:
                    logger.error(f"Synthesis error: {e}")
                    error_response = {
                        "status": "error",
                        "error": str(e)
                    }
                    try:
                        conn.sendall(json.dumps(error_response).encode())
                    except:
                        pass
                    
                finally:
                    try:
                        conn.close()
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Worker error: {e}")
                try:
                    conn.close()
                except:
                    pass
                                
    def handle_shutdown(self, signo, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signo}, initiating shutdown")
        self.running = False
        
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")
        self.running = False
        
        # Stop synthesis thread
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=5)
            
        # Clean up thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Close socket
        try:
            self.tcp_sock.close()
        except:
            pass
        
        # Clear model
        self.model = None
        self.pipeline = None
