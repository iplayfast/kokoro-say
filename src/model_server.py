#!/usr/bin/env python3

import os
import sys
import json
import socket
import threading
import select
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from kokoro_onnx import Kokoro

from .base_server import UnixSocketServer
from .constants import MODEL_SERVER_SOCKET, MODEL_PID_FILE, SERVER_TIMEOUT
from .fetchers import VoiceFetcher

class ModelServer(UnixSocketServer):
    """Central server that manages the shared Kokoro model"""
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'ModelServer':
        """Get or create singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(MODEL_SERVER_SOCKET, MODEL_PID_FILE)
        return cls._instance
    
    def __init__(self, socket_path: str, pid_file: str):
        """Initialize the model server"""
        if ModelServer._instance is not None:
            raise RuntimeError("Use get_instance() instead")
            
        # Initialize file descriptors before anything else
        null_fd = os.open(os.devnull, os.O_RDWR)
        if null_fd != 0:
            os.dup2(null_fd, 0)
        if null_fd != 1:
            os.dup2(null_fd, 1)
        if null_fd != 2:
            os.dup2(null_fd, 2)
            
        # Now initialize base server
        super().__init__(socket_path, pid_file)
        
        self.kokoro: Optional[Kokoro] = None
        self.voice_fetcher = VoiceFetcher()
        self.ready = threading.Event()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Kokoro model"""
        script_dir = Path(__file__).parent.parent.absolute()
        model_path = script_dir / "kokoro-v0_19.onnx"
        voices_path = script_dir / "voices.json"
        
        self.voice_fetcher.fetch_voices(str(voices_path))
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please download it from: https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
        
        try:
            self.kokoro = Kokoro(str(model_path), str(voices_path))
            self.logger.info("Kokoro model loaded in central model server")
            self.ready.set()
        except Exception as e:
            self.logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Wait until the model server is fully initialized"""
        return self.ready.wait(timeout)

    def send_chunked_response(self, conn: socket.socket, response: dict) -> None:
        """Send response in chunks to handle large data"""
        try:
            response_data = json.dumps(response).encode('utf-8')
            length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
            
            # Send length prefix
            conn.sendall(length_prefix)
            
            # Send response data in chunks
            chunk_size = 8192
            start_time = time.time()
            sent = 0
            
            while sent < len(response_data):
                if time.time() - start_time > SERVER_TIMEOUT:
                    raise socket.timeout("Timeout while sending response")
                
                chunk = response_data[sent:sent + chunk_size]
                conn.sendall(chunk)
                sent += len(chunk)
                
            self.logger.debug(f"Successfully sent response of {len(response_data)} bytes")
            
        except Exception as e:
            self.logger.error(f"Error sending chunked response: {e}")
            raise
        
    def handle_client(self, conn: socket.socket) -> None:
        """Handle requests from voice servers"""
        try:
            conn.setblocking(True)
            conn.settimeout(SERVER_TIMEOUT)
            self.logger.debug("Start handling client connection")
            self.update_activity()
            
            # Receive request with timeout
            request = self.receive_request(conn)
            self.logger.debug(f"Received request: {request}")
            
            # Handle ping requests
            if request.get("type") == "ping":
                self.logger.debug("Processing ping request")
                response = {
                    "status": "ok",
                    "timestamp": time.time()
                }
                self.send_response(conn, response)
                return
                
            # Handle TTS requests
            text = request.get('text')
            voice = request.get('voice')
            lang = request.get('lang')
            
            if not all([text, voice, lang]):
                self.logger.error(f"Incomplete request: text={text}, voice={voice}, lang={lang}")
                response = {
                    "status": "error",
                    "message": "Incomplete request parameters"
                }
                self.send_response(conn, response)
                return
            
            self.logger.debug(f"Processing TTS request: text='{text}', voice={voice}, lang={lang}")
            
            # Process TTS request with shared model
            if not self.wait_until_ready():
                raise RuntimeError("Model server not ready")
            
            with self.lock:
                samples, sample_rate = self.kokoro.create(
                    text,
                    voice=voice,
                    speed=1.0,
                    lang=lang
                )
                
            self.logger.debug("TTS processing successful")
            
            # Prepare response
            response = {
                "status": "success",
                "samples": samples.tolist(),
                "sample_rate": sample_rate
            }
            
            # Send the response
            self.send_chunked_response(conn, response)
            self.logger.debug("Response sent successfully")
                
        except Exception as e:
            self.logger.error(f"Error in ModelServer client handler: {e}")
            error_response = {
                "status": "error",
                "message": str(e)
            }
            try:
                self.send_response(conn, error_response)
            except Exception as send_err:
                self.logger.error(f"Error sending error response: {send_err}")
        finally:
            try:
                conn.close()
            except:
                pass

    def send_response(self, conn: socket.socket, response: dict) -> None:
        """Send a response with proper length prefix"""
        try:
            response_data = json.dumps(response).encode('utf-8')
            length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
            conn.sendall(length_prefix + response_data)
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            raise

    def send_chunked_response(self, conn: socket.socket, response: dict) -> None:
        """Send a large response in chunks"""
        try:
            response_data = json.dumps(response).encode('utf-8')
            length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
            
            # Send length prefix
            conn.sendall(length_prefix)
            
            # Send response data in chunks
            chunk_size = 8192
            sent = 0
            while sent < len(response_data):
                chunk = response_data[sent:sent + chunk_size]
                conn.sendall(chunk)
                sent += len(chunk)
                
            self.logger.debug(f"Successfully sent {sent} bytes")
        except Exception as e:
            self.logger.error(f"Error sending chunked response: {e}")
            raise
                
        def start(self) -> None:
            """Start the server with proper initialization checks"""
            if self.is_running():
                self.logger.error("Server already running")
                sys.exit(1)

            try:
                # Daemonize the process
                self.daemonize()
                
                # Create and bind socket
                self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server.bind(self.socket_path)
                self.server.listen(5)
                self.server.setblocking(False)
                
                self.logger.info(f"Model server listening on {self.socket_path}")
                
                # Main server loop
                while self.running:
                    try:
                        ready = select.select([self.server], [], [], 1.0)
                        if ready[0]:
                            conn, _ = self.server.accept()
                            conn.setblocking(True)
                            thread = threading.Thread(target=self.handle_client, args=(conn,))
                            thread.daemon = True
                            thread.start()
                    except socket.error as e:
                        if self.running:  # Only log if we're still meant to be running
                            self.logger.error(f"Socket error: {e}")
                            time.sleep(1)
                    except Exception as e:
                        if self.running:
                            self.logger.error(f"Server error: {e}")
                            time.sleep(1)
                            
            except Exception as e:
                self.logger.error(f"Fatal server error: {e}")
                self.cleanup()
                raise