#!/usr/bin/env python3

import os
import sys
import json
import socket
import select
import threading
import logging
import time
import signal
import sounddevice as sd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

from .base_server import UnixSocketServer
from .constants import (
    SOCKET_BASE_PATH,
    PID_BASE_PATH,
    MODEL_SERVER_SOCKET,
    SERVER_TIMEOUT
)
from .constants import MAX_CHUNK_SIZE, SENTENCE_END_MARKERS

class VoiceServer(UnixSocketServer):
    """Server for handling voice-specific TTS requests"""
    
    def _initialize_instance_vars(self, voice: str, lang: str):
        """Initialize all instance variables"""
        # Voice-specific attributes
        self.voice = voice
        self.lang = lang
        self.model_socket = MODEL_SERVER_SOCKET
        
        # Playback control - initialize these first
        self.current_playback_lock = threading.Lock()
        self.current_playback = None
        self.stream = None
        
        # Running state
        self.running = True
        
        # Audio primer
        self.primer_word = "hmm"
        
        self.logger.debug("Instance variables initialized")

    def __init__(self, voice: str, lang: str):
        """Initialize voice server"""
        # Calculate paths
        socket_path = f"{SOCKET_BASE_PATH}_{voice}_{lang}"
        pid_file = f"{PID_BASE_PATH}_{voice}_{lang}.pid"
        
        # Initialize base server
        super().__init__(socket_path, pid_file)
        
        # Initialize instance variables
        self._initialize_instance_vars(voice, lang)
        
        # Set up signal handlers
        signal.signal(signal.SIGCHLD, self.handle_child)
        
        self.logger.info(f"VoiceServer initialized for voice={voice}, lang={lang}")

    def _start_audio_playback(self, samples: np.ndarray, sample_rate: int) -> None:
        """Initialize and start audio playback"""
        try:
            # Configure sounddevice for main playback
            sd.default.reset()
            sd.default.device = None
            sd.default.latency = 'high'
            sd.default.dtype = samples.dtype
            sd.default.channels = 1
            sd.default.samplerate = sample_rate
            
            # Define callbacks
            def finished_callback():
                self.logger.debug("Audio finished")
                if self.stream:
                    self.stream.close()
                    self.stream = None
                self.current_playback = False
            
            # Create output stream
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=samples.dtype,
                finished_callback=finished_callback
            )
            
            # Start stream
            self.stream.start()
            
            # Small wait after stream start
            time.sleep(0.1)
            
            # Start playback
            sd.play(samples, sample_rate, blocking=False)
            self.current_playback = True
            
            self.logger.debug(
                f"Started playback: {len(samples)} samples "
                f"at {sample_rate}Hz"
            )
            
        except Exception as e:
            self.logger.error(f"Error in audio playback: {e}")
            raise

    def handle_client(self, conn: socket.socket) -> None:
        """Handle requests from clients"""
        model_sock = None
        try:
            conn.setblocking(True)
            conn.settimeout(SERVER_TIMEOUT)
            self.logger.debug("Start handling client connection")

            # Receive request from client
            request = self.receive_request(conn)
            original_text = request.get("text", "")
            self.logger.debug(f"Processing request: {original_text[:50]}...")
            
            # Add primer word to text
            modified_text = f"{self.primer_word} {original_text}"
            self.logger.debug(f"Modified text with primer: {modified_text[:50]}...")

            # Connect to model server
            model_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            model_sock.settimeout(SERVER_TIMEOUT)
            model_sock.connect(self.model_socket)
            
            # Forward request to model server
            self.send_request(model_sock, {
                "text": modified_text,
                "voice": self.voice,
                "lang": self.lang
            })
            
            # Read response from model server
            response = self.receive_response(model_sock)
            
            if response.get("status") == "success":
                try:
                    # Stop any current playback before starting new one
                    self.stop_current_playback()
                    
                    # Process audio data
                    samples = np.array(response["samples"], dtype=np.float32)
                    sample_rate = int(response["sample_rate"])
                    
                    with self.current_playback_lock:
                        self._start_audio_playback(samples, sample_rate)
                    
                    # Send success response to client
                    self.send_response(conn, {
                        "status": "success",
                        "message": "Audio playback started"
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing audio: {e}")
                    self.send_response(conn, {
                        "status": "error",
                        "message": f"Audio processing error: {str(e)}"
                    })
            else:
                # Forward error from model server
                self.send_response(conn, response)

        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
            try:
                self.send_response(conn, {
                    "status": "error",
                    "message": str(e)
                })
            except:
                pass
        finally:
            # Clean up connections
            if model_sock:
                try:
                    model_sock.close()
                except:
                    pass
            try:
                conn.close()
            except:
                pass

    def stop_current_playback(self):
        """Safely stop any current audio playback"""
        if hasattr(self, 'current_playback_lock'):
            with self.current_playback_lock:
                if self.current_playback:
                    try:
                        sd.stop()
                        if hasattr(self, 'stream') and self.stream:
                            self.stream.close()
                            self.stream = None
                        self.current_playback = False
                        self.logger.debug("Stopped current playback")
                    except Exception as e:
                        self.logger.error(f"Error stopping playback: {e}")

    def handle_child(self, signum: int, frame: Any) -> None:
        """Handle child process termination"""
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                self.logger.debug(f"Child process {pid} terminated with status {status}")
        except ChildProcessError:
            pass
        except Exception as e:
            self.logger.error(f"Error in handle_child: {e}")

    def send_request(self, sock: socket.socket, request: dict) -> None:
        """Send a request with proper length prefix"""
        request_data = json.dumps(request).encode('utf-8')
        length_prefix = f"{len(request_data):08d}\n".encode('utf-8')
        sock.sendall(length_prefix + request_data)

    def receive_response(self, sock: socket.socket) -> dict:
        """Receive a length-prefixed response with proper error handling"""
        # Read length prefix
        length_data = sock.recv(9)
        if not length_data:
            raise RuntimeError("Connection closed while reading length")
        
        try:
            expected_length = int(length_data.decode().strip())
            self.logger.debug(f"Expecting response of length {expected_length}")
        except ValueError:
            raise ValueError(f"Invalid length prefix: {length_data!r}")
        
        # Read response data in chunks
        response_data = bytearray()
        remaining = expected_length
        start_time = time.time()
        
        while remaining > 0:
            if time.time() - start_time > SERVER_TIMEOUT:
                raise socket.timeout("Timeout while reading response")
                
            chunk = sock.recv(min(8192, remaining))
            if not chunk:
                raise RuntimeError("Connection closed while reading response")
            
            response_data.extend(chunk)
            remaining -= len(chunk)
        
        try:
            return json.loads(response_data.decode())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

    def cleanup(self) -> None:
        """Clean up resources before shutdown"""
        self.stop_current_playback()
        super().cleanup()

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)