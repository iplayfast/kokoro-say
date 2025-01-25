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
        
        # Playback control
        self.current_playback_lock = threading.Lock()
        self.current_playback = None
        self.stream = None
        
        # Running state
        self.running = True
        
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
        """Initialize and start audio playback with warm-up tone"""
        try:
            self.logger.debug("Starting audio playback setup")
            
            # Configure sounddevice
            sd.default.reset()
            sd.default.device = None
            sd.default.latency = 'high'
            sd.default.dtype = samples.dtype
            sd.default.channels = 1
            sd.default.samplerate = sample_rate
            
            device_info = sd.query_devices(sd.default.device)
            self.logger.debug(f"Using audio device: {device_info}")
            
            # Create and start stream
            self.logger.debug("Creating output stream")
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=samples.dtype,
                callback=lambda *args: self.logger.debug(f"Audio callback: {args}")
            )
            
            self.logger.debug("Starting stream")
            self.stream.start()
            
            # Create and play warm-up tone
            duration = 0.1  # seconds
            frequency = 21  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            warmup_tone = 0.1 * np.sin(2 * np.pi * frequency * t)
            
            self.logger.debug("Playing warm-up tone")
            sd.play(warmup_tone, sample_rate, blocking=True)
            self.logger.debug("Warm-up tone complete")
            
            sd.wait()
            self.logger.debug("Starting main audio playback")
            
            sd.play(samples, sample_rate, blocking=False)
            self.current_playback = True
            
            self.logger.debug("Audio playback started successfully")
            
        except Exception as e:
            self.logger.error(f"Error in audio playback: {e}", exc_info=True)
            raise
    
        
    def _start_audio_playback1(self, samples: np.ndarray, sample_rate: int) -> None:
        """Initialize and start audio playback"""
        try:
            self.logger.debug("Starting audio playback setup")
            
            # Log input state
            self.logger.debug(f"Input samples shape: {samples.shape}, dtype: {samples.dtype}")
            self.logger.debug(f"Sample rate: {sample_rate}")
            self.logger.debug(f"Sample range: min={np.min(samples)}, max={np.max(samples)}")
            
            # Get available audio devices
            devices = sd.query_devices()
            self.logger.debug(f"Available audio devices:\n{devices}")
            
            # Stop any existing playback
            sd.stop()
            if self.stream:
                self.stream.close()
                self.stream = None
                self.logger.debug("Closed existing stream")
            
            # Configure sounddevice
            sd.default.reset()
            sd.default.device = None  # Use system default
            sd.default.latency = 'high'
            sd.default.dtype = np.float32
            sd.default.channels = 1
            sd.default.samplerate = sample_rate
            
            current_device = sd.query_devices(sd.default.device)
            self.logger.debug(f"Selected device settings: {current_device}")
            
            # Normalize samples
            max_amp = np.max(np.abs(samples))
            if max_amp > 0:
                samples = 0.95 * (samples / max_amp)
                self.logger.debug(f"Normalized sample range: min={np.min(samples)}, max={np.max(samples)}")
            
            # Create stream with logging callbacks
            def stream_callback(outdata, frames, time, status):
                if status:
                    self.logger.warning(f"Stream callback status: {status}")
                    
            def finished_callback():
                self.logger.debug("Stream finished callback triggered")
                if self.stream:
                    self.stream.close()
                    self.stream = None
                self.current_playback = False
            
            self.logger.debug("Creating output stream")
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                callback=stream_callback,
                finished_callback=finished_callback
            )
            
            self.logger.debug("Starting stream")
            self.stream.start()
            
            if not self.stream.active:
                raise RuntimeError("Stream failed to start")
            
            self.logger.debug("Starting audio playback")
            sd.play(samples, sample_rate, blocking=False)
            self.current_playback = True
            
            self.logger.debug("Audio playback started successfully")
            
            # Keep main thread alive briefly to ensure playback starts
            time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"Error in audio playback: {e}", exc_info=True)
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
            text = request.get("text", "")
            output_format = request.get("format", "audio")
            self.logger.debug(f"Processing request: {text[:50]}...")

            # Connect to model server
            model_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            model_sock.settimeout(SERVER_TIMEOUT)
            model_sock.connect(self.model_socket)
        
            # Forward request to model server
            self.send_request(model_sock, {
                "text": text,
                "voice": self.voice,
                "lang": self.lang
            })
            
            # Read response from model server
            response = self.receive_response(model_sock)
            self.logger.debug("Received response from model server")
            
            if response.get("status") == "success":
                try:
                    samples = np.array(response["samples"], dtype=np.float32)
                    sample_rate = int(response["sample_rate"])
                    self.logger.debug(f"Processing audio data: shape={samples.shape}, rate={sample_rate}")
                
                    if output_format == "wav":
                        self.logger.debug("Generating WAV file output")
                        import io
                        import soundfile as sf
                        wav_buffer = io.BytesIO()
                        sf.write(wav_buffer, samples, sample_rate, format='WAV')
                        wav_data = wav_buffer.getvalue()
                        self.logger.debug(f"Generated WAV file of {len(wav_data)} bytes")
                    
                        self.send_response(conn, {
                            "status": "success",
                            "samples": samples.tolist(),
                            "sample_rate": sample_rate,
                            "wav_data": wav_data
                        })
                        self.logger.debug("WAV data sent to client")
                    else:
                        self.logger.debug("Starting audio playback")
                        with self.current_playback_lock:
                            self._start_audio_playback(samples, sample_rate)
                    
                        self.send_response(conn, {
                            "status": "success",
                            "message": "Audio playback started"
                        })
                    
                except Exception as e:
                    self.logger.error(f"Error processing audio: {e}", exc_info=True)
                    self.send_response(conn, {
                        "status": "error",
                        "message": f"Audio processing error: {str(e)}"
                    })
            else:
                self.logger.error(f"Model server error: {response.get('message', 'Unknown error')}")
                self.send_response(conn, response)

        except Exception as e:
            self.logger.error(f"Error handling client: {e}", exc_info=True)
            try:
                self.send_response(conn, {
                    "status": "error",
                    "message": str(e)
                })
            except:
                pass
        finally:
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
