#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import sounddevice as sd
from datetime import datetime, timedelta

from src.constants import (
    SOCKET_BASE_PATH,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICE_SERVER_INACTIVE_TIMEOUT,
    SAMPLE_RATE,
    LOG_FILE
)

from src.socket_protocol import SocketProtocol

logger = logging.getLogger(__name__)

class AudioManager:
    """Manages audio playback with immediate interruption support."""
    
    def __init__(self):
        self.current_stream: Optional[sd.OutputStream] = None
        self.lock = threading.Lock()
        self.playing = False
        
    def stop_current(self):
        """Immediately stop any current playback."""
        with self.lock:
            if self.current_stream is not None:
                try:
                    self.current_stream.stop()
                    self.current_stream.close()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                finally:
                    self.current_stream = None
                    self.playing = False
                    
    def play(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
        """Play audio, stopping any current playback."""
        self.stop_current()
        
        with self.lock:
            try:
                # Make sure audio is a numpy array 
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                    
                # Ensure audio is properly shaped
                if len(audio.shape) == 1:
                    channels = 1
                else:
                    channels = audio.shape[1]
                
                self.current_stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    callback=None,
                    finished_callback=self._on_playback_finished
                )
                self.current_stream.start()
                self.playing = True
                self.current_stream.write(audio)
                logger.debug(f"Playing audio: {audio.shape} at {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Error starting playback: {e}")
                self.stop_current()
                
    def _on_playback_finished(self):
        """Callback when playback finishes."""
        with self.lock:
            self.playing = False
            if self.current_stream is not None:
                try:
                    self.current_stream.close()
                except:
                    pass
                self.current_stream = None
                logger.debug("Audio playback finished")

class VoiceServer:
    """Server handling voice-specific TTS pipeline and playback."""
    
    def __init__(self, voice: str, lang: str):
        self.voice = voice
        self.lang = lang
        self.running = True
        self.last_activity = datetime.now()
        
        # Initialize components
        self.audio_manager = AudioManager()
        self.pipeline = None
        self.monitor_thread = None
        self.socket_path = f"{SOCKET_BASE_PATH}_{voice}_{lang}"
        
        # Set up sockets
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        # Remove existing socket file if it exists
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
            
    def initialize_pipeline(self):
        """Initialize voice-specific KPipeline."""
        from kokoro import KPipeline
        
        try:
            # Determine pipeline language code based on voice prefix
            is_british = self.voice.startswith(('bf_', 'bm_'))
            pipeline_lang = 'b' if is_british else 'a'
            
            # Initialize pipeline without model (using central model server)
            self.pipeline = KPipeline(lang_code=pipeline_lang)
            logger.info(f"Initialized pipeline for {self.voice} ({self.lang})")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
                
    def start(self):
        """Start the voice server."""
        try:
            # Initialize pipeline if not already done
            if self.pipeline is None:
                self.initialize_pipeline()
            
            # Bind Unix domain socket
            self.server_sock.bind(self.socket_path)
            self.server_sock.listen(5)
            self.server_sock.settimeout(1.0)  # Allow for clean shutdown
            
            # Start activity monitor if not already running
            if self.monitor_thread is None:
                self.monitor_thread = threading.Thread(target=self._monitor_activity)
                self.monitor_thread.daemon = True
                self.monitor_thread.start()
            
            logger.info(f"Voice server for {self.voice}/{self.lang} listening on {self.socket_path}")
            
            while self.running:
                try:
                    conn, _ = self.server_sock.accept()
                    # Handle client in new thread
                    threading.Thread(target=self.handle_client, 
                                  args=(conn,), 
                                  daemon=True).start()
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
        """Handle client connection and synthesis request."""
        try:
            # Update activity timestamp
            self.last_activity = datetime.now()
            
            # Receive message using SocketProtocol
            try:
                data = SocketProtocol.receive_message(conn)
                request = json.loads(data.decode('utf-8'))
                logger.debug(f"Received request: {request}")
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                return
                
            # Handle exit command
            if request.get("command") == "exit":
                logger.info("Received exit command")
                self.running = False
                response = {"status": "shutting_down"}
                try:
                    SocketProtocol.send_message(conn, response)
                except:
                    pass
                return
                
            # Handle synthesis request
            if "text" in request:
                self.process_synthesis(request, conn)
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
                
    def process_synthesis(self, request: Dict, conn: socket.socket):
        """Process synthesis request through model server and play audio."""
        try:
            # Get synthesis parameters
            text = request["text"]
            speed = request.get("speed", 1.0)
            
            logger.info(f"Processing synthesis request: '{text[:30]}...' for voice {self.voice}")
            
            # Stop any current playback
            self.audio_manager.stop_current()
            
            # Send request to model server
            model_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            model_sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
            
            # Prepare model server request
            model_request = {
                "text": text,
                "voice": self.voice,
                "speed": speed,
                "lang": self.lang
            }
            
            # Send request (direct JSON for compatibility with current model server)
            model_sock.sendall(json.dumps(model_request).encode())
            
            # Receive response
            model_sock.settimeout(30.0)  # Longer timeout for synthesis
            response_data = model_sock.recv(10 * 1024 * 1024)  # Allow for large responses
            model_sock.close()
            
            if not response_data:
                raise RuntimeError("No response from model server")
                
            # Parse response
            response = json.loads(response_data.decode())
            
            if response.get("status") == "error":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Model server error: {error_msg}")
                SocketProtocol.send_message(conn, {"status": "error", "error": error_msg})
                return
                
            # Convert audio data to numpy array
            audio_data = np.array(response["audio"], dtype=np.float32)
            sample_rate = response.get("sample_rate", SAMPLE_RATE)
            
            logger.info(f"Received audio: {len(audio_data)} samples")
            
            # Play audio
            self.audio_manager.play(audio_data, sample_rate)
            
            # Send success response to client
            SocketProtocol.send_message(conn, {"status": "success"})
            
        except Exception as e:
            logger.error(f"Error processing synthesis: {e}")
            try:
                SocketProtocol.send_message(conn, {"status": "error", "error": str(e)})
            except:
                pass
        finally:
            try:
                # Close model server socket if still open
                if 'model_sock' in locals() and model_sock:
                    model_sock.close()
            except:
                pass
    
    def _monitor_activity(self):
        """Monitor for inactivity and shut down if inactive too long."""
        while self.running:
            try:
                if datetime.now() - self.last_activity > timedelta(seconds=VOICE_SERVER_INACTIVE_TIMEOUT):
                    logger.info(f"Voice server {self.voice}/{self.lang} inactive for {VOICE_SERVER_INACTIVE_TIMEOUT} seconds, shutting down")
                    self.running = False
                    break
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in activity monitor: {e}")
                
    def stop(self):
        """Stop the voice server gracefully."""
        logger.info("Stopping voice server")
        self.running = False
        
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")
        
        # Stop audio playback
        self.audio_manager.stop_current()
        
        # Close and remove Unix domain socket
        try:
            self.server_sock.close()
        except:
            pass
            
        try:
            os.unlink(self.socket_path)
        except:
            pass
        
        # Clear pipeline
        self.pipeline = None
