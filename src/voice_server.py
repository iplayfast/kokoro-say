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
    """Manages audio playback with support for concurrent voices."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.playing = False
        
    def stop_current(self):
        """Immediately stop any current playback."""
        with self.lock:
            try:
                sd.stop()
                logger.debug("Stopped all playback")
            except Exception as e:
                logger.error(f"Error stopping playback: {e}")
            finally:
                self.playing = False
                    
    def play(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
        """Play audio using non-blocking approach with improved performance."""
        # First stop any current playback
        self.stop_current()
        
        # Ensure audio is a float32 array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        
        # Log audio details including duration
        audio_duration = len(audio) / sample_rate
        logger.info(f"Playing audio: {len(audio)} samples, {audio_duration:.2f} seconds, " 
                   f"min={audio.min():.4f}, max={audio.max():.4f}")
        
        try:
            # Use a smaller buffer size for quicker start
            buffer_size = min(2048, len(audio))
            
            # Try using low latency for faster startup
            sd.play(audio, samplerate=sample_rate, blocking=False, 
                    latency='low', blocksize=buffer_size)
            
            # Force a small delay to let playback start
            time.sleep(0.1)
            
            # Verify the stream is active
            stream = sd.get_stream()
            if stream and stream.active:
                logger.info(f"Stream is active with latency: {stream.latency:.4f}s")
            else:
                logger.warning("Stream is not active, trying fallback playback...")
                # Try with different settings
                sd.stop()
                sd.play(audio, samplerate=sample_rate, blocking=False)
                time.sleep(0.1)
                
                # Check again
                stream = sd.get_stream()
                if stream and stream.active:
                    logger.info("Fallback stream is active")
                else:
                    logger.error("Failed to create active audio stream!")
                
            self.playing = True
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            self.playing = False
    
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
        
        # Set up logging for this specific voice server
        file_handler = logging.FileHandler(f"/tmp/voice_server_{voice}_{lang}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Log startup
        logger.info(f"=== VOICE SERVER STARTING: {voice}/{lang} ===")
        logger.info(f"Process ID: {os.getpid()}")
        
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
                    # Handle client directly - no threading
                    # This ensures each request is processed sequentially
                    # and can properly interrupt previous synthesis
                    self.handle_client(conn)
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
            thread_id = threading.get_ident()
            logger.info(f"Processing synthesis in thread {thread_id}")
            start_time = time.time()
            
            # Get synthesis parameters
            text = request["text"]
            speed = request.get("speed", 1.0)
            output_file = request.get("output_file")
            
            logger.info(f"Processing synthesis request: '{text[:30]}...' for voice {self.voice}")
            
            # Always stop any current playback when a new request is received
            # This ensures proper interruption behavior
            logger.info(f"Stopping any current audio for {self.voice}")
            self.audio_manager.stop_current()
            
            # Send request to model server
            model_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn_start = time.time()
            model_sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
            logger.info(f"Connected to model server in {time.time() - conn_start:.4f} seconds")
            
            # Prepare model server request
            model_request = {
                "text": text,
                "voice": self.voice,
                "speed": speed,
                "lang": self.lang
            }
            
            # Send request with special flag indicating it's from voice server
            model_request["from_voice_server"] = True
            logger.info(f"Sending request to model server: {len(json.dumps(model_request))} bytes")
            req_send_time = time.time()
            model_sock.sendall(json.dumps(model_request).encode())
            logger.info(f"Request sent in {time.time() - req_send_time:.4f} seconds")
            
            # Receive response with better error handling
            try:
                model_sock.settimeout(300.0)  # Longer timeout for synthesis (5 minutes)
                response_data = b""
                chunk_size = 8192
                
                recv_start = time.time()
                logger.info(f"Waiting for model server response")
                
                # Read in chunks until we get all data
                chunks_received = 0
                while True:
                    chunk = model_sock.recv(chunk_size)
                    if not chunk:
                        break
                    response_data += chunk
                    chunks_received += 1
                    
                    # Every 10 chunks, log progress
                    if chunks_received % 10 == 0:
                        logger.debug(f"Received {len(response_data)/1024:.2f} KB in {chunks_received} chunks")
                    
                    # Try parsing after each chunk to see if we have complete data
                    try:
                        json.loads(response_data.decode())
                        break  # Successfully parsed, we have complete data
                    except json.JSONDecodeError:
                        # Not complete yet, continue reading
                        pass
                
                logger.info(f"Response received in {time.time() - recv_start:.2f} seconds, size: {len(response_data)/1024:.2f} KB")
                model_sock.close()
                
                if not response_data:
                    raise RuntimeError("No response from model server")
            except socket.timeout:
                model_sock.close()
                raise RuntimeError("Timeout waiting for model server response")
                
            # Parse response
            parse_start = time.time()
            response = json.loads(response_data.decode())
            logger.info(f"Response parsed in {time.time() - parse_start:.4f} seconds")
            
            if response.get("status") == "error":
                error_msg = response.get("error", "Unknown error")
                logger.error(f"Model server error: {error_msg}")
                SocketProtocol.send_message(conn, {"status": "error", "error": error_msg})
                return
                
            # Convert audio data to numpy array
            convert_start = time.time()
            audio_data = np.array(response["audio"], dtype=np.float32)
            sample_rate = response.get("sample_rate", SAMPLE_RATE)
            logger.info(f"Audio data converted in {time.time() - convert_start:.4f} seconds")
            
            logger.info(f"Received audio: {len(audio_data)} samples, min={audio_data.min():.4f}, max={audio_data.max():.4f}")
            
            # Handle output file if specified
            if output_file:
                try:
                    import soundfile as sf
                    sf.write(output_file, audio_data, sample_rate)
                    logger.info(f"Saved audio to file: {output_file}")
                    
                    # Send success response to client
                    SocketProtocol.send_message(conn, {
                        "status": "success",
                        "message": f"Audio saved to {output_file}"
                    })
                except Exception as e:
                    logger.error(f"Failed to save audio file: {e}")
                    SocketProtocol.send_message(conn, {
                        "status": "error", 
                        "error": f"Failed to save audio file: {str(e)}"
                    })
            else:
                # Play audio and log the exact moment playback begins
                logger.info(f"Starting audio playback for voice {self.voice}: {text[:30]}...")
                playback_start = time.time()
                
                # Play the audio using our improved audio manager
                self.audio_manager.play(audio_data, sample_rate)
                
                logger.info(f"Playback initiated for voice {self.voice} in {time.time() - playback_start:.4f} seconds")
                
                # Send success response to client (client has already exited by now)
                try:
                    SocketProtocol.send_message(conn, {"status": "success"})
                except:
                    # Client likely already disconnected, which is expected
                    logger.debug("Client already disconnected when sending playback success")
            
            total_time = time.time() - start_time
            logger.info(f"Total synthesis processing completed in {total_time:.2f} seconds")
            
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