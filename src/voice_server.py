#!/usr/bin/env python3
import os
import sys
import json
import socket
import logging
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
        
        # For handling ongoing synthesis
        self.current_synthesis_id = None
        self.synthesis_lock = threading.Lock()
        
        # Accumulated audio for saving to file if needed
        self.current_audio_buffer = []
        self.current_audio_lock = threading.Lock()
        
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
            
            # Receive message using enhanced SocketProtocol
            try:
                command, data = SocketProtocol.receive_message(conn)
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
                    SocketProtocol.send_json(conn, response)
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
        # Generate a unique ID for this synthesis request
        synthesis_id = f"{int(time.time())}-{os.getpid()}"
        
        with self.synthesis_lock:
            # Stop any current synthesis and playback
            if self.current_synthesis_id:
                logger.info(f"Interrupting previous synthesis {self.current_synthesis_id}")
                
            # Set as current synthesis
            self.current_synthesis_id = synthesis_id
            
            # Stop any current audio playback
            self.audio_manager.stop_current()
            
            # Clear audio buffer for new synthesis
            with self.current_audio_lock:
                self.current_audio_buffer.clear()
        
        try:
            thread_id = threading.get_ident()
            logger.info(f"Processing synthesis {synthesis_id} in thread {thread_id}")
            start_time = time.time()
            
            # Get synthesis parameters
            text = request["text"]
            speed = request.get("speed", 1.0)
            output_file = request.get("output_file")
            
            logger.info(f"Processing synthesis request: '{text[:30]}...' for voice {self.voice}")
            
            # Send streaming request to model server
            model_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn_start = time.time()
            model_sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
            logger.info(f"Connected to model server in {time.time() - conn_start:.4f} seconds")
            
            # Prepare model server request with streaming flag
            model_request = {
                "text": text,
                "voice": self.voice,
                "speed": speed,
                "lang": self.lang,
                "streaming": True,
                "from_voice_server": True
            }
            
            # Send request to model server
            SocketProtocol.send_json(model_sock, model_request)
            logger.info(f"Request sent to model server in {time.time() - conn_start:.4f} seconds")
            
            # Send immediate success response to client
            try:
                SocketProtocol.send_json(conn, {"status": "success"})
            except:
                # Client might have disconnected, which is expected
                pass
            
            # Process streaming response from model server
            model_sock.settimeout(300.0)  # 5 minute timeout for long texts
            
            # First message should be header with stream info
            command, data = SocketProtocol.receive_message(model_sock)
            if command != SocketProtocol.CMD_JSON:
                raise RuntimeError(f"Expected JSON stream header, got {command}")
                
            stream_info = json.loads(data.decode())
            logger.info(f"Received stream info: {stream_info}")
            
            if stream_info.get("status") == "error":
                raise RuntimeError(f"Model server error: {stream_info.get('error')}")
                
            # Flag indicating if we're saving to file
            is_saving = output_file is not None
            
            # Process audio chunks as they arrive
            is_synthesis_complete = False
            full_audio_data = []
            
            while not is_synthesis_complete:
                try:
                    # Check if we've been interrupted
                    with self.synthesis_lock:
                        if self.current_synthesis_id != synthesis_id:
                            logger.info(f"Synthesis {synthesis_id} was interrupted, stopping")
                            break
                    
                    # Receive next message
                    command, data = SocketProtocol.receive_message(model_sock)
                    
                    if command == SocketProtocol.CMD_JSON:
                        # JSON message - could be status update or error
                        message = json.loads(data.decode())
                        logger.info(f"Received JSON message: {message}")
                        
                        if message.get("status") == "error":
                            logger.error(f"Error from model server: {message.get('error')}")
                            break
                            
                        if message.get("status") == "chunk_error":
                            logger.warning(f"Chunk error: {message.get('error')}")
                            # Continue processing other chunks
                            continue
                            
                    elif command == SocketProtocol.CMD_AUDIO:
                        # Audio chunk
                        audio_data = np.frombuffer(data, dtype=np.float32)
                        full_audio_data.append(audio_data)
                        
                        # Store for saving if needed
                        if is_saving:
                            with self.current_audio_lock:
                                self.current_audio_buffer.append(audio_data)
                                
                    elif command == SocketProtocol.CMD_END:
                        # Final audio chunk
                        audio_data = np.frombuffer(data, dtype=np.float32)
                        full_audio_data.append(audio_data)
                        
                        # Store for saving
                        if is_saving:
                            with self.current_audio_lock:
                                self.current_audio_buffer.append(audio_data)
                        
                        # Mark synthesis as complete
                        is_synthesis_complete = True
                        logger.info(f"Synthesis {synthesis_id} complete")
                        
                    else:
                        logger.warning(f"Unknown command from model server: {command}")
                        
                except Exception as e:
                    logger.error(f"Error processing streaming response: {e}")
                    break
            
            # Close model server connection
            model_sock.close()
            
            # If we collected audio, play or save it
            if full_audio_data:
                # Concatenate audio chunks
                final_audio = np.concatenate(full_audio_data)
                
                # If output file is specified, save to file
                if is_saving:
                    try:
                        import soundfile as sf
                        sf.write(output_file, final_audio, SAMPLE_RATE)
                        logger.info(f"Saved audio to file: {output_file}")
                    except Exception as e:
                        logger.error(f"Failed to save audio file: {e}")
                else:
                    # Play audio using the simpler playback method
                    self.audio_manager.play(final_audio, SAMPLE_RATE)
            
            total_time = time.time() - start_time
            logger.info(f"Total synthesis processing completed in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing synthesis: {e}")
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
                # Check if we're still playing audio
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
        self.audio_manager.stop_current()
        
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
        
        logger.info(f"Voice server for {self.voice}/{self.lang} shut down")

