#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import signal
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.constants import (
    MODEL_PATH,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICES_CONFIG_PATH,
    SOCKET_BASE_PATH,
    SAMPLE_RATE
)

from src.socket_protocol import SocketProtocol

logger = logging.getLogger(__name__)

class ModelServer:
    """Central server that maintains single KModel instance and handles synthesis requests.
    
    This server acts as a coordinator for voice-specific servers, routing requests to the
    appropriate voice server and managing the shared model resources.
    """
    
    def __init__(self):
        self.running = True
        self.model = None
        self.pipeline = None
        self.model_lock = threading.Lock()
        self.request_queue = queue.Queue()
        self.synthesis_thread = None
        
        # Dictionary to track running voice servers
        self.voice_servers = {}
        self.voice_servers_lock = threading.Lock()
        
        # Thread pool for handling synthesis requests
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # TCP socket for voice server communication
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Socket directory
        os.makedirs(os.path.dirname(SOCKET_BASE_PATH), exist_ok=True)
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        
    def _initialize(self):
        """Custom initialization."""
        # Ensure socket directory exists
        socket_dir = os.path.dirname(SOCKET_BASE_PATH)
        os.makedirs(socket_dir, exist_ok=True)
        
        # Clean up any old socket files
        self._clean_old_sockets()
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
            
    def _clean_old_sockets(self):
        """Clean up any old socket files that might be left from previous runs."""
        try:
            import glob
            socket_pattern = f"{SOCKET_BASE_PATH}_*"
            old_sockets = glob.glob(socket_pattern)
            
            for socket_file in old_sockets:
                try:
                    os.unlink(socket_file)
                    logger.info(f"Removed old socket file: {socket_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove old socket file {socket_file}: {e}")
        except Exception as e:
            logger.warning(f"Error cleaning old sockets: {e}")
            
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
            
    def get_voice_server_socket(self, voice: str, lang: str) -> str:
        """Get the socket path for a voice server."""
        return f"{SOCKET_BASE_PATH}_{voice}_{lang}"
        
    def is_voice_server_running(self, voice: str, lang: str) -> bool:
        """Check if voice server is running by attempting a connection."""
        socket_path = self.get_voice_server_socket(voice, lang)
        
        try:
            # Check if the socket file exists
            if not os.path.exists(socket_path):
                return False
                
            # Try to connect to the socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(socket_path)
            sock.close()
            
            # If we can connect, the server is running
            return True
        except:
            # If any error occurs, the server is not running properly
            return False
            
    def start_voice_server(self, voice: str, lang: str) -> bool:
        """Start a new voice server process for the given voice and language."""
        
        # Import here to avoid circular imports
        import subprocess
        import sys
        
        try:
            logger.info(f"Starting voice server for {voice}/{lang}")
            
            # Create a script to start the voice server
            command = [
                sys.executable,
                "-c",
                f"""
import sys
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level='DEBUG',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/voice_server_{voice}_{lang}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("voice_server")

# Add project directory to Python path
project_dir = "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
sys.path.insert(0, project_dir)

try:
    logger.info("Starting voice server")
    from src.voice_server import VoiceServer
    
    server = VoiceServer("{voice}", "{lang}")
    server.start()
except Exception as e:
    logger.error(f"Voice server startup failed: {{e}}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
                """
            ]
            
            # Start the voice server in a new process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach the process
            )
            
            # Wait for the server to be available
            max_attempts = 10
            for attempt in range(max_attempts):
                if self.is_voice_server_running(voice, lang):
                    logger.info(f"Voice server for {voice}/{lang} started successfully")
                    
                    # Store the process reference
                    with self.voice_servers_lock:
                        self.voice_servers[(voice, lang)] = process
                        
                    return True
                    
                # Check if process is still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"Voice server process exited with code {process.returncode}")
                    logger.error(f"STDOUT: {stdout.decode()}")
                    logger.error(f"STDERR: {stderr.decode()}")
                    return False
                    
                # Wait before checking again
                time.sleep(0.5)
                
            # Timeout waiting for server to start
            logger.error(f"Timeout waiting for voice server {voice}/{lang} to start")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start voice server: {e}")
            return False
            
    def forward_to_voice_server(self, request: Dict, client_conn: socket.socket):
        """Forward synthesis request to the appropriate voice server."""
        voice = request.get("voice", "af_bella")
        lang = request.get("lang", "en-us")
        voice_sock = None
        
        try:
            # Ensure voice server is running
            if not self.is_voice_server_running(voice, lang):
                logger.info(f"Voice server for {voice}/{lang} not running, starting it")
                if not self.start_voice_server(voice, lang):
                    raise RuntimeError(f"Failed to start voice server for {voice}/{lang}")
                    
            # Get the socket path
            socket_path = self.get_voice_server_socket(voice, lang)
            
            # Connect to voice server
            voice_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            voice_sock.connect(socket_path)
            
            # Forward the request
            request_data = json.dumps(request).encode()
            SocketProtocol.send_message(voice_sock, request_data)
            
            # Return immediate acknowledgment to client
            # This lets the client exit right away
            success_response = {"status": "success", "message": "Request received, processing..."}
            client_conn.sendall(json.dumps(success_response).encode())
            
            # No need to wait for voice server response, as client has already gotten acknowledgment
            # The voice server will handle playback independently
        
        except Exception as e:
            logger.error(f"Error forwarding to voice server: {e}")
            try:
                error_response = {"status": "error", "error": str(e)}
                client_conn.sendall(json.dumps(error_response).encode())
            except:
                pass
        
        finally:
            # Close connections
            if voice_sock:
                try:
                    voice_sock.close()
                except:
                    pass
            
            try:
                client_conn.close()
            except:
                pass
                
    def kill_all_voice_servers(self):
        """Kill all running voice servers."""
        with self.voice_servers_lock:
            logger.info(f"Killing all voice servers: {len(self.voice_servers)} running")
            
            # First try graceful shutdown
            for (voice, lang), process in list(self.voice_servers.items()):
                try:
                    socket_path = self.get_voice_server_socket(voice, lang)
                    if os.path.exists(socket_path):
                        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        sock.connect(socket_path)
                        request = {"command": "exit"}
                        sock.sendall(json.dumps(request).encode())
                        sock.close()
                except:
                    pass
                    
            # Wait briefly for servers to shut down
            time.sleep(0.5)
            
            # Force kill any remaining processes
            for (voice, lang), process in list(self.voice_servers.items()):
                try:
                    if process.poll() is None:  # Process still running
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except:
                    pass
                    
            # Clear the dictionary
            self.voice_servers.clear()
                
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
        """Handle client connection and process requests by routing to voice servers."""
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
                
                # Kill all voice servers
                self.kill_all_voice_servers()
                
                response = {"status": "shutdown_initiated"}
                conn.sendall(json.dumps(response).encode())
                return
                
            # Handle synthesis request
            if "text" in request:
                # Get voice and language from request
                voice = request.get("voice", "af_bella")
                lang = request.get("lang", "en-us")
                
                # Check if this is a direct request from a voice server
                if request.get("from_voice_server", False):
                    logger.info(f"Received direct synthesis request from voice server for {voice}")
                    # Handle synthesis directly as it's from voice server
                    self.handle_synthesis_request(request, conn)
                else:
                    # This is from the client - route to voice server
                    logger.info(f"Routing client request for voice {voice} and language {lang}")
                    self.forward_to_voice_server(request, conn)
                    
        except Exception as e:
            logger.error(f"Error handling client request: {e}")
            try:
                error_response = {"status": "error", "error": str(e)}
                conn.sendall(json.dumps(error_response).encode())
            except:
                pass
            finally:
                try:
                    conn.close()
                except:
                    pass
    
    def handle_synthesis_request(self, request, conn):
        """Handle synthesis request directly"""
        try:
            start_time = time.time()
            logger.info(f"Starting synthesis at {start_time}")
            
            # Get synthesis parameters
            voice = request.get("voice", "af_bella")
            speed = request.get("speed", 1.0)
            text = request.get("text", "")
            
            logger.info(f"Model server processing text with voice {voice} at speed {speed}")
            
            # Use pipeline to generate audio
            audio_chunks = []
            
            # Process text through pipeline
            try:
                # Process through pipeline
                synthesis_start = time.time()
                for graphemes, phonemes, audio in self.pipeline(
                    text=text,
                    voice=voice,
                    speed=speed
                ):
                    logger.debug(f"Generated audio chunk: {len(audio)} samples")
                    audio_chunks.append(audio)
                
                synthesis_time = time.time() - synthesis_start
                logger.info(f"Pipeline processing completed in {synthesis_time:.2f} seconds")
                    
                if not audio_chunks:
                    raise ValueError("No audio chunks generated")
                    
                # Concatenate all chunks
                concat_start = time.time()
                final_audio = np.concatenate(audio_chunks)
                logger.info(f"Audio concatenation completed in {time.time() - concat_start:.2f} seconds")
                
                # Send audio data back in correct format for voice server
                logger.info(f"Sending audio data: {len(final_audio)} samples ({len(final_audio)*4/1024:.2f} KB)")
                
                # Voice server expects an "audio" key with the audio data
                response = {
                    "status": "success", 
                    "audio": final_audio.tolist(),
                    "sample_rate": SAMPLE_RATE
                }
                
                # Measure serialization time
                serialize_start = time.time()
                response_json = json.dumps(response)
                logger.info(f"JSON serialization completed in {time.time() - serialize_start:.2f} seconds, size: {len(response_json)/1024:.2f} KB")
                
                # Measure sending time
                send_start = time.time()
                conn.sendall(response_json.encode())
                logger.info(f"Audio data sent in {time.time() - send_start:.2f} seconds")
                
                total_time = time.time() - start_time
                logger.info(f"Total synthesis request handling completed in {total_time:.2f} seconds")
                    
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
               
    def synthesis_worker(self):
        """Worker thread that processes synthesis requests from voice servers."""
        while self.running:
            try:
                # Get next request from queue with timeout
                try:
                    request, conn = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Process synthesis request
                try:
                    # Get synthesis parameters
                    voice = request.get("voice", "af_bella")
                    speed = request.get("speed", 1.0)
                    text = request.get("text", "")
                    
                    # We only need model lock for model access, not for the entire synthesis
                    with self.model_lock:
                        if self.pipeline is None:
                            error_response = {
                                "status": "error",
                                "error": "Pipeline not initialized"
                            }
                            conn.sendall(json.dumps(error_response).encode())
                            continue                    
                    logger.info(f"Model server processing text with voice {voice} at speed {speed}")
                    
                    # Use pipeline to generate audio
                    audio_chunks = []
                    
                    # Process text through pipeline
                    try:
                        # NOTE: We don't need the model lock for this part - the pipeline
                        # and model are thread-safe for inference
                        for graphemes, phonemes, audio in self.pipeline(
                            text=text,
                            voice=voice,
                            speed=speed
                        ):
                            logger.debug(f"Generated audio chunk: {len(audio)} samples")
                            audio_chunks.append(audio)
                            
                        if not audio_chunks:
                            raise ValueError("No audio chunks generated")
                            
                        # Concatenate all chunks
                        final_audio = np.concatenate(audio_chunks)
                        logger.info(f"final audio synthesis complete")
                        # Send audio data back to voice server
                        response = {
                            "status": "success",
                            "audio": final_audio.tolist(),
                            "sample_rate": SAMPLE_RATE
                        }
                        
                        logger.info(f"Sending audio data: {len(final_audio)} samples")
                        conn.sendall(json.dumps(response).encode())
                        logger.info("audio sent")
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
        
        # Kill all voice servers
        self.kill_all_voice_servers()
        
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
