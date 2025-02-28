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
import re
from pathlib import Path
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICES_CONFIG_PATH,
    SOCKET_BASE_PATH,
    SAMPLE_RATE
)

from src.socket_protocol import SocketProtocol
from src.fetchers import ensure_voice

logger = logging.getLogger(__name__)

class ModelServer:
    """Central server that maintains single KModel instance and handles synthesis requests."""
    
    def __init__(self):
        self.running = True
        self.model = None
        self.pipeline = None
        self.model_lock = threading.Lock()
        self.request_queue = queue.Queue()
        self.worker_threads = []
        self.num_workers = 4  # Number of worker threads
        
        # Dictionary to track running voice servers
        self.voice_servers = {}
        self.voice_servers_lock = threading.Lock()
        
        # Thread pool for handling client requests
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

        try:
            # Assume model has been installed properly and is available
            logger.info("Initializing KModel...")
            
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
            SocketProtocol.send_json(voice_sock, request)
            
            # Return immediate acknowledgment to client
            # This lets the client exit right away
            success_response = {"status": "success", "message": "Request received, processing..."}
            SocketProtocol.send_json(client_conn, success_response)
            
        except Exception as e:
            logger.error(f"Error forwarding to voice server: {e}")
            try:
                error_response = {"status": "error", "error": str(e)}
                SocketProtocol.send_json(client_conn, error_response)
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
                        SocketProtocol.send_json(sock, request)
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
    
    def start_worker_threads(self):
        """Start worker threads for synthesis processing."""
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self.synthesis_worker)
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        logger.info(f"Started {self.num_workers} worker threads")
            
    def start(self):
        """Start the model server."""
        try:
            logger.info("Starting model server")
            # Initialize model if not already done
            if self.model is None:
                self.initialize_model()
                if self.model is None:
                    raise RuntimeError("Model initialization failed")
            
            # Start worker threads
            self.start_worker_threads()
            
            # Bind TCP socket
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
            # Receive data using socket protocol
            try:
                request = SocketProtocol.receive_json(conn)
                logger.debug(f"Received request: {request}")
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                return
            
            # Process request
            if "command" in request and request["command"] == "exit":
                logger.info("Received exit command")
                self.running = False
                
                # Kill all voice servers
                self.kill_all_voice_servers()
                
                response = {"status": "shutdown_initiated"}
                SocketProtocol.send_json(conn, response)
                return
                
            # Handle synthesis request
            if "text" in request:
                # Get voice and language from request
                voice = request.get("voice", "af_bella")
                lang = request.get("lang", "en-us")
                
                # Check if this is a direct request from a voice server
                if request.get("from_voice_server", False):
                    logger.info(f"Received direct synthesis request from voice server for {voice}")
                    # Queue the request for processing by a worker thread
                    self.request_queue.put((request, conn))
                else:
                    # This is from the client - route to voice server
                    logger.info(f"Routing client request for voice {voice} and language {lang}")
                    self.forward_to_voice_server(request, conn)
                    
        except Exception as e:
            logger.error(f"Error handling client request: {e}")
            try:
                error_response = {"status": "error", "error": str(e)}
                SocketProtocol.send_json(conn, error_response)
            except:
                pass
            finally:
                try:
                    conn.close()
                except:
                    pass
    
    def _split_text(self, text):
        """Split text into reasonable chunks for processing."""
        if not text:
            return []
            
        # Try to split on sentence boundaries
        # Split on sentence endings followed by spaces or newlines
        chunks = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty chunks
        chunks = [c for c in chunks if c.strip()]
        
        # If we have very large sentences, split them further
        MAX_CHUNK_LENGTH = 500  # Characters
        result = []
        
        for chunk in chunks:
            if len(chunk) <= MAX_CHUNK_LENGTH:
                result.append(chunk)
            else:
                # Split long chunk further at commas, semicolons, etc.
                subchunks = re.split(r'(?<=[,;:])\s+', chunk)
                
                # If still too large, just use fixed-size chunks
                if any(len(sc) > MAX_CHUNK_LENGTH for sc in subchunks):
                    current = ""
                    for sc in subchunks:
                        if len(current) + len(sc) <= MAX_CHUNK_LENGTH:
                            current += sc + " "
                        else:
                            if current:
                                result.append(current.strip())
                            current = sc + " "
                    if current:
                        result.append(current.strip())
                else:
                    result.extend(subchunks)
        
        return result
    
    def synthesis_worker(self):
        """Worker thread that processes synthesis requests from voice servers."""
        thread_id = threading.get_ident()
        logger.info(f"Synthesis worker {thread_id} started")
        
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
                    
                    # Ensure voice file exists
                    ensure_voice(voice)
                    
                    logger.info(f"Worker {thread_id} processing text with voice {voice} at speed {speed}")
                    
                    # Chunk text for more efficient processing
                    text_chunks = self._split_text(text)
                    
                    # Send streaming response start
                    header = {
                        "status": "streaming",
                        "sample_rate": SAMPLE_RATE,
                        "chunk_count": len(text_chunks)
                    }
                    SocketProtocol.send_json(conn, header)
                    
                    # Process each text chunk
                    chunk_index = 0
                    for text_chunk in text_chunks:
                        start_time = time.time()
                        logger.info(f"Processing chunk {chunk_index+1}/{len(text_chunks)}: {text_chunk[:30]}...")
                        
                        try:
                            # Process chunk through pipeline
                            audio_segments = []
                            
                            for graphemes, phonemes, audio in self.pipeline(
                                text=text_chunk,
                                voice=voice,
                                speed=speed
                            ):
                                audio_segments.append(audio)
                            
                            # Concatenate audio segments for this chunk
                            if audio_segments:
                                chunk_audio = np.concatenate(audio_segments)
                                
                                # Stream the audio chunk
                                logger.info(f"Streaming audio chunk {chunk_index+1}: {len(chunk_audio)} samples")
                                is_final = (chunk_index == len(text_chunks) - 1)
                                SocketProtocol.send_audio_chunk(conn, chunk_audio, is_final)
                                
                            chunk_index += 1
                            logger.info(f"Chunk {chunk_index}/{len(text_chunks)} completed in {time.time() - start_time:.2f} seconds")
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {chunk_index}: {e}")
                            # Send error message but continue with next chunks
                            error_msg = {
                                "status": "chunk_error",
                                "chunk_index": chunk_index,
                                "error": str(e)
                            }
                            SocketProtocol.send_json(conn, error_msg)
                            chunk_index += 1
                    
                    logger.info(f"Completed processing all {len(text_chunks)} chunks for request")
                    
                except Exception as e:
                    logger.error(f"Synthesis error: {e}")
                    error_response = {
                        "status": "error",
                        "error": str(e)
                    }
                    try:
                        SocketProtocol.send_json(conn, error_response)
                    except:
                        pass
                
                finally:
                    self.request_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
                                
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
        
        # Stop worker threads
        logger.info("Waiting for worker threads to finish...")
        self.request_queue.join()  # Wait for all tasks to complete
        
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
        
        logger.info("Model server shutdown complete")