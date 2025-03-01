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

# Import constants first, before using them
from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    VOICES_CONFIG_PATH,
    SOCKET_BASE_PATH,
    SAMPLE_RATE,
    LOG_FILE,
    LOG_FORMAT
)

from src.socket_protocol import SocketProtocol
from src.fetchers import ensure_voice

# Now configure logging after importing constants
def configure_model_server_logging():
    """Configure logging specifically for the model server process."""
    # Clear any existing handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up the file handler to use the main log file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Add a stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Configure the root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stream_handler)
    
    # Get a logger for the model server specifically
    logger = logging.getLogger(__name__)
    logger.info("==== MODEL SERVER LOGGING INITIALIZED ====")
    logger.info(f"Logging to {LOG_FILE}")
    
    return logger

# Replace the existing logger with our configured one
logger = configure_model_server_logging()


class ModelServer:
    """Central server that maintains single KModel instance and handles synthesis requests."""
    
    def __init__(self):
        pid = os.getpid()
        ppid = os.getppid()
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
        self.last_synthesis_time = 0
        self.synthesis_time_lock = threading.Lock()
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
        first_time_voice = request.get("first_time_voice", False)
        voice_sock = None
        
        try:
            # Log extended info if this is a first-time voice
            if first_time_voice:
                logger.info(f"First-time use of voice {voice}, may need extended initialization time")
            
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
            
            # Forward the request, including first_time_voice flag
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
        """Kill all voice servers and ensure they are terminated."""
        import glob
        import signal
        import os
        
        with self.voice_servers_lock:
            logger.info(f"Killing all voice servers: {len(self.voice_servers)} running")
            
            # First try graceful shutdown through sockets
            for (voice, lang), process in list(self.voice_servers.items()):
                try:
                    socket_path = self.get_voice_server_socket(voice, lang)
                    if os.path.exists(socket_path):
                        logger.info(f"Sending exit command to {voice}/{lang} server")
                        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        sock.settimeout(2.0)  # Short timeout to avoid hanging
                        sock.connect(socket_path)
                        request = {"command": "exit"}
                        SocketProtocol.send_json(sock, request)
                        
                        # Try to get acknowledgment
                        try:
                            response = SocketProtocol.receive_json(sock, timeout=1.0)
                            logger.info(f"Voice server {voice}/{lang} shutdown response: {response}")
                        except Exception as e:
                            logger.warning(f"No response from voice server {voice}/{lang}: {e}")
                        
                        sock.close()
                except Exception as e:
                    logger.warning(f"Error sending exit command to {voice}/{lang}: {e}")
                    
            # Wait briefly for servers to shut down
            logger.info("Waiting for voice servers to shut down...")
            time.sleep(1.0)
            
            # Force kill any remaining processes
            for (voice, lang), process in list(self.voice_servers.items()):
                try:
                    if process.poll() is None:  # Process still running
                        logger.info(f"Force killing voice server process for {voice}/{lang}")
                        
                        # Try SIGTERM first
                        try:
                            process_group_id = os.getpgid(process.pid)
                            os.killpg(process_group_id, signal.SIGTERM)
                            
                            # Give it a moment to terminate
                            for _ in range(5):
                                if process.poll() is not None:
                                    logger.info(f"Voice server {voice}/{lang} terminated with SIGTERM")
                                    break
                                time.sleep(0.1)
                        except Exception as e:
                            logger.warning(f"Error sending SIGTERM to {voice}/{lang}: {e}")
                        
                        # If still running, use SIGKILL
                        if process.poll() is None:
                            try:
                                process_group_id = os.getpgid(process.pid)
                                os.killpg(process_group_id, signal.SIGKILL)
                                logger.info(f"Voice server {voice}/{lang} terminated with SIGKILL")
                            except Exception as e:
                                logger.error(f"Error sending SIGKILL to {voice}/{lang}: {e}")
                                
                                # Last resort: direct kill
                                try:
                                    process.kill()
                                    logger.info(f"Voice server {voice}/{lang} terminated with process.kill()")
                                except Exception as e2:
                                    logger.error(f"Failed to kill {voice}/{lang} process: {e2}")
                except Exception as e:
                    logger.error(f"Error killing process for {voice}/{lang}: {e}")
                    
            # Look for any orphaned voice server processes
            try:
                # This will find any Python processes with 'voice_server' in the command line
                import subprocess
                output = subprocess.check_output(
                    ["pgrep", "-f", "python.*voice_server"],
                    universal_newlines=True
                ).strip()
                
                orphan_pids = output.split('\n') if output else []
                for pid in orphan_pids:
                    if pid.strip():
                        try:
                            pid = int(pid)
                            logger.warning(f"Found orphaned voice server process: {pid}")
                            
                            # Try to get process group and kill it
                            try:
                                process_group_id = os.getpgid(pid)
                                os.killpg(process_group_id, signal.SIGKILL)
                                logger.info(f"Killed orphaned process group {process_group_id}")
                            except:
                                # Direct kill if process group kill fails
                                os.kill(pid, signal.SIGKILL)
                                logger.info(f"Killed orphaned process {pid}")
                        except Exception as e:
                            logger.error(f"Error killing orphaned process {pid}: {e}")
            except subprocess.CalledProcessError:
                # No processes found
                pass
            except Exception as e:
                logger.error(f"Error finding orphaned processes: {e}")
                        
            # Clean up socket files
            try:
                for pattern in [f"{SOCKET_BASE_PATH}_*"]:
                    for socket_file in glob.glob(pattern):
                        try:
                            os.unlink(socket_file)
                            logger.info(f"Removed socket file: {socket_file}")
                        except Exception as e:
                            logger.warning(f"Failed to remove socket file {socket_file}: {e}")
            except Exception as e:
                logger.error(f"Error cleaning socket files: {e}")
                        
            # Clear the dictionary
            self.voice_servers.clear()
            logger.info("Voice server dictionary cleared")
            
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
        
    def forward_to_voice_server_and_wait(self, request: Dict, client_conn: socket.socket) -> bool:
        """Forward synthesis request to the voice server and wait for file generation to complete."""
        voice = request.get("voice", "af_bella")
        lang = request.get("lang", "en-us")
        output_file = request.get("output_file")
        voice_sock = None
        
        if not output_file:
            logger.error("Called forward_to_voice_server_and_wait without output_file")
            return False
        
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
            
            # Wait for completion response from voice server
            voice_response = SocketProtocol.receive_json(voice_sock, timeout=30.0)
            logger.info(f"Voice server response: {voice_response}")
            
            # Check if file was created
            max_attempts = 20
            for attempt in range(max_attempts):
                if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
                    logger.info(f"File {output_file} was successfully created")
                    # Forward success to client
                    success_response = {
                        "status": "success", 
                        "message": f"File created: {output_file}",
                        "file_size": os.path.getsize(output_file)
                    }
                    SocketProtocol.send_json(client_conn, success_response)
                    return True
                
                logger.info(f"Waiting for file ({attempt+1}/{max_attempts})...")
                time.sleep(0.5)
                
            # If we get here, file wasn't created
            logger.error(f"Timeout waiting for file {output_file} to be created")
            return False
            
        except Exception as e:
            logger.error(f"Error in forward_to_voice_server_and_wait: {e}")
            return False
        
        finally:
            # Close voice server connection
            if voice_sock:
                try:
                    voice_sock.close()
                except:
                    pass
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
                output_file = request.get("output_file")
                wait_for_completion = request.get("wait_for_completion", False)
                
                # Check if this is a direct request from a voice server
                if request.get("from_voice_server", False):
                    logger.info(f"Received direct synthesis request from voice server for {voice}")
                    # Queue the request for processing by a worker thread
                    self.request_queue.put((request, conn))
                else:
                    # This is from the client - route to voice server
                    logger.info(f"Routing client request for voice {voice} and language {lang}")
                    
                    if output_file and wait_for_completion:
                        # For file output with wait_for_completion flag,
                        # we need to wait until the file is actually generated
                        success = self.forward_to_voice_server_and_wait(request, conn)
                        if not success:
                            logger.error("Forward to voice server with wait failed")
                            error_response = {"status": "error", "error": "File generation failed"}
                            try:
                                SocketProtocol.send_json(conn, error_response)
                            except:
                                pass
                    else:
                        # Standard forwarding for non-file or non-waiting requests
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
    
    # Additional debugging and fixes for the audio primer issue

    def should_add_audio_primer(self):
        """
        Determine if we need to add a primer character to the speech.
        Returns True if more than 5 seconds have passed since last synthesis.
        """
        try:
            logger.info("should_add_audio_primer() called")
            with self.synthesis_time_lock:
                current_time = time.time()
                
                # Fix 3: Always log this information with full detail
                time_since_last_synthesis = current_time - self.last_synthesis_time
                
                # Fix 4: More reliable primer condition - add primer after inactivity or on first run
                needs_primer = self.last_synthesis_time == 0 or time_since_last_synthesis > 5
                
                logger.info(f"Audio primer check: time_since_last={time_since_last_synthesis:.2f}s, last_time={self.last_synthesis_time}, needs_primer={needs_primer}")
                
                return needs_primer
        except Exception as e:
            logger.error(f"Error in should_add_audio_primer: {e}")
            # Default to adding a primer if there's an error
            return True

    def update_synthesis_time(self):
        """Update the timestamp of the last synthesis operation."""
        try:
            with self.synthesis_time_lock:
                old_time = self.last_synthesis_time
                self.last_synthesis_time = time.time()
                logger.info(f"update_synthesis_time: old={old_time:.2f}, new={self.last_synthesis_time:.2f}")
        except Exception as e:
            logger.error(f"Error in update_synthesis_time: {e}")

    def synthesis_worker(self):
        """Worker thread that processes synthesis requests from voice servers."""
        thread_id = threading.get_ident()
        logger.info(f"Synthesis worker {thread_id} started")
        
        while self.running:
            try:
                # Get next request from queue with timeout
                try:
                    request, conn = self.request_queue.get(timeout=1.0)
                    logger.info(f"Worker {thread_id} received synthesis request")
                except queue.Empty:
                    continue
                    
                # Process synthesis request
                try:
                    # Get synthesis parameters
                    voice = request.get("voice", "af_bella")
                    speed = request.get("speed", 1.0)
                    text = request.get("text", "")
                    
                    logger.info(f"Worker {thread_id} processing request, original text: '{text[:30]}...'")
                    
                    # Fix 1: Change the primer logic to be more reliable
                    # Always add a very short primer sound and log it
                    needs_primer = self.should_add_audio_primer()
                    original_text = text
                    if needs_primer:
                        text = "  " + text
                        logger.info(f"Added audio primer to text: '{original_text[:30]}...' -> '{text[:30]}...'")
                    else:
                        logger.info(f"No primer needed for text: '{text[:30]}...'")
                    
                    # Fix 2: Ensure we update the synthesis timestamp AFTER confirming we need a primer
                    self.update_synthesis_time()
                    
                    # Ensure voice file exists
                    ensure_voice(voice)
                    
                    logger.info(f"Worker {thread_id} processing text with voice {voice} at speed {speed}")
                    
                    # Chunk text for more efficient processing
                    text_chunks = self._split_text(text)
                    logger.info(f"Split text into {len(text_chunks)} chunks")
                    
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
                        logger.info(f"Processing chunk {chunk_index+1}/{len(text_chunks)}: '{text_chunk[:30]}...'")
                        
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