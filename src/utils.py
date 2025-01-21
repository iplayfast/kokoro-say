#!/usr/bin/env python3

"""
Utility functions for the TTS application.
"""

#!/usr/bin/env python3

import os
import sys
import signal
import psutil
import logging
import socket
import time
import json
import stat
import threading
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from .constants import (
    LANGUAGES,
    SOCKET_BASE_PATH,
    MODEL_SERVER_SOCKET,
    MODEL_PID_FILE,
    MAX_RETRIES,
    RETRY_DELAY,
    MAX_CHUNK_SIZE,
    SENTENCE_END_MARKERS
)
from .fetchers import ensure_model_and_voices

logger = logging.getLogger(__name__)

def wait_for_socket(socket_path: str, timeout: int = 30) -> bool:
    """Wait for Unix domain socket to become available and writable"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(socket_path):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(socket_path)
                sock.close()
                return True
            except (socket.error, OSError):
                pass
        time.sleep(0.1)
    return False

def test_model_server(timeout: int = 5) -> bool:
    """Test if model server is responsive"""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(MODEL_SERVER_SOCKET)
            
            request = {"type": "ping"}
            request_data = json.dumps(request).encode('utf-8')
            length_prefix = f"{len(request_data):08d}\n".encode('utf-8')
            
            # Send request
            sock.sendall(length_prefix + request_data)
            
            # Read response length
            length_data = sock.recv(9)
            if not length_data:
                return False
                
            try:
                expected_length = int(length_data.decode().strip())
                response_data = sock.recv(expected_length).decode()
                response = json.loads(response_data)
                return response.get("status") == "ok"
            except (ValueError, json.JSONDecodeError):
                return False
                
    except Exception as e:
        logger.debug(f"Model server test failed: {e}")
        return False

def ensure_model_server_running() -> bool:
    """Ensure the model server is running and ready"""
    logger.debug("Checking model server status")
    
    # Check if process exists
    if os.path.exists(MODEL_PID_FILE):
        try:
            with open(MODEL_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process is running
            os.kill(pid, 0)
            
            # Check if socket exists and is valid
            if os.path.exists(MODEL_SERVER_SOCKET):
                if stat.S_ISSOCK(os.stat(MODEL_SERVER_SOCKET).st_mode):
                    # Test if server is responsive
                    if test_model_server():
                        logger.debug("Existing model server is responsive")
                        return True
        except Exception as e:
            logger.debug(f"Existing model server check failed: {e}")
    
    logger.debug("Starting new model server")
    
    # Clean up any existing socket
    try:
        if os.path.exists(MODEL_SERVER_SOCKET):
            os.unlink(MODEL_SERVER_SOCKET)
    except Exception as e:
        logger.error(f"Failed to clean up existing socket: {e}")
        return False
    
    # Start new server process
    try:
        pid = os.fork()
        
        if pid == 0:  # Child process
            try:
                # Redirect standard file descriptors
                null_fd = os.open(os.devnull, os.O_RDWR)
                os.dup2(null_fd, 0)  # stdin
                os.dup2(null_fd, 1)  # stdout
                os.dup2(null_fd, 2)  # stderr
                os.close(null_fd)
                
                # Import here to avoid circular imports
                from .model_server import ModelServer
                model_server = ModelServer.get_instance()
                model_server.start()
            except Exception as e:
                logger.error(f"Failed to start model server: {e}")
                sys._exit(1)
        else:  # Parent process
            logger.debug(f"Started model server process with PID {pid}")
            
            # Wait for socket to appear and become writable
            if not wait_for_socket(MODEL_SERVER_SOCKET):
                logger.error("Timeout waiting for model server socket")
                return False
            
            # Test if server is responsive
            retries = 10
            for i in range(retries):
                if test_model_server():
                    logger.debug("Model server is now responsive")
                    return True
                time.sleep(0.5)
            
            logger.error("Model server failed to become responsive")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        return False
    
    return False

def kill_all_daemons() -> bool:
    """Kill all running TTS daemons and the model server"""
    killed = False
    
    # Stop model server first
    try:
        if os.path.exists(MODEL_PID_FILE):
            with open(MODEL_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            killed = True
            print(f"Killed model server with PID {pid}")
            # Wait for process to actually terminate
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
    except Exception as e:
        print(f"Error killing model server: {e}")
    
    # Kill voice servers
    for pid_file in Path('/tmp').glob('tts_daemon_*.pid'):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            killed = True
            print(f"Killed voice server with PID {pid}")
            # Wait for process to terminate
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
        except Exception as e:
            print(f"Error killing voice server: {e}")
        finally:
            pid_file.unlink(missing_ok=True)
    
    # Clean up socket files
    for socket_file in Path('/tmp').glob('tts_socket_*'):
        try:
            socket_file.unlink(missing_ok=True)
        except Exception as e:
            print(f"Error removing socket file: {e}")
            
    if os.path.exists(MODEL_SERVER_SOCKET):
        try:
            os.unlink(MODEL_SERVER_SOCKET)
        except Exception as e:
            print(f"Error removing model server socket: {e}")

    return killed


def get_voice_from_input(voice_input: str, voices: List[str]) -> Optional[str]:
    """
    Convert voice input (name or number) to voice name
    
    Args:
        voice_input: Voice name or number
        voices: List of available voices
        
    Returns:
        str | None: Voice name if valid input, None otherwise
    """
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(voices):
            return voices[index]
    elif voice_input in voices:
        return voice_input
    return None

def get_language_from_input(lang_input: str) -> Optional[str]:
    """
    Convert language input (code or number) to language code
    
    Args:
        lang_input: Language code or number
        
    Returns:
        str | None: Language code if valid input, None otherwise
    """
    if lang_input.isdigit():
        index = int(lang_input) - 1
        if 0 <= index < len(LANGUAGES):
            return LANGUAGES[index][0]
    else:
        for code, _ in LANGUAGES:
            if code == lang_input:
                return code
    return None