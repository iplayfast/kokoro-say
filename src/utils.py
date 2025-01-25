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
import multiprocessing
from pathlib import Path
from typing import Optional, Tuple, List

from .constants import (
    LANGUAGES,
    MODEL_SERVER_SOCKET,
    MODEL_PID_FILE,
)
from .fetchers import ModelFetcher, VoiceFetcher

logger = logging.getLogger(__name__)

def ensure_model_server_running() -> bool:
    """Ensure the model server is running and ready"""
    logger.debug("Starting ensure_model_server_running")
    
    # Start new server process
    try:
        pid = os.fork()
        
        if pid == 0:  # Child process
            try:
                logger.debug("In child process, setting up model server")
                # Import here to avoid circular imports
                from .model_server import ModelServer
                
                # Configure logging for child process
                child_logger = logging.getLogger('model_server_child')
                child_logger.setLevel(logging.DEBUG)
                fh = logging.FileHandler('/tmp/model_server_child.log')
                fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                child_logger.addHandler(fh)
                
                child_logger.debug("Starting ModelServer instance")
                model_server = ModelServer.get_instance()
                child_logger.debug("Starting model server")
                model_server.start()
            except Exception as e:
                logger.error(f"Failed to start model server in child process: {e}", exc_info=True)
                sys._exit(1)
        else:  # Parent process
            logger.debug(f"Started model server process with PID {pid}")
            
            # Wait for socket to appear
            logger.debug("Waiting for socket to appear")
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                if os.path.exists(MODEL_SERVER_SOCKET):
                    logger.debug("Socket file exists")
                    # Check if process is still running
                    try:
                        os.kill(pid, 0)
                        logger.debug("Model server process is running")
                        return True
                    except OSError:
                        logger.error("Model server process died")
                        return False
                logger.debug(f"Socket not found, attempt {attempts + 1}/{max_attempts}")
                time.sleep(1)
                attempts += 1
            
            logger.error("Timeout waiting for model server socket")
            try:
                os.kill(pid, signal.SIGTERM)
                logger.debug("Sent SIGTERM to model server process")
            except OSError:
                pass
            return False
            
    except Exception as e:
        logger.error(f"Failed to start model server: {e}", exc_info=True)
        return False
    
    return False

def ensure_model_and_voices(script_dir: str | os.PathLike) -> Tuple[str, str]:
    """
    Ensure both model and voices exist before starting
    
    Args:
        script_dir: Directory where models should be stored
        
    Returns:
        tuple[str, str]: Paths to the model and voices files
    
    Raises:
        RuntimeError: If downloads fail
    """
    script_dir = os.path.expanduser(script_dir)
    model_path = os.path.join(script_dir, "kokoro-v0_19.onnx")
    voices_path = os.path.join(script_dir, "voices.json")
    
    model_fetcher = ModelFetcher()
    voice_fetcher = VoiceFetcher()
    
    try:
        model_fetcher.fetch_model(model_path)
        voice_fetcher.fetch_voices(voices_path)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise RuntimeError(f"Failed to ensure model and voices: {e}")
    
    return model_path, voices_path

def get_voice_from_input(voice_input: str, voices: List[str]) -> Optional[str]:
    """Convert voice input (name or number) to voice name"""
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(voices):
            return voices[index]
    elif voice_input in voices:
        return voice_input
    return None

def get_language_from_input(lang_input: str) -> Optional[str]:
    """Convert language input (code or number) to language code"""
    if lang_input.isdigit():
        index = int(lang_input) - 1
        if 0 <= index < len(LANGUAGES):
            return LANGUAGES[index][0]
    else:
        for code, _ in LANGUAGES:
            if code == lang_input:
                return code
    return None

def wait_for_socket(socket_path: str, timeout: int = 30) -> bool:
    """Wait for Unix domain socket to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(socket_path):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(socket_path)
                sock.close()
                return True
            except (socket.error, OSError) as e:
                logger.debug(f"Socket connection attempt failed: {e}")
        time.sleep(0.1)
    return False

def start_model_server() -> Optional[int]:
    """Start the model server process and return its PID"""
    try:
        # Create a pipe for error reporting
        read_fd, write_fd = os.pipe()
        
        pid = os.fork()
        if pid == 0:  # Child process
            try:
                os.close(read_fd)
                write_pipe = os.fdopen(write_fd, 'w')
                
                # Redirect standard file descriptors
                null_fd = os.open(os.devnull, os.O_RDWR)
                os.dup2(null_fd, 0)  # stdin
                os.dup2(null_fd, 1)  # stdout
                os.dup2(null_fd, 2)  # stderr
                os.close(null_fd)
                
                # Import here to avoid circular imports
                from .model_server import ModelServer
                server = ModelServer.get_instance()
                server.start()
                
                # If we get here, server started successfully
                write_pipe.write("OK")
                write_pipe.close()
                sys._exit(0)
                
            except Exception as e:
                # Report error back to parent
                try:
                    write_pipe.write(str(e))
                    write_pipe.close()
                except:
                    pass
                sys._exit(1)
                
        else:  # Parent process
            os.close(write_fd)
            read_pipe = os.fdopen(read_fd, 'r')
            
            # Wait briefly for child process to start
            start_time = time.time()
            while time.time() - start_time < 5:
                result = read_pipe.read()
                if result:
                    if result != "OK":
                        logger.error(f"Model server startup error: {result}")
                        return None
                    break
                time.sleep(0.1)
            
            logger.debug(f"Started model server process with PID {pid}")
            return pid
            
    except Exception as e:
        logger.error(f"Failed to start model server process: {e}")
        return None

def ensure_model_server_running1() -> bool:
    """Ensure the model server is running and ready"""
    logger.debug("Checking model server status")
    
    # Check if process exists and is responsive
    if os.path.exists(MODEL_PID_FILE):
        try:
            with open(MODEL_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process is running
            os.kill(pid, 0)
            
            # Test socket connection
            if os.path.exists(MODEL_SERVER_SOCKET):
                sock_stat = os.stat(MODEL_SERVER_SOCKET)
                if stat.S_ISSOCK(sock_stat.st_mode):
                    if wait_for_socket(MODEL_SERVER_SOCKET, timeout=5):
                        logger.debug("Existing model server is responsive")
                        return True
        except Exception as e:
            logger.debug(f"Existing model server check failed: {e}")
    
    # Clean up any existing socket
    try:
        if os.path.exists(MODEL_SERVER_SOCKET):
            os.unlink(MODEL_SERVER_SOCKET)
    except Exception as e:
        logger.error(f"Failed to clean up existing socket: {e}")
        return False
    
    # Start new server process
    pid = start_model_server()
    if not pid:
        logger.error("Failed to start model server process")
        return False
        
    # Wait for socket to appear and become writable
    if not wait_for_socket(MODEL_SERVER_SOCKET):
        logger.error("Timeout waiting for model server socket")
        return False
        
    logger.info("Model server started successfully")
    return True

def kill_all_daemons() -> bool:
    """Kill all running TTS daemons"""
    killed = False
    
    # First kill model server
    if os.path.exists(MODEL_PID_FILE):
        try:
            with open(MODEL_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            killed = True
            logger.info(f"Killed model server (PID {pid})")
            time.sleep(0.1)  # Give process time to clean up
        except Exception as e:
            logger.error(f"Error killing model server: {e}")
    
    # Kill any lingering voice servers
    for pid_file in Path('/tmp').glob('tts_daemon_*.pid'):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            killed = True
            logger.info(f"Killed voice server with PID {pid}")
            time.sleep(0.1)  # Give process time to clean up
        except Exception as e:
            logger.error(f"Error killing voice server: {e}")
        finally:
            try:
                pid_file.unlink(missing_ok=True)
            except:
                pass
    
    # Clean up socket files
    for socket_file in Path('/tmp').glob('tts_socket_*'):
        try:
            socket_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error removing socket file: {e}")
            
    if os.path.exists(MODEL_SERVER_SOCKET):
        try:
            os.unlink(MODEL_SERVER_SOCKET)
        except Exception as e:
            logger.error(f"Error removing model server socket: {e}")
    
    return killed
