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
import atexit
import fcntl
from pathlib import Path
from typing import Optional, Any, Dict, Set

class UnixSocketServer:
    """Base class for Unix domain socket servers with timeout management"""
    
    def __init__(self, socket_path: str, pid_file: str, server_type: str = "voice"):
        """
        Initialize the base server
        
        Args:
            socket_path: Path to Unix domain socket
            pid_file: Path to PID file
            server_type: Either "model" or "voice"
        """
        self.socket_path = socket_path
        self.pid_file = pid_file
        self.lock_file = f"{pid_file}.lock"
        self.server_type = server_type
        self.last_activity = time.time()
        
        # Timeouts
        self.VOICE_TIMEOUT = 300  # 5 minutes for individual voices
        self.MODEL_TIMEOUT = 300  # 5 minutes for model server
        self.CHECK_INTERVAL = 30  # Check every 30 seconds
        
        # Server state
        self.lock = threading.Lock()
        self.running = True
        self.server: Optional[socket.socket] = None
        
        # Voice tracking (for model server)
        self.voice_servers: Dict[str, Dict[str, Any]] = {}
        self.last_any_voice_activity = time.time()
        
        # Configure logging
        self.setup_logging()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        
        # Clean up any leftover files
        self.cleanup()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_timeouts)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = time.time()
        if self.server_type == "model":
            self.last_any_voice_activity = time.time()

    def _monitor_timeouts(self):
        """Monitor for timeouts and handle shutdowns"""
        while self.running:
            time.sleep(self.CHECK_INTERVAL)
            
            if not self.running:  # Check if we were shut down
                break
                
            current_time = time.time()
            
            if self.server_type == "model":
                # Check overall model timeout
                if current_time - self.last_any_voice_activity > self.MODEL_TIMEOUT:
                    self.logger.info("Model server idle timeout reached")
                    self.shutdown_all()
                    break
                    
                # Check individual voice timeouts
                for voice_id in list(self.voice_servers.keys()):
                    if current_time - self.voice_servers[voice_id]['last_activity'] > self.VOICE_TIMEOUT:
                        self.logger.info(f"Voice {voice_id} idle timeout reached")
                        self.shutdown_voice(voice_id)
                        
            else:  # Voice server
                if current_time - self.last_activity > self.VOICE_TIMEOUT:
                    self.logger.info("Voice server idle timeout reached")
                    self.handle_signal(signal.SIGTERM, None)
                    break

    def setup_logging(self):
        """Configure logging to write to separate file descriptor"""
        logger_name = f"server_{os.path.basename(self.socket_path)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        log_path = f"/tmp/{logger_name}.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.propagate = False

    def send_response(self, conn: socket.socket, response: Dict):
        """Send response with length prefix"""
        with self.lock:
            try:
                response_data = json.dumps(response).encode('utf-8')
                length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
                conn.sendall(length_prefix + response_data)
            except Exception as e:
                self.logger.error(f"Error sending response: {e}")
                raise

    def receive_request(self, conn: socket.socket) -> Dict:
        """Receive length-prefixed request data"""
        try:
            length_data = conn.recv(9)
            if not length_data:
                raise RuntimeError("Empty request")
            
            try:
                expected_length = int(length_data.decode().strip())
            except ValueError as e:
                raise RuntimeError(f"Invalid length prefix: {e}")
            
            data = bytearray()
            remaining = expected_length
            
            while remaining > 0:
                chunk = conn.recv(min(8192, remaining))
                if not chunk:
                    raise RuntimeError("Connection closed while reading data")
                data.extend(chunk)
                remaining -= len(chunk)
            
            try:
                request = json.loads(data.decode())
                return request
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON: {e}")
                
        except Exception as e:
            self.logger.error(f"Error receiving request: {e}")
            raise

    def cleanup(self):
        """Clean up socket and PID files"""
        for file_path in [self.socket_path, self.pid_file, self.lock_file]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                self.logger.error(f"Error removing {file_path}: {e}")

    def handle_signal(self, signum: int, frame: Any):
        """Handle termination signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        if self.server:
            try:
                self.server.close()
            except Exception as e:
                self.logger.error(f"Error closing server socket: {e}")
        
        self.cleanup()
        sys.exit(0)

    def shutdown_voice(self, voice_id: str):
        """Shutdown a specific voice server (model server only)"""
        if self.server_type != "model":
            return
            
        if voice_id in self.voice_servers:
            try:
                # Send shutdown command to voice server
                info = self.voice_servers[voice_id]
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(info['socket_path'])
                    command = {"command": "shutdown"}
                    self.send_response(sock, command)
                    
                # Wait for confirmation or timeout
                time.sleep(1)
                
                # Clean up voice server entry
                del self.voice_servers[voice_id]
                self.logger.info(f"Voice server {voice_id} shut down")
                
            except Exception as e:
                self.logger.error(f"Error shutting down voice {voice_id}: {e}")
                # Remove from registry even if shutdown fails
                del self.voice_servers[voice_id]

    def shutdown_all(self):
        """Shutdown all voice servers and then self (model server only)"""
        if self.server_type != "model":
            return
            
        self.logger.info("Shutting down all voice servers...")
        for voice_id in list(self.voice_servers.keys()):
            self.shutdown_voice(voice_id)
            
        self.handle_signal(signal.SIGTERM, None)

    def start(self):
        """Start the server"""
        if self.is_running():
            self.logger.error("Server already running")
            sys.exit(1)

        self.daemonize()
        
        try:
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(self.socket_path)
            self.server.listen(5)
            self.server.setblocking(False)
            
            self.logger.info(f"Server listening on {self.socket_path}")
            
            while self.running:
                try:
                    ready = select.select([self.server], [], [], 1.0)
                    if ready[0]:
                        conn, addr = self.server.accept()
                        conn.setblocking(True)
                        self.update_activity()
                        thread = threading.Thread(target=self.handle_client, args=(conn,))
                        thread.daemon = True
                        thread.start()
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Server error: {e}")
                        time.sleep(1)
                        
        except Exception as e:
            self.logger.error(f"Fatal server error: {e}")
            self.cleanup()
            raise

    def is_running(self) -> bool:
        """Check if server is already running"""
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return True
            except (OSError, ValueError):
                self.cleanup()
        return False

    def handle_client(self, conn: socket.socket):
        """Handle client connection - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement handle_client")
