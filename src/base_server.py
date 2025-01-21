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
from typing import Optional, Any, Dict

class UnixSocketServer:
    """Base class for Unix domain socket servers with clean logging separation"""
    
    def __init__(self, socket_path: str, pid_file: str, idle_timeout: int = 300):
        """
        Initialize the base server with separate logging
        
        Args:
            socket_path: Path to Unix domain socket
            pid_file: Path to PID file
            idle_timeout: Seconds of inactivity before shutdown (default 300 = 5 minutes)
        """
        self.socket_path = socket_path
        self.pid_file = pid_file
        self.lock_file = f"{pid_file}.lock"
        self.lock = threading.Lock()
        self.running = True
        self.server: Optional[socket.socket] = None
        self.idle_timeout = idle_timeout
        self.last_activity = time.time()
        self.activity_check_interval = 30  # Check for inactivity every 30 seconds
        
        # Configure logging to write to file only, not stdout
        self.setup_logging()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        
        # Clean up any leftover files on init
        self.cleanup()
        
        # Start activity monitor thread
        self.activity_monitor = threading.Thread(target=self._monitor_activity)
        self.activity_monitor.daemon = True
        self.activity_monitor.start()

    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = time.time()
        
    def _monitor_activity(self):
        """Monitor for inactivity and shutdown if idle too long"""
        while self.running:
            time.sleep(self.activity_check_interval)
            if self.running:  # Check again in case we were shut down
                idle_time = time.time() - self.last_activity
                if idle_time >= self.idle_timeout:
                    self.logger.info(f"Server idle for {idle_time:.1f} seconds, shutting down")
                    self.handle_signal(signal.SIGTERM, None)
                    break

    def setup_logging(self):
        """Configure logging to write to separate file descriptor"""
        # Create a specific logger for this server instance
        logger_name = f"server_{os.path.basename(self.socket_path)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to prevent duplication
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Create file handler with its own formatter
        log_path = f"/tmp/{logger_name}.log"
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(fh)
        
        # Prevent propagation to root logger to avoid stdout
        self.logger.propagate = False

    def send_response(self, conn, response):
        """Send response with length prefix, using lock to prevent interleaving"""
        with self.lock:
            try:
                response_data = json.dumps(response).encode('utf-8')
                length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
                
                # Send length prefix and data atomically
                conn.sendall(length_prefix + response_data)
                
            except Exception as e:
                self.logger.error(f"Error sending response: {e}")
                raise

    def receive_request(self, conn):
        """Receive length-prefixed request data"""
        try:
            # Read exactly 9 bytes (8 length + newline)
            length_data = conn.recv(9)
            if not length_data:
                raise RuntimeError("Empty request")
            
            # Parse length, stripping newline
            try:
                expected_length = int(length_data.decode().strip())
            except ValueError as e:
                raise RuntimeError(f"Invalid length prefix: {e}")
            
            # Read exactly the expected number of bytes
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

    def daemonize(self):
        """Daemonize the process, maintaining separate logging"""
        try:
            # First fork
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as err:
            self.logger.error(f'First fork failed: {err}')
            sys.exit(1)
        
        # Decouple from parent environment
        os.chdir('/')
        os.umask(0)
        os.setsid()
        
        # Second fork
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as err:
            self.logger.error(f'Second fork failed: {err}')
            sys.exit(1)
        
        # Close all file descriptors except log
        log_fds = {handler.stream.fileno() for handler in self.logger.handlers 
                  if hasattr(handler, 'stream')}
        
        # Redirect stdin, stdout, stderr to /dev/null
        with open(os.devnull) as devnull:
            os.dup2(devnull.fileno(), sys.stdin.fileno())
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
        
        # Write PID file
        pid = os.getpid()
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))
        except Exception as e:
            self.logger.error(f"Failed to write PID file: {e}")
            sys.exit(1)
    
    def handle_client(self, conn: socket.socket):
        """Handle client connection - should be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement handle_client")

    def start(self):
        """Start the server with proper logging separation"""
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
                        self.update_activity()  # Connection received, update activity timestamp
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

    def is_running(self):
        """Check if daemon is already running"""
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return True
            except (OSError, ValueError):
                self.cleanup()
        return False