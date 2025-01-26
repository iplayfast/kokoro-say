#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import time
import signal
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceServer:
    def __init__(
        self,
        voice: str,
        lang: str,
        port: int,
        model_host: str = "127.0.0.1",
        model_port: int = 5000
    ):
        self.voice = voice
        self.lang = lang
        self.port = port
        self.host = "127.0.0.1"
        self.model_host = model_host
        self.model_port = model_port
        self.socket: Optional[socket.socket] = None
        self.running = True
        self.current_stream: Optional[sd.OutputStream] = None
        
    def register_with_model_server(self) -> bool:
        """Register this voice server with the model server"""
        try:
            with socket.create_connection((self.model_host, self.model_port), timeout=5) as conn:
                request = {
                    "command": "register",
                    "voice": self.voice,
                    "lang": self.lang,
                    "port": self.port
                }
                data = json.dumps(request).encode()
                length_prefix = f"{len(data):08d}\n".encode()
                conn.sendall(length_prefix + data)
                
                length_data = conn.recv(9)
                expected_length = int(length_data.decode().strip())
                response_data = bytearray()
                remaining = expected_length
                
                while remaining > 0:
                    chunk = conn.recv(min(8192, remaining))
                    if not chunk:
                        break
                    response_data.extend(chunk)
                    remaining -= len(chunk)
                
                response = json.loads(response_data.decode())
                return response.get("status") == "success"
                
        except Exception as e:
            logger.error(f"Failed to register with model server: {e}")
            return False
            
    def stop_current_stream(self) -> None:
        """Safely stop the current audio stream if one exists"""
        if self.current_stream:
            try:
                self.current_stream.stop()
                self.current_stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            finally:
                self.current_stream = None

    def send_to_model_server(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to model server and return response"""
        try:
            with socket.create_connection((self.model_host, self.model_port), timeout=30) as conn:
                data = json.dumps(request).encode()
                length_prefix = f"{len(data):08d}\n".encode()
                conn.sendall(length_prefix + data)
                
                length_data = conn.recv(9)
                expected_length = int(length_data.decode().strip())
                response_data = bytearray()
                remaining = expected_length
                
                while remaining > 0:
                    chunk = conn.recv(min(8192, remaining))
                    if not chunk:
                        break
                    response_data.extend(chunk)
                    remaining -= len(chunk)
                
                return json.loads(response_data.decode())
                
        except Exception as e:
            logger.error(f"Error communicating with model server: {e}")
            raise

    def handle_client(self, conn: socket.socket) -> None:
        """Handle incoming client connection"""
        try:
            length_data = conn.recv(9)
            if not length_data:
                return
                
            expected_length = int(length_data.decode().strip())
            data = bytearray()
            remaining = expected_length
            
            while remaining > 0:
                chunk = conn.recv(min(8192, remaining))
                if not chunk:
                    break
                data.extend(chunk)
                remaining -= len(chunk)
            
            request = json.loads(data.decode())
            
            # Handle exit command
            if request.get("command") == "exit":
                logger.info("Received exit command")
                self.stop_current_stream()
                response = {"status": "success", "message": "Server shutting down"}
                response_data = json.dumps(response).encode()
                length_prefix = f"{len(response_data):08d}\n".encode()
                conn.sendall(length_prefix + response_data)
                self.running = False
                return
            
            # Handle stop command
            if request.get("command") == "stop":
                logger.info("Received stop command")
                self.stop_current_stream()
                response = {"status": "success", "message": "Playback stopped"}
                response_data = json.dumps(response).encode()
                length_prefix = f"{len(response_data):08d}\n".encode()
                conn.sendall(length_prefix + response_data)
                return
            
            # Handle text-to-speech request
            if "text" in request:
                # Stop any current playback
                self.stop_current_stream()
                
                # Add voice and language to request
                request["voice"] = self.voice
                request["lang"] = self.lang
                
                # Forward request to model server
                response = self.send_to_model_server(request)
                
                if response.get("status") == "success":
                    # Convert samples back to numpy array
                    samples = np.array(response["samples"])
                    sample_rate = response["sample_rate"]
                    
                    if request.get("play", True):
                        # Create and start new audio stream
                        self.current_stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=1,
                            callback=None,
                            finished_callback=self.stop_current_stream,
                            blocksize=1024
                        )
                        self.current_stream.start()
                        self.current_stream.write(samples,non_blocking=True)
                
                # Send response back to client
                response_data = json.dumps(response).encode()
                length_prefix = f"{len(response_data):08d}\n".encode()
                conn.sendall(length_prefix + response_data)
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            error_response = {
                "status": "error",
                "message": str(e)
            }
            try:
                response_data = json.dumps(error_response).encode()
                length_prefix = f"{len(response_data):08d}\n".encode()
                conn.sendall(length_prefix + response_data)
            except:
                pass
        finally:
            conn.close()

    def start(self) -> None:
        """Start the voice server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            logger.info(f"Voice server for {self.voice}/{self.lang} listening on {self.host}:{self.port}")
            
            # Register with model server
            if not self.register_with_model_server():
                raise RuntimeError("Failed to register with model server")
            
            # Set up signal handlers
            signal.signal(signal.SIGTERM, lambda signo, frame: self.handle_shutdown())
            signal.signal(signal.SIGINT, lambda signo, frame: self.handle_shutdown())
            
            while self.running:
                self.socket.settimeout(1.0)
                try:
                    conn, addr = self.socket.accept()
                    logger.debug(f"Connection from {addr}")
                    self.handle_client(conn)
                except socket.timeout:
                    continue
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.cleanup()

    def handle_shutdown(self) -> None:
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.running = False

    def cleanup(self) -> None:
        """Clean up resources"""
        self.stop_current_stream()
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info("Server shut down cleanly")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice-specific TTS server")
    parser.add_argument("--voice", required=True, help="Voice name")
    parser.add_argument("--lang", required=True, help="Language code")
    parser.add_argument("--port", required=True, type=int, help="Port to listen on")
    
    args = parser.parse_args()
    
    server = VoiceServer(args.voice, args.lang, args.port)
    server.start()