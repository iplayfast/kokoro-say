#!/usr/bin/env python3

import os
import sys
import json
import socket
import logging
import time
import signal
import psutil
import inspect
import scipy.io.wavfile as wavfile
import numpy as np
import sounddevice as sd
import argparse
from pathlib import Path
from typing import Optional, Dict, Set, Tuple
from kokoro_onnx import Kokoro
log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/model_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('model_server')
logger.setLevel(log_level)


class ModelServer:
    def __init__(self, script_dir: Path, host: str = "127.0.0.1", port: int = 5000):    
        self.script_dir = script_dir
        self.host = host
        self.port = port
        self.socket = None
        self.kokoro: Optional[Kokoro] = None
        self.running = True
        self.voice_servers = {}
        self.next_port = 5001
        self.current_position = 0
        self.current_stream = None

    def initialize_model(self) -> None:
        model_path = self.script_dir / "kokoro-v0_19.onnx"
        voices_path = self.script_dir / "voices.json"
        
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(f"Required files not found in {self.script_dir}")
            
        try:
            self.kokoro = Kokoro(str(model_path), str(voices_path))
            logger.info("Kokoro model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def get_next_port(self) -> int:
        while True:
            if not any(port == self.next_port for port in self.voice_servers.values()):
                port = self.next_port
                self.next_port += 1
                return port
            self.next_port += 1

    def play_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        try:
            samples = samples.reshape(-1)  # Flatten to 1D array
            sd.play(samples, sample_rate)
            sd.wait()  # Wait until audio finishes playing
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise        
    def handle_client(self, conn: socket.socket) -> None:
        try:
            logger.debug("=== New client connection ===")
            conn.settimeout(float(os.getenv('SOCKET_TIMEOUT', '30')))
            
            length_data = b''
            while len(length_data) < 9:
                chunk = conn.recv(9 - len(length_data))
                if not chunk:
                    raise ValueError(f"Connection closed while reading length prefix (received {len(length_data)} bytes)")
                length_data += chunk
                logger.debug(f"Length prefix chunk: {chunk!r}")
            
            expected_length = int(length_data.decode('utf-8').strip())
            logger.debug(f"Expected data length: {expected_length}")
            
            data = bytearray()
            remaining = expected_length
            while remaining > 0:
                chunk = conn.recv(min(8192, remaining))
                if not chunk:
                    raise ValueError(f"Connection closed with {remaining} bytes remaining")
                data.extend(chunk)
                remaining -= len(chunk)
                logger.debug(f"Received data chunk: {len(chunk)} bytes, {remaining} remaining")
            
            logger.debug("Sending ACK")
            conn.send(b'ACK')
            
            request = json.loads(data.decode('utf-8'))
            logger.debug(f"Parsed request: {request}")

            if request.get("command") == "ping":
                return
            
            if request.get("command") == "exit":
                response = {"status": "success", "message": "Server shutting down"}
                response_data = json.dumps(response).encode('utf-8')
                length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
                conn.sendall(length_prefix + response_data)
                self.running = False
                return
                
            if "text" in request:
                if not self.kokoro:
                    self.initialize_model()
                    
                samples, sample_rate = self.kokoro.create(
                    request["text"],
                    voice=request["voice"],
                    speed=1.0,
                    lang=request["lang"]
                )
                
                if request.get("save_wav") and request.get("wav_path"):
                    samples_16bit = (samples * 32767).astype(np.int16)
                    wavfile.write(request["wav_path"], sample_rate, samples_16bit)
                    
                if request.get("play", True):
                    # Start playback in a non-blocking way
                    def play_async():
                        try:
                            samples_reshaped = samples.reshape(-1)
                            sd.play(samples_reshaped, sample_rate, blocking=False)
                        except Exception as e:
                            logger.error(f"Error in async playback: {e}")
                    
                    import threading
                    threading.Thread(target=play_async, daemon=True).start()
                    
                response = {
                    "status": "success",
                    "samples": samples.tolist(),
                    "sample_rate": sample_rate
                }
                
                response_data = json.dumps(response).encode('utf-8')
                length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
                conn.sendall(length_prefix + response_data)
                
                # Wait for final confirmation with timeout
                try:
                    fin = conn.recv(3)
                    if fin != b'FIN':
                        logger.warning(f"Client sent invalid final confirmation: {fin!r}")
                except socket.timeout:
                    logger.warning("Timeout waiting for client's final confirmation")
                    
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            error_response = {
                "status": "error",
                "message": str(e)
            }
            try:
                response_data = json.dumps(error_response).encode('utf-8')
                length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
                conn.sendall(length_prefix + response_data)
            except:
                pass
        finally:
            conn.close()
            
    def cleanup(self) -> None:
        if self.socket:
            self.socket.close()

    def start(self) -> None:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            logger.info(f"Model server listening on {self.host}:{self.port}")
            
            # Initialize model before accepting connections
            self.initialize_model()
            
            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    logger.debug(f"Accepted connection from {addr}")
                    try:
                        self.handle_client(conn)
                    except Exception as e:
                        logger.error(f"Client handler error: {e}")
                        try:
                            error_response = {
                                "status": "error",
                                "message": str(e)
                            }
                            response_data = json.dumps(error_response).encode('utf-8')
                            length_prefix = f"{len(response_data):08d}\n".encode('utf-8')
                            conn.sendall(length_prefix + response_data)
                        except:
                            pass
                        finally:
                            conn.close()
                except Exception as e:
                    logger.error(f"Accept error: {e}")
                    
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.cleanup()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-dir", required=True, type=Path)
    args = parser.parse_args()
    
    server = ModelServer(args.script_dir)
    server.start()