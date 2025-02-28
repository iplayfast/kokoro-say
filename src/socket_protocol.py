#!/usr/bin/env python3

import socket
import json
import numpy as np
from typing import Dict, Any, Union, Optional, Tuple

from src.constants import (
    MAX_CHUNK_SIZE,
    SOCKET_HEADER_SIZE
)

class SocketProtocol:
    """Enhanced socket protocol for audio streaming."""
    
    # Command types
    CMD_JSON = b"JSON"   # For JSON messages
    CMD_AUDIO = b"AUDI"  # For binary audio data
    CMD_END = b"DONE"    # End of stream
    
    MAX_CHUNK_SIZE = MAX_CHUNK_SIZE
    HEADER_SIZE = SOCKET_HEADER_SIZE
    
    @staticmethod
    def send_json(sock: socket.socket, data: Dict[str, Any]) -> None:
        """Send JSON data with proper framing."""
        json_data = json.dumps(data).encode('utf-8')
        header = f"{len(json_data):08d}\n".encode('utf-8')
        sock.sendall(SocketProtocol.CMD_JSON + header + json_data)
    
    @staticmethod
    def send_audio_chunk(sock: socket.socket, audio_data: np.ndarray, is_final: bool = False) -> None:
        """Send binary audio data efficiently."""
        if isinstance(audio_data, np.ndarray):
            # Convert to binary, preserving float32 format
            binary_data = audio_data.astype(np.float32).tobytes()
        else:
            binary_data = audio_data
            
        # Prepare header
        header = f"{len(binary_data):08d}\n".encode('utf-8')
        
        # Choose command based on whether this is the final chunk
        command = SocketProtocol.CMD_END if is_final else SocketProtocol.CMD_AUDIO
        
        # Send command, header, and data
        sock.sendall(command + header + binary_data)
    
    @staticmethod
    def receive_message(sock: socket.socket, timeout: Optional[float] = None) -> Tuple[bytes, bytes]:
        """Receive a message with type detection.
        
        Returns:
            Tuple[bytes, bytes]: (command, data)
        """
        original_timeout = sock.gettimeout()
        if timeout is not None:
            sock.settimeout(timeout)
            
        try:
            # Read command (4 bytes)
            command = sock.recv(4)
            if not command or len(command) < 4:
                raise RuntimeError("Connection closed or invalid command")
            
            # Read header (8 bytes + newline)
            header = sock.recv(SocketProtocol.HEADER_SIZE + 1)
            if not header or len(header) < SocketProtocol.HEADER_SIZE:
                raise RuntimeError("Connection closed while reading header")
                
            try:
                # Parse length from header
                length = int(header.decode().strip())
                if length < 0 or length > 100 * 1024 * 1024:  # 100MB max for safety
                    raise RuntimeError(f"Invalid message length: {length}")
            except ValueError:
                raise RuntimeError(f"Invalid length header: {header!r}")
            
            # Read data in chunks
            data = bytearray()
            received = 0
            while received < length:
                chunk_size = min(SocketProtocol.MAX_CHUNK_SIZE, length - received)
                chunk = sock.recv(chunk_size)
                if not chunk:
                    raise RuntimeError(f"Connection closed after receiving {received} of {length} bytes")
                data.extend(chunk)
                received += len(chunk)
                
            # Return command and data
            return command, bytes(data)
            
        finally:
            # Restore original timeout
            sock.settimeout(original_timeout)
    
    @staticmethod
    def receive_json(sock: socket.socket, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Receive and parse JSON message."""
        command, data = SocketProtocol.receive_message(sock, timeout)
        if command != SocketProtocol.CMD_JSON:
            raise RuntimeError(f"Expected JSON command, got {command}")
        
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON: {e}")
