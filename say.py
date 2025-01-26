#!/usr/bin/env python3

import os
import sys
import json
import logging
import argparse
import socket
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import inspect
from src.fetchers import VoiceFetcher, ensure_model_and_voices
from src.constants import (
    LANGUAGES,
    DEFAULT_VOICE,
    DEFAULT_LANG,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-to-Speech with persistent voice-specific servers"
    )
    parser.add_argument('--voice', help='Voice to use (name or number)', default=DEFAULT_VOICE)
    parser.add_argument('--lang', help='Language to use (code or number)', default=DEFAULT_LANG)
    parser.add_argument('--list', action='store_true', help='List available voices and languages')
    parser.add_argument('--kill', action='store_true', help='Kill all running TTS servers')
    parser.add_argument('--log', action='store_true', help='Enable detailed logging')
    parser.add_argument('text', nargs='*', help='Text to speak')
    parser.add_argument('--output', help='Output WAV file path', type=str)
    return parser.parse_args()

def show_help(voices: List[str]) -> None:
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"  {i:2d}. {voice}")
    
    print("\nAvailable languages:")
    for i, (code, name) in enumerate(LANGUAGES, 1):
        print(f"  {i:2d}. {code:6s} - {name}")
    
    print("\nUsage examples:")
    cmd = os.path.basename(sys.argv[0])
    print(f"  1. List available voices and languages:")
    print(f"     {cmd} --help")
    print(f"\n  2. Using voice and language by name/code:")
    print(f"     {cmd} --voice {voices[0]} --lang en-us \"Hello World\"")
    print(f"\n  3. Using voice and language by number:")
    print(f"     {cmd} --voice 1 --lang 3 \"Bonjour le monde\"")
    print(f"\n  4. Using pipe with mixed selection:")
    print(f"     echo \"こんにちは\" | {cmd} --voice 1 --lang ja")
    print(f"\n  5. Kill all TTS servers:")
    print(f"     {cmd} --kill")

def send_request(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = socket.create_connection((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        conn.settimeout(float(os.getenv('SOCKET_TIMEOUT', '30')))
    
        data = json.dumps(request).encode('utf-8')
        length_prefix = f"{len(data):08d}\n".encode('utf-8')        
        conn.sendall(length_prefix)        
        time.sleep(0.1)      
        conn.sendall(data)        
        
        ack = conn.recv(3)
        if ack != b'ACK':
            raise RuntimeError(f"Server did not acknowledge data receipt: got {ack!r}")
        
        raw_length = conn.recv(9)
        expected_length = int(raw_length.decode('utf-8').strip())
        
        response_data = bytearray()
        remaining = expected_length
        while remaining > 0:
            chunk = conn.recv(min(8192, remaining))
            if not chunk:
                raise RuntimeError(f"Connection closed with {remaining} bytes remaining")
            response_data.extend(chunk)
            remaining -= len(chunk)
        
        response = json.loads(response_data.decode('utf-8'))
        conn.send(b'FIN')
        conn.close()
        
        return response
        
    except Exception as e:
        raise RuntimeError(f"Error communicating with server: {e}")
def send_request1(request: Dict[str, Any]) -> Dict[str, Any]:
    """Send request to model server and wait for confirmation"""
    try:
        print("Connecting to model server...")
        conn = socket.create_connection((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        conn.settimeout(float(os.getenv('SOCKET_TIMEOUT', '30')))
        print("Connected")
    
        # Send data with length prefix
        data = json.dumps(request).encode('utf-8')
        length_prefix = f"{len(data):08d}\n".encode('utf-8')        
        conn.sendall(length_prefix)        
        time.sleep(0.1)  # Give server time to process length prefix        
        conn.sendall(data)        
        # Wait for receipt confirmation
        ack = conn.recv(3)
        if ack != b'ACK':
            raise RuntimeError(f"Server did not acknowledge data receipt: got {ack!r}")
        
        # Handle response
        raw_length = conn.recv(9)
        if not raw_length:
            raise RuntimeError("Connection closed before receiving response length")
            
        expected_length = int(raw_length.decode('utf-8').strip())
        
        response_data = bytearray()
        remaining = expected_length
        
        while remaining > 0:
            chunk = conn.recv(min(8192, remaining))
            if not chunk:
                raise RuntimeError(f"Connection closed with {remaining} bytes remaining")
            response_data.extend(chunk)
            remaining -= len(chunk)
        
        # Send final confirmation
        conn.send(b'FIN')
        
        return json.loads(response_data.decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Error communicating with server: {e}")
    finally:
        conn.close()

def ensure_model_server() -> bool:
    """Ensure model server is running and ready"""
    def test_server():
        try:
            with socket.create_connection((MODEL_SERVER_HOST, MODEL_SERVER_PORT), timeout=1) as test_conn:
                # Send test request
                test_data = json.dumps({"command": "ping"}).encode('utf-8')
                length_prefix = f"{len(test_data):08d}\n".encode('utf-8')
                test_conn.sendall(length_prefix + test_data)
                
                # Check for ACK
                ack = test_conn.recv(3)
                return ack == b'ACK'
        except:
            return False

    
    if test_server():    
        return True
    
    script_dir = Path(__file__).parent.absolute()
    server_path = script_dir / "src" / "model_server.py"
    
    pid = os.fork()
    if pid == 0:  # Child process
        try:
            os.execv(sys.executable, [
                sys.executable,
                str(server_path),
                "--script-dir",
                str(script_dir)
            ])
        except Exception as e:
            logger.error(f"Failed to start model server: {e}")
            sys.exit(1)
        
    start_time = time.time()
    while time.time() - start_time < 30:
        if test_server():    
#            time.sleep(2)  # Extra wait for stability
            return True
        time.sleep(0.5)
    
    return False

def get_voice_from_input(voice_input: str, available_voices: List[str]) -> str:
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(available_voices):
            return available_voices[index]
    elif voice_input in available_voices:
        return voice_input
    return ""

def get_language_from_input(lang_input: str) -> str:
    if lang_input.isdigit():
        index = int(lang_input) - 1
        if 0 <= index < len(LANGUAGES):
            return LANGUAGES[index][0]
    else:
        for code, _ in LANGUAGES:
            if code == lang_input:
                return code
    return ""

def main():
    args = parse_args()
    
    try:
        if args.kill:
            try:
                send_request({"command": "exit"})
                logger.info("Servers terminated successfully")
            except:
                logger.error("Failed to terminate servers gracefully")
            sys.exit(0)

        voice_fetcher = VoiceFetcher()
        available_voices = sorted(voice_fetcher.get_available_voices())

        if args.list or len(sys.argv) == 1:
            show_help(available_voices)
            sys.exit(0)

        # Get text from args or stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text) if args.text else ""
        
        if not text:
            logger.error("No text provided")
            show_help(available_voices)
            sys.exit(1)

        # Validate voice and language
        voice = get_voice_from_input(args.voice, available_voices)
        if not voice:
            logger.error(f"Invalid voice '{args.voice}'")
            show_help(available_voices)
            sys.exit(1)

        lang = get_language_from_input(args.lang)
        if not lang:
            logger.error(f"Invalid language '{args.lang}'")
            show_help(available_voices)
            sys.exit(1)
            
        # Ensure model server is running
        if not ensure_model_server():
            raise RuntimeError("Could not start model server")
            
        # Send TTS request
        response = send_request({
            "text": text,
            "voice": voice,
            "lang": lang,
            "play": not args.output,
            "save_wav": bool(args.output),
            "wav_path": args.output
        })
        
        if response.get("status") == "error":
            raise RuntimeError(response.get("message", "Unknown error"))
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        try:
            send_request({"command": "exit"})
        except:
            pass
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()