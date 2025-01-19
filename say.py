#!/usr/bin/env python3

import os
import sys
import json
import socket
import threading
import signal
import logging
import time
from pathlib import Path
import sounddevice as sd
import argparse
from kokoro_onnx import Kokoro
import subprocess
import psutil
import io
import numpy as np
import requests
import torch

# Configure logging to write to a file
logging.basicConfig(
    filename='/tmp/tts_daemon.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LANGUAGES = [
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)"),
    ("fr-fr", "French"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("cmn", "Mandarin Chinese")
]

class VoiceFetcher:
    def __init__(self):
        self.voices = [
            "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
        self.pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt"
    
    def fetch_voices(self, path: str = "voices.json") -> None:
        """Fetch voices if voices.json doesn't exist"""
        if os.path.exists(path):
            return
        
        print("\nFirst-time setup: Downloading voice models...")
        print("This may take several minutes but only needs to be done once.\n")
        
        voices_json = {}
        for i, voice in enumerate(self.voices, 1):
            url = self.pattern.format(voice=voice)
            print(f"Downloading voice {i}/{len(self.voices)}: {voice}")
            try:
                r = requests.get(url)
                r.raise_for_status()
                content = io.BytesIO(r.content)
                voice_data: np.ndarray = torch.load(content).numpy()
                voices_json[voice] = voice_data.tolist()
            except Exception as e:
                print(f"Error downloading voice {voice}: {str(e)}")
                if not voices_json:  # If no voices downloaded yet
                    raise RuntimeError("Failed to download any voices. Please check your internet connection and try again.")
                print("Continuing with downloaded voices...")
                break
        
        if not voices_json:
            raise RuntimeError("No voices were successfully downloaded")
        
        print("\nSaving voice data...")
        with open(path, "w") as f:
            json.dump(voices_json, f, indent=4)
        print(f"Voice setup complete! Saved to {path}\n")



def kill_all_daemons():
    """Kill all running TTS daemons"""
    killed = False

    # First try using PID files
    for file in os.listdir('/tmp'):
        if file.startswith('tts_daemon_') and file.endswith('.pid'):
            pid_file = os.path.join('/tmp', file)
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Terminated daemon process {pid} (via PID file)")
                    killed = True
                except ProcessLookupError:
                    print(f"Process {pid} not found, cleaning up files")
                
                # Clean up associated files
                voice_lang = file[11:-4]
                socket_file = f"/tmp/tts_socket_{voice_lang}"
                if os.path.exists(socket_file):
                    os.unlink(socket_file)
                os.unlink(pid_file)
                
            except (ValueError, OSError) as e:
                print(f"Error processing {pid_file}: {e}")
                try:
                    os.unlink(pid_file)
                except OSError:
                    pass

    # Then look for any Python processes running say.py
    script_path = os.path.abspath(__file__)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'say.py' in cmdline and proc.pid != os.getpid():
                    try:
                        proc.terminate()
                        print(f"Terminated daemon process {proc.pid} (found via process list)")
                        killed = True
                        # Give it a moment to terminate gracefully
                        proc.wait(timeout=3)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        # If it didn't terminate gracefully, try to kill it
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Clean up any leftover socket files
    for file in os.listdir('/tmp'):
        if file.startswith('tts_socket_'):
            socket_file = os.path.join('/tmp', file)
            try:
                os.unlink(socket_file)
                print(f"Cleaned up socket file: {socket_file}")
            except OSError:
                pass

    if not killed:
        print("No running TTS daemons found")
    
    return killed

class TTSServer:
    def __init__(self, voice, lang, socket_path=None, pid_file=None):
        self.voice = voice
        self.lang = lang
        self.socket_path = socket_path or f"/tmp/tts_socket_{voice}_{lang}"
        self.pid_file = pid_file or f"/tmp/tts_daemon_{voice}_{lang}.pid"
        self.kokoro = None
        self.lock = threading.Lock()
        self.running = True
        self.server = None
        self.voice_fetcher = VoiceFetcher()

        self.voice = voice
        self.lang = lang
        self.socket_path = socket_path or f"/tmp/tts_socket_{voice}_{lang}"
        self.pid_file = pid_file or f"/tmp/tts_daemon_{voice}_{lang}.pid"
        self.kokoro = None
        self.lock = threading.Lock()
        self.running = True
        self.server = None

    def daemonize(self):
        """Properly daemonize the process"""
        try:
            # First fork
            pid = os.fork()
            if pid > 0:
                os._exit(0)
        except OSError as err:
            logging.error(f'Initial fork failed: {err}')
            sys.exit(1)

        # Decouple from parent environment
        os.chdir('/')
        os.umask(0)
        os.setsid()

        # Second fork
        try:
            pid = os.fork()
            if pid > 0:
                os._exit(0)
        except OSError as err:
            logging.error(f'Second fork failed: {err}')
            sys.exit(1)

        # Replace file descriptors
        with open('/dev/null', 'r') as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open('/dev/null', 'a+') as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open('/dev/null', 'a+') as f:
            os.dup2(f.fileno(), sys.stderr.fileno())

        # Write pidfile
        pid = str(os.getpid())
        with open(self.pid_file, 'w+') as f:
            f.write(pid)

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGHUP, self.handle_signal)

    def handle_signal(self, signum, frame):
        """Handle signals gracefully"""
        self.running = False
        if self.server:
            self.server.close()
        self.cleanup()
        sys.exit(0)

    def initialize_kokoro(self):
        """Initialize Kokoro model if not already loaded"""
        if self.kokoro is None:
            script_dir = Path(__file__).parent.absolute()
            model_path = script_dir / "kokoro-v0_19.onnx"
            voices_path = script_dir / "voices.json"
            
            # Ensure voices exist
            self.voice_fetcher.fetch_voices(str(voices_path))
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    "Please download it from: https://github.com/thewh1teagle/kokoro-onnx/releases"
                )
            
            self.kokoro = Kokoro(str(model_path), str(voices_path))
            logging.info(f"Kokoro model loaded for voice {self.voice} and language {self.lang}")

    def handle_client(self, conn):
        """Handle client connection"""
        try:
            data = conn.recv(4096).decode('utf-8')
            if not data:
                return
            
            text = json.loads(data)['text']
            
            with self.lock:
                try:
                    samples, sample_rate = self.kokoro.create(
                        text,
                        voice=self.voice,
                        speed=1.0,
                        lang=self.lang
                    )
                    sd.play(samples, sample_rate)
                    sd.wait()
                    conn.send(json.dumps({"status": "success"}).encode('utf-8'))
                except Exception as e:
                    error_msg = {"status": "error", "message": str(e)}
                    conn.send(json.dumps(error_msg).encode('utf-8'))
        except Exception as e:
            logging.error(f"Error handling client: {e}")
        finally:
            conn.close()

    def cleanup(self):
        """Clean up socket and PID files"""
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self.pid_file):
                os.unlink(self.pid_file)
        except OSError as e:
            logging.error(f"Cleanup error: {e}")

    def is_running(self):
        """Check if a daemon for this voice+lang is already running"""
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if process exists
                return True
            except (OSError, ValueError):
                self.cleanup()
        return False

    def start(self):
        """Start the TTS daemon"""
        if self.is_running():
            logging.error(f"Daemon already running for voice {self.voice} and language {self.lang}")
            sys.exit(1)

        # Daemonize first, before initializing Kokoro
        self.daemonize()
        
        # Now initialize Kokoro after forking
        try:
            self.initialize_kokoro()
        except Exception as e:
            logging.error(f"Failed to initialize Kokoro: {e}")
            self.cleanup()
            sys.exit(1)
        
        # Create Unix domain socket
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.cleanup()
            self.server.bind(self.socket_path)
            self.server.listen(5)
            
            logging.info(f"TTS Daemon listening on {self.socket_path}")
            
            while self.running:
                try:
                    conn, addr = self.server.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn,))
                    thread.daemon = True
                    thread.start()
                except socket.error as e:
                    if self.running:
                        logging.error(f"Socket accept error: {e}")
                        
        except Exception as e:
            logging.error(f"Server error: {e}")
            self.cleanup()
            sys.exit(1)

def get_voice_from_input(voice_input, voices):
    """Convert voice input (name or number) to voice name"""
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(voices):
            return voices[index]
    elif voice_input in voices:
        return voice_input
    return None

def get_language_from_input(lang_input):
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

def send_request(text, voice, lang):
    """Send text to existing daemon or start a new one"""
    socket_path = f"/tmp/tts_socket_{voice}_{lang}"
    
    def try_connect():
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        request = {"text": text}
        client.send(json.dumps(request).encode('utf-8'))
        response = json.loads(client.recv(1024).decode('utf-8'))
        client.close()
        return response

    try:
        # Try connecting to existing daemon
        response = try_connect()
        if response["status"] != "success":
            print(f"Error: {response.get('message', 'Unknown error')}")
            sys.exit(1)
    except (FileNotFoundError, ConnectionRefusedError):
        # Start new daemon for this voice+lang combination
        pid = os.fork()
        if pid == 0:  # Child process
            server = TTSServer(voice, lang)
            try:
                server.start()
            except Exception as e:
                logging.error(f"Failed to start daemon: {e}")
                sys.exit(1)
        else:  # Parent process
            time.sleep(1)  # Wait for daemon to start
            max_retries = 3
            for i in range(max_retries):
                try:
                    response = try_connect()
                    if response["status"] == "success":
                        return
                except (FileNotFoundError, ConnectionRefusedError):
                    if i < max_retries - 1:
                        time.sleep(1)
                    else:
                        print("Error: Failed to connect to daemon after multiple attempts")
                        sys.exit(1)

def show_help(voices):
    """Display help message"""
    print("\nAvailable voices:")
    for i, voice in enumerate(voices, 1):
        print(f"  {i:2d}. {voice}")
    
    print("\nAvailable languages:")
    for i, (code, name) in enumerate(LANGUAGES, 1):
        print(f"  {i:2d}. {code:6s} - {name}")
    
    print("\nUsage examples:")
    print(f"  {sys.argv[0]} --voice <name/number> [--lang <code/number>] \"Text to speak\"")
    print(f"  echo \"Text\" | {sys.argv[0]} --voice <name/number>")

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech with persistent voice-specific daemons", add_help=False)
    parser.add_argument('--voice', help='Voice to use (name or number)', default='1')
    parser.add_argument('--lang', help='Language to use (code or number)', default='en-us')
    parser.add_argument('--list', action='store_true', help='List available voices and languages')
    parser.add_argument('--kill', action='store_true', help='Kill all running TTS daemons')
    parser.add_argument('text', nargs='*', help='Text to speak')
    
    args = parser.parse_args()

    # Handle kill command first
    if args.kill:
        kill_all_daemons()
        sys.exit(0)

    # Ensure voices are available before creating Kokoro instance
    script_dir = Path(__file__).parent.absolute()
    voices_path = script_dir / "voices.json"
    model_path = script_dir / "kokoro-v0_19.onnx"
    
    # Create voice fetcher and ensure voices exist
    voice_fetcher = VoiceFetcher()
    voice_fetcher.fetch_voices(str(voices_path))
    
    # Now create temporary Kokoro instance to get voices
    kokoro = Kokoro(str(model_path), str(voices_path))
    available_voices = sorted(list(kokoro.get_voices()))
    del kokoro  # Clean up the temporary instance
    
    if args.list or len(sys.argv) == 1:
        show_help(available_voices)
        sys.exit(0)
    
    # Get text from arguments or pipe
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = ' '.join(args.text)
    
    if not text:
        print("Error: No text provided")
        show_help(available_voices)
        sys.exit(1)
    
    # Validate voice
    voice = get_voice_from_input(args.voice, available_voices)
    if not voice:
        print(f"Error: Invalid voice '{args.voice}'")
        show_help(available_voices)
        sys.exit(1)
    
    # Validate language
    lang = get_language_from_input(args.lang)
    if not lang:
        print(f"Error: Invalid language '{args.lang}'")
        show_help(available_voices)
        sys.exit(1)
    
    send_request(text, voice, lang)

if __name__ == "__main__":
    main()
