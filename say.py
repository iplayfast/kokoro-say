#!/usr/bin/env python3

import os
import sys
import warnings
import json
import socket
import logging
import argparse
import multiprocessing
import time
import termios
import tty
import signal
from pathlib import Path

from src import constants
from src.utils import VoiceManager, get_voice_from_input, get_language_from_input, show_help
from src.model_server import ModelServer
from src.fetchers import ensure_model_and_voices, VoiceFetcher


def print_process_info():
    """Print information about the current process."""
    pid = os.getpid()
    ppid = os.getppid()
    logger.info(f"Process info - PID: {pid}, Parent PID: {ppid}")

# Configure logging
logging.basicConfig(
    level=constants.DEFAULT_LOG_LEVEL,
    format=constants.LOG_FORMAT,
    handlers=[
        logging.FileHandler(constants.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Improved model server launch function for say.py

def ensure_model_server() -> bool:
    """Ensure model server is running, start it if needed with better error handling."""
    try:
        # Print process info
        print_process_info()
        
        # Check if server is already running
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(constants.SERVER_TIMEOUT)
        result = sock.connect_ex((constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT))
        sock.close()
        
        if result == 0:  # Port is open, server running
            logger.info("Model server already running")
            return True
            
        # Start model server explicitly using subprocess
        logger.info("Starting model server...")
        
        # Use subprocess to start the server
        import tempfile
        import subprocess
        
        # Get the absolute path to the current project directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create a script to launch the server with proper error handling
        launch_script = f"""#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level='DEBUG',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/kokoro_launch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("launcher")

# Add project directory to the Python path
project_dir = {repr(current_dir)}
logger.info(f"Project directory: {{project_dir}}")
sys.path.insert(0, project_dir)

try:
    logger.info("Starting model server")
    from src.model_server import ModelServer
    
    server = ModelServer()
    server.start()
except Exception as e:
    logger.error(f"Server startup failed: {{e}}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
"""
        
        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(launch_script)
            temp_script = f.name
        
        # Make the script executable
        os.chmod(temp_script, 0o755)
        
        # Start the server in a separate process
        process = subprocess.Popen(
            [sys.executable, temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Detach the process
        )
        
        # Wait for server to become available
        logger.info("Waiting for model server to start...")
        start_time = time.time()
        attempts = 0
        
        while time.time() - start_time < 30:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(constants.SERVER_TIMEOUT)
            result = sock.connect_ex((constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT))
            sock.close()
            
            if result == 0:
                logger.info("Model server started successfully")
                os.unlink(temp_script)  # Clean up temp file
                return True
            
            # Check if process is still running and capture output
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server process exited with code {process.returncode}")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                logger.error(f"Check /tmp/kokoro_launch.log for more details")
                os.unlink(temp_script)  # Clean up temp file
                return False
            
            # Exponential backoff
            sleep_time = min(2 ** attempts * 0.5, 1.0)
            time.sleep(sleep_time)
            attempts += 1
            
        # If we get here, timeout occurred
        logger.error("Timeout waiting for model server to start")
        logger.error("Check /tmp/kokoro_launch.log for error details")
        os.unlink(temp_script)  # Clean up temp file
        raise RuntimeError("Timeout waiting for model server to start")
        
    except Exception as e:
        logger.error(f"Failed to ensure model server: {e}")
        return False
    
def send_text(text: str, voice: str, lang: str, speed: float = 1.0, output_file: str = None) -> bool:
    """Send text to model server for synthesis."""
    try:
        # Ensure we have the voice downloaded before sending the request
        voice_manager = VoiceManager()
        voice_manager.ensure_voice_available(voice)
        
        # Connect to model server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(constants.SERVER_TIMEOUT)
        sock.connect((constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT))
        
        # Prepare message
        message = {
            "text": text,
            "voice": voice,
            "lang": lang,
            "speed": speed,
            "output_file": output_file  # Add output file path if specified
        }
        
        logger.debug(f"Sending message: {message}")
        
        try:
            # Use SocketProtocol to send JSON message
            from src.socket_protocol import SocketProtocol
            SocketProtocol.send_json(sock, message)
            
            # Receive response using SocketProtocol
            try:
                response = SocketProtocol.receive_json(sock)
                
                if response.get("status") == "success":
                    if output_file:
                        logger.info(f"Successfully synthesized speech and saved to {output_file}")
                    else:
                        logger.info("Successfully synthesized speech")
                    return True
                elif response.get("status") == "error":
                    error_msg = response.get("error", "Unknown server error")
                    logger.error(f"Server error: {error_msg}")
                    return False
                else:
                    logger.error(f"Unexpected response status: {response.get('status')}")
                    return False
                
            except Exception as e:
                logger.error(f"Error receiving response: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
        
    except socket.timeout:
        logger.error("Connection to model server timed out")
        return False
    except ConnectionRefusedError:
        logger.error("Connection to model server was refused")
        return False
    except Exception as e:
        logger.error(f"Failed to send text: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass
                        
def kill_server():
    """Send exit command to model server."""
    try:
        with socket.create_connection(
            (constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT),
            timeout=constants.SERVER_TIMEOUT
        ) as sock:
            request = {"command": "exit"}
            sock.sendall(json.dumps(request).encode())
            
            # Wait for confirmation
            try:
                sock.settimeout(5)
                response = sock.recv(1024)
                logger.info(f"Server shutdown response: {response.decode()}")
            except:
                pass
                
        logger.info("Kill command sent to model server")
        return True
    except Exception as e:
        logger.error(f"Failed to send kill command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Speech using Kokoro with voice persistence"
    )
    parser.add_argument('--voice', default='af_bella', help='Voice to use (name or number)')
    parser.add_argument('--lang', default='en-us', help='Language code or number')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--kill', action='store_true', help='Send kill command to servers')
    parser.add_argument('--list', action='store_true', help='List available voices and languages')
    parser.add_argument('--update-voices', action='store_true', help='Force update of available voices list')
    parser.add_argument('--no-exit', action='store_true', help='Don\'t exit after sending request (for debugging)')
    parser.add_argument('--output', help='Save audio to specified WAV file path')
    parser.add_argument('text', nargs='*', help='Text to speak')
    
    args = parser.parse_args()
    
    voice_manager = VoiceManager()
    
    try:
        # Handle voice list update
        if args.update_voices:
            logger.info("Forcing update of voice list...")
            voice_fetcher = VoiceFetcher()
            voices = voice_fetcher.get_available_voices(force_update=True)
            print("\nAvailable voices updated:")
            for i, voice in enumerate(sorted(voices), 1):
                print(f"  {i:2d}. {voice}")
            return

        # Handle list command
        if args.list or len(sys.argv) == 1:
            show_help(voice_manager, constants.LANGUAGES)
            return

        # Handle kill command
        if args.kill:
            kill_server()
            # Wait briefly to allow server to fully shut down
            time.sleep(0.5)
            return

        # Get input text
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text)
            
        if not text:
            logger.error("No text provided")
            parser.print_help()            
            sys.exit(1)

        # Validate voice and language
        try:
            voice = get_voice_from_input(args.voice, voice_manager)
            lang = get_language_from_input(args.lang, constants.LANGUAGES)
        except ValueError as e:
            logger.error(str(e))
            parser.print_help()            
            sys.exit(1)
            
        # Check if we need to wait for server shutdown before starting
        # This helps with the timing issue when running after a kill command
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(constants.SERVER_TIMEOUT)
        result = sock.connect_ex((constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT))
        sock.close()
        
        # If socket is still open, wait for it to close
        if result == 0:
            logger.info("Waiting for previous server to shutdown...")
            # Wait for up to 2 seconds for server to shut down
            for _ in range(4):
                time.sleep(0.5)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(constants.SERVER_TIMEOUT)
                result = sock.connect_ex((constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT))
                sock.close()
                if result != 0:
                    break
        
        # Ensure model server is running
        if not ensure_model_server():
            logger.error("Failed to start model server")            
            sys.exit(1)
            
        # Wait briefly to ensure server is fully initialized 
        time.sleep(0.5)
        
        # Send text for synthesis
        if not send_text(text, voice, lang, args.speed, args.output):
            logger.error("Failed to send text for synthesis")
            sys.exit(1)
            
        # If using no-exit option, wait for user to press a key
        if args.no_exit:
            print("Request sent successfully. Press any key to exit...")            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        elif args.output:
            print(f"Audio saved to {args.output}")
        else:
            # Wait briefly for audio to start playing before exiting
            print("Request sent successfully. Audio synthesis in progress...")            
            time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        logger.info("Exiting main program")
        sys.exit(0)
