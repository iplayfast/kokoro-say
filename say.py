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
from src.fetchers import ensure_voice


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
        logging.FileHandler('{constants.LOG_FILE}'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("launcher")
logger.info("==== MODEL SERVER LAUNCHER STARTED ====")

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
                logger.error(f"Check {constants.LOG_FILE} for more details")
                os.unlink(temp_script)  # Clean up temp file
                return False
            
            # Exponential backoff
            sleep_time = min(2 ** attempts * 0.5, 1.0)
            time.sleep(sleep_time)
            attempts += 1
            
        # If we get here, timeout occurred
        logger.error("Timeout waiting for model server to start")
        logger.error(f"Check {constants.LOG_FILE} for error details")
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
    """Send exit command to model server and ensure processes are terminated."""
    try:
        # Try to connect to the model server
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(constants.SERVER_TIMEOUT)
            sock.connect((constants.MODEL_SERVER_HOST, constants.MODEL_SERVER_PORT))
        except ConnectionRefusedError:
            logger.info("Model server not running. Nothing to kill.")
            return True
        except Exception as e:
            logger.error(f"Could not connect to model server: {e}")
            # Continue with forceful kill as fallback
            return force_kill_processes()
            
        # Send properly formatted exit command
        try:
            from src.socket_protocol import SocketProtocol
            request = {"command": "exit"}
            SocketProtocol.send_json(sock, request)
            
            # Wait for confirmation with proper error handling
            try:
                sock.settimeout(5)
                response = SocketProtocol.receive_json(sock)
                logger.info(f"Server shutdown response: {response}")
                
                if response.get("status") != "shutdown_initiated":
                    logger.warning(f"Unexpected response: {response}")
            except Exception as e:
                logger.warning(f"Error receiving shutdown confirmation: {e}")
        finally:
            sock.close()
            
        # Wait for processes to terminate
        logger.info("Kill command sent to model server, waiting for shutdown...")
        
        # Give processes time to terminate gracefully
        for i in range(6):  # Wait up to 3 seconds
            time.sleep(0.5)
            if not check_processes_running():
                logger.info("All processes terminated successfully")
                return True
        
        # If processes are still running, force kill them
        logger.warning("Processes still running after graceful shutdown attempt, forcing termination...")
        return force_kill_processes()
        
    except Exception as e:
        logger.error(f"Failed to send kill command: {e}")
        # Try forceful kill as fallback
        return force_kill_processes()

def check_processes_running():
    """Check if any kokoro TTS processes are still running."""
    try:
        import subprocess
        import re
        
        # Get list of running python processes
        output = subprocess.check_output(["ps", "aux"], universal_newlines=True)
        
        # Find model server and voice server processes
        model_pattern = re.compile(r'python.*kokoro.*model_server|tmpxz.*\.py')
        voice_pattern = re.compile(r'python.*voice_server')
        
        model_running = any(model_pattern.search(line) for line in output.splitlines())
        voice_running = any(voice_pattern.search(line) for line in output.splitlines())
        
        return model_running or voice_running
    except Exception as e:
        logger.error(f"Error checking for running processes: {e}")
        return False  # Assume no processes running to avoid false positives

def force_kill_processes():
    """Forcefully terminate all kokoro TTS processes."""
    try:
        import subprocess
        import signal
        import os
        
        logger.info("Forcefully terminating all kokoro TTS processes...")
        
        # Find all related processes
        try:
            # Get model server PID(s)
            output = subprocess.check_output(
                ["pgrep", "-f", "python.*kokoro.*model_server|tmpxz.*\\.py"], 
                universal_newlines=True
            ).strip()
            pids = output.split('\n') if output else []
            
            # Kill model server process(es)
            for pid in pids:
                if pid.strip():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"Killed model server process {pid}")
                    except Exception as e:
                        logger.error(f"Error killing model server process {pid}: {e}")
            
            # Get voice server PID(s)
            output = subprocess.check_output(
                ["pgrep", "-f", "python.*voice_server"], 
                universal_newlines=True
            ).strip()
            pids = output.split('\n') if output else []
            
            # Kill voice server process(es)
            for pid in pids:
                if pid.strip():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"Killed voice server process {pid}")
                    except Exception as e:
                        logger.error(f"Error killing voice server process {pid}: {e}")
        except subprocess.CalledProcessError:
            # No processes found
            pass
        
        # Clean up socket files
        socket_pattern = f"{constants.SOCKET_BASE_PATH}_*"
        import glob
        for socket_file in glob.glob(socket_pattern):
            try:
                os.unlink(socket_file)
                logger.info(f"Removed socket file: {socket_file}")
            except Exception as e:
                logger.warning(f"Failed to remove socket file {socket_file}: {e}")
        
        # Verify all processes are terminated
        if not check_processes_running():
            logger.info("All processes successfully terminated")
            return True
        else:
            logger.error("Failed to terminate all processes")
            return False
            
    except Exception as e:
        logger.error(f"Error in force_kill_processes: {e}")
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
    # Rename --no-exit to --interactive for clarity
    parser.add_argument('--interactive', action='store_true', 
                        help='Start interactive mode: read multiple lines from stdin and speak each one')
    parser.add_argument('--log-level', default=constants.DEFAULT_LOG_LEVEL, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--output', help='Save audio to specified WAV file path')
    parser.add_argument('text', nargs='*', help='Text to speak')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(args.log_level)
    
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
        if not sys.stdin.isatty() and not args.interactive:
            text = sys.stdin.read().strip()
        else:
            text = ' '.join(args.text)
            
        if not text and not args.interactive:
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
        
        # Ensure model server is running
        if not ensure_model_server():
            logger.error("Failed to start model server")            
            sys.exit(1)
            
        # Wait briefly to ensure server is fully initialized 
        time.sleep(0.5)
        
        # Interactive mode: continuously read from stdin and speak each line
        if args.interactive:
            print(f"Interactive mode enabled. Using voice: {voice}, language: {lang}, speed: {args.speed}")
            print("Enter text to speak (Ctrl+D or Ctrl+C to exit):")
            
            line_num = 1
            try:
                while True:
                    try:
                        # Print a prompt
                        print(f"\n[{line_num}] > ", end="", flush=True)
                        line = input()
                        
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        print(f"Speaking: {line}")
                        
                        # Generate output filename if --output is specified
                        output_file = None
                        if args.output:
                            # Create sequentially numbered files
                            base, ext = os.path.splitext(args.output)
                            if not ext:
                                ext = ".wav"
                            output_file = f"{base}_{line_num}{ext}"
                            
                        # Send text for synthesis
                        if not send_text(line, voice, lang, args.speed, output_file):
                            logger.error("Failed to send text for synthesis")
                            continue
                            
                        if output_file:
                            print(f"Audio saved to {output_file}")
                        else:
                            # Just a small delay to ensure audio starts playing
                            time.sleep(0.2)
                            
                        line_num += 1
                            
                    except EOFError:
                        # Exit on Ctrl+D
                        print("\nExiting interactive mode.")
                        break
            except KeyboardInterrupt:
                # Exit on Ctrl+C
                print("\nInterrupted. Exiting interactive mode.")
                
        else:
            # Normal mode: process single text
            if not send_text(text, voice, lang, args.speed, args.output):
                logger.error("Failed to send text for synthesis")
                sys.exit(1)
                
            if args.output:
                print(f"Audio saved to {args.output}")
            else:
                # Wait briefly for audio to start playing before exiting
                print("Request sent successfully. Audio synthesis in progress...")            
                time.sleep(0.2)
            
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