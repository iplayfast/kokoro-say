#!/usr/bin/env python3

"""
TTS Client Library - Common code for interacting with the Kokoro TTS server.
"""

import os
import sys
import json
import socket
import logging
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    SERVER_TIMEOUT,
    LOG_FILE,
    LOG_FORMAT,
    SAMPLE_RATE,
    LANGUAGES
)
from src.utils import VoiceManager, get_voice_from_input, get_language_from_input
from src.socket_protocol import SocketProtocol

logger = logging.getLogger(__name__)

def print_process_info() -> None:
    """Print information about the current process."""
    pid = os.getpid()
    ppid = os.getppid()
    logger.info(f"Process info - PID: {pid}, Parent PID: {ppid}")

def is_model_server_running() -> bool:
    """Check if the model server is already running."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SERVER_TIMEOUT)
        result = sock.connect_ex((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking if model server is running: {e}")
        return False

def ensure_model_server() -> bool:
    """Ensure model server is running, start it if needed with better error handling."""
    try:
        # Print process info
        print_process_info()
        
        # Check if server is already running
        if is_model_server_running():
            logger.info("Model server already running")
            return True
            
        # Start model server explicitly using subprocess
        logger.info("Starting model server...")
        
        # Get the absolute path to the current project directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
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
        logging.FileHandler('{LOG_FILE}'),
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
            if is_model_server_running():
                logger.info("Model server started successfully")
                os.unlink(temp_script)  # Clean up temp file
                return True
            
            # Check if process is still running and capture output
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server process exited with code {process.returncode}")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                logger.error(f"Check {LOG_FILE} for more details")
                os.unlink(temp_script)  # Clean up temp file
                return False
            
            # Exponential backoff
            sleep_time = min(2 ** attempts * 0.5, 1.0)
            time.sleep(sleep_time)
            attempts += 1
            
        # If we get here, timeout occurred
        logger.error("Timeout waiting for model server to start")
        logger.error(f"Check {LOG_FILE} for error details")
        os.unlink(temp_script)  # Clean up temp file
        raise RuntimeError("Timeout waiting for model server to start")
        
    except Exception as e:
        logger.error(f"Failed to ensure model server: {e}")
        return False

def send_text(
    text: str, 
    voice: str, 
    lang: str, 
    speed: float = 1.0,
    output_file: Optional[str] = None, 
    wait_for_completion: bool = False
) -> bool:
    """
    Send text to model server for synthesis.
    
    Args:
        text: Text to synthesize
        voice: Voice to use
        lang: Language code
        speed: Speech speed multiplier
        output_file: Optional path to save WAV file
        wait_for_completion: Whether to wait for synthesis to complete
        
    Returns:
        bool: True if request was successful
    """
    try:
        # Ensure we have the voice downloaded before sending the request
        voice_manager = VoiceManager()
        # Track if voice was just downloaded
        _, voice_downloaded = voice_manager.ensure_voice_available(voice)
        
        # Adjust timeout based on whether voice was just downloaded
        socket_timeout = 60.0 if voice_downloaded else SERVER_TIMEOUT
        logger.info(f"Using socket timeout of {socket_timeout}s (voice downloaded: {voice_downloaded})")
        
        # Connect to model server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(socket_timeout)  # Use extended timeout if voice was just downloaded
        sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        
        # Prepare message
        message = {
            "text": text,
            "voice": voice,
            "lang": lang,
            "speed": speed,
            "output_file": output_file,
            "wait_for_completion": wait_for_completion,
            "first_time_voice": voice_downloaded  # Inform server this is a first-time voice
        }
        
        logger.debug(f"Sending message: {message}")
        
        try:
            # Send JSON message
            SocketProtocol.send_json(sock, message)
            
            # Receive response with extended timeout
            try:
                response = SocketProtocol.receive_json(sock, timeout=socket_timeout)
                
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
                    logger.info(f"Server response: {response}")
                    return True
                
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

def get_direct_audio(
    text: str, 
    voice: str, 
    lang: str, 
    speed: float = 1.0
) -> Tuple[bool, Optional[List[float]], Optional[int]]:
    """
    Get audio data directly from the model server without playing it.
    
    Args:
        text: Text to synthesize
        voice: Voice to use
        lang: Language code
        speed: Speech speed multiplier
        
    Returns:
        Tuple[bool, Optional[List[float]], Optional[int]]: 
            (success, audio_data, sample_rate)
    """
    try:
        # Ensure we have the voice downloaded before sending the request
        voice_manager = VoiceManager()
        _, voice_downloaded = voice_manager.ensure_voice_available(voice)
        
        # Adjust timeout based on whether voice was just downloaded
        socket_timeout = 60.0 if voice_downloaded else SERVER_TIMEOUT
        
        # Connect to model server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(socket_timeout)
        sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        
        # Prepare message
        message = {
            "text": text,
            "voice": voice,
            "lang": lang,
            "speed": speed,
            "return_audio": True,  # Request direct audio return
            "first_time_voice": voice_downloaded
        }
        
        # Send JSON message
        SocketProtocol.send_json(sock, message)
        
        # Receive response
        response = SocketProtocol.receive_json(sock, timeout=socket_timeout)
        
        # Check for audio data
        if "audio" in response:
            return True, response["audio"], response.get("sample_rate", SAMPLE_RATE)
        else:
            error_msg = response.get("error", "No audio data returned")
            logger.error(f"Error getting audio: {error_msg}")
            return False, None, None
            
    except Exception as e:
        logger.error(f"Error getting direct audio: {e}")
        return False, None, None
    finally:
        try:
            sock.close()
        except:
            pass

def validate_voice_and_language(
    voice_input: str, 
    lang_input: str
) -> Tuple[str, str]:
    """
    Validate and convert voice and language inputs to their proper values.
    
    Args:
        voice_input: Voice name or number
        lang_input: Language code or number
        
    Returns:
        Tuple[str, str]: (voice_name, language_code)
        
    Raises:
        ValueError: If voice or language is invalid
    """
    voice_manager = VoiceManager()
    
    # Validate voice
    try:
        voice = get_voice_from_input(voice_input, voice_manager)
    except ValueError as e:
        raise ValueError(f"Invalid voice: {e}")
    
    # Validate language
    try:
        lang = get_language_from_input(lang_input, LANGUAGES)
    except ValueError as e:
        raise ValueError(f"Invalid language: {e}")
    
    return voice, lang

def kill_server() -> bool:
    """
    Send exit command to model server and ensure processes are terminated.
    
    Returns:
        bool: True if server was successfully terminated
    """
    try:
        # Try to connect to the model server
        if not is_model_server_running():
            logger.info("Model server not running. Nothing to kill.")
            return True
            
        # Connect to the server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SERVER_TIMEOUT)
        sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
            
        # Send properly formatted exit command
        try:
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
            if not is_model_server_running():
                logger.info("All processes terminated successfully")
                return True
        
        # If processes are still running, force kill them
        logger.warning("Processes still running after graceful shutdown attempt, forcing termination...")
        return force_kill_processes()
        
    except Exception as e:
        logger.error(f"Failed to send kill command: {e}")
        # Try forceful kill as fallback
        return force_kill_processes()

def check_processes_running() -> bool:
    """
    Check if any kokoro TTS processes are still running.
    
    Returns:
        bool: True if processes are running
    """
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

def force_kill_processes() -> bool:
    """
    Forcefully terminate all kokoro TTS processes.
    
    Returns:
        bool: True if all processes were terminated
    """
    try:
        import subprocess
        import signal
        import os
        import glob
        
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
        from src.constants import SOCKET_BASE_PATH
        socket_pattern = f"{SOCKET_BASE_PATH}_*"
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

# Audio file helper functions
def create_sample_wav_file(filename: str, duration_seconds: float = 2) -> bool:
    """
    Create a sample WAV file with an audible tone.
    
    Args:
        filename: Path to save the WAV file
        duration_seconds: Duration of the tone in seconds
        
    Returns:
        bool: True if file was successfully created
    """
    try:
        # Create a sine wave at 440Hz (A4 note)
        import numpy as np
        import soundfile as sf
        
        t = np.linspace(0., duration_seconds, int(SAMPLE_RATE * duration_seconds))
        amplitude = 0.3
        frequency = 440.0  # A4 note
        
        # Create a sine wave
        audio = amplitude * np.sin(2. * np.pi * frequency * t)
        
        # Apply fade in/out to avoid clicks
        fade_duration = 0.1  # seconds
        fade_samples = int(fade_duration * SAMPLE_RATE)
        
        # Apply fade in
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        
        # Apply fade out
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Save to WAV
        sf.write(filename, audio.astype(np.float32), SAMPLE_RATE)
        logger.info(f"Created sample WAV file with tone: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to create sample WAV file: {e}")
        return False

def create_dummy_wav_file(filename: str, duration_seconds: float = 1) -> bool:
    """
    Create a dummy WAV file with silence.
    
    Args:
        filename: Path to save the WAV file
        duration_seconds: Duration of silence in seconds
        
    Returns:
        bool: True if file was successfully created
    """
    try:
        # Create silence at sample rate
        import numpy as np
        import soundfile as sf
        
        audio = np.zeros(int(SAMPLE_RATE * duration_seconds), dtype=np.float32)
        sf.write(filename, audio, SAMPLE_RATE)
        logger.info(f"Created dummy WAV file: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to create dummy WAV file: {e}")
        return False
