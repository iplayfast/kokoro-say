#!/usr/bin/env python3

import os
import sys
import json
import time
import socket
import logging
import argparse
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import uvicorn

# Import constants at the top
from src.constants import (
    SAMPLE_RATE,
    SERVER_TIMEOUT,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    LOG_FILE,
    LANGUAGES
)
from src.utils import VoiceManager, get_voice_from_input, get_language_from_input
from src.fetchers import ensure_model_and_voices, VoiceFetcher

# Configure logging
logging.basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Kokoro TTS API",
    description="API for Kokoro Text-to-Speech synthesis",
    version="1.0.0"
)

# Create temporary directory for WAV files
temp_dir = Path("/tmp/kokoro_api")
temp_dir.mkdir(parents=True, exist_ok=True)

# Models for API
class SynthesisRequest(BaseModel):
    text: str
    voice: str = "af_bella"
    language: str = "en-us"
    speed: float = 1.0

# Global variables
model_server_running = False

def ensure_model_server() -> bool:
    """Ensure model server is running, start if needed."""
    global model_server_running
    
    # Check if server is already running
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SERVER_TIMEOUT)
    result = sock.connect_ex((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
    sock.close()
    
    if result == 0:
        # Server is running
        model_server_running = True
        return True
        
    # Start server using say.py's functionality
    from say import ensure_model_server as start_model_server
    if start_model_server():
        model_server_running = True
        return True
    
    return False

def send_to_model_server(request: SynthesisRequest) -> Dict:
    """Send synthesis request to model server and get audio data."""
    try:
        # Connect to model server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        
        # Prepare and send message
        message = json.dumps({
            "text": request.text,
            "voice": request.voice,
            "lang": request.language,
            "speed": request.speed
        }).encode()
        
        logger.debug(f"Sending synthesis request to model server")
        sock.sendall(message)
        
        # Receive response with timeout
        sock.settimeout(30.0)  # Longer timeout for synthesis
        
        # Receive all data in chunks
        chunks = []
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
            
        response_data = b''.join(chunks)
        sock.close()
        
        if not response_data:
            raise RuntimeError("No response received from model server")
            
        # Parse response
        response = json.loads(response_data.decode())
        if response.get("status") == "error":
            raise RuntimeError(f"Model server error: {response.get('error')}")
            
        return response
        
    except Exception as e:
        logger.error(f"Failed to communicate with model server: {e}")
        raise RuntimeError(f"Model server communication error: {str(e)}")

def play_audio(audio_data: List[float], sample_rate: int = SAMPLE_RATE):
    """Play audio data through sounddevice."""
    try:
        audio_array = np.array(audio_data, dtype=np.float32)
        sd.play(audio_array, sample_rate)
        sd.wait()
    except Exception as e:
        logger.error(f"Failed to play audio: {e}")
        raise RuntimeError(f"Audio playback error: {str(e)}")

def save_audio_to_file(audio_data: List[float], sample_rate: int = SAMPLE_RATE) -> Path:
    """Save audio data to WAV file and return the path."""
    try:
        # Create unique filename based on timestamp
        filename = f"tts_{int(time.time())}_{os.getpid()}.wav"
        file_path = temp_dir / filename
        
        # Convert to numpy array and save
        audio_array = np.array(audio_data, dtype=np.float32)
        sf.write(file_path, audio_array, sample_rate)
        
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save audio to file: {e}")
        raise RuntimeError(f"Audio file creation error: {str(e)}")

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "server": "Kokoro TTS API"}

@app.get("/")
async def get_info():
    """Get system information including available voices and languages."""
    try:
        voice_manager = VoiceManager()
        
        return {
            "status": "ok",
            "voices": {
                "american": sorted(list(voice_manager.american_voices)),
                "british": sorted(list(voice_manager.british_voices))
            },
            "languages": LANGUAGES
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/voices")
async def get_voices():
    """Get list of available voices."""
    try:
        voice_manager = VoiceManager()
        voices = {
            "american": {
                voice: voice_manager.get_voice_info(voice) 
                for voice in sorted(voice_manager.american_voices)
            },
            "british": {
                voice: voice_manager.get_voice_info(voice) 
                for voice in sorted(voice_manager.british_voices)
            }
        }
        return voices
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/languages")
async def get_languages():
    """Get list of available languages."""
    return {code: name for code, name in LANGUAGES}

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """Synthesize speech and play it directly."""
    try:
        # Ensure model server is running
        if not ensure_model_server():
            raise HTTPException(status_code=500, detail="Failed to start model server")
            
        # Validate voice and language
        voice_manager = VoiceManager()
        try:
            voice = get_voice_from_input(request.voice, voice_manager)
            lang = get_language_from_input(request.language, LANGUAGES)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        # Download voice if needed
        voice_manager.ensure_voice_available(voice)
        
        # Send request to model server
        response = send_to_model_server(SynthesisRequest(
            text=request.text,
            voice=voice,
            language=lang,
            speed=request.speed
        ))
        
        # Play audio in background task
        audio_data = response["audio"]
        sample_rate = response.get("sample_rate", SAMPLE_RATE)
        
        background_tasks.add_task(play_audio, audio_data, sample_rate)
        
        return {"status": "success", "message": "Speech synthesized and playing"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")

@app.post("/synthesize-file")
async def synthesize_to_file(request: SynthesisRequest):
    """Synthesize speech and return WAV file."""
    try:
        # Ensure model server is running
        if not ensure_model_server():
            raise HTTPException(status_code=500, detail="Failed to start model server")
            
        # Validate voice and language
        voice_manager = VoiceManager()
        try:
            voice = get_voice_from_input(request.voice, voice_manager)
            lang = get_language_from_input(request.language, LANGUAGES)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        # Download voice if needed
        voice_manager.ensure_voice_available(voice)
        
        # Send request to model server
        response = send_to_model_server(SynthesisRequest(
            text=request.text,
            voice=voice,
            language=lang,
            speed=request.speed
        ))
        
        # Save audio to file
        audio_data = response["audio"]
        sample_rate = response.get("sample_rate", SAMPLE_RATE)
        file_path = save_audio_to_file(audio_data, sample_rate)
        
        # Return file response
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis to file error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Ensure voices directory exists
    voice_manager = VoiceManager()
    
    print(f"Starting Kokoro TTS API server on {args.host}:{args.port}")
    uvicorn.run(
        "api:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
