#!/usr/bin/env python3

import os
import sys
import socket
import logging
import time
import psutil
from pathlib import Path
from typing import Optional, Tuple, List

from .constants import LANGUAGES
from .fetchers import ModelFetcher, VoiceFetcher

logger = logging.getLogger(__name__)

def ensure_model_and_voices(script_dir: str | os.PathLike) -> Tuple[str, str]:
    """Ensure model and voices exist, downloading if needed"""
    script_dir = os.path.expanduser(script_dir)
    model_path = os.path.join(script_dir, "kokoro-v0_19.onnx")
    voices_path = os.path.join(script_dir, "voices.json")
    
    try:
        model_fetcher = ModelFetcher()
        voice_fetcher = VoiceFetcher()
        
        model_fetcher.fetch_model(model_path)
        voice_fetcher.fetch_voices(voices_path)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise RuntimeError(f"Failed to ensure model and voices: {e}")
    
    return model_path, voices_path

def get_voice_from_input(voice_input: str, voices: List[str]) -> Optional[str]:
    """Convert voice input (name or number) to voice name"""
    if voice_input.isdigit():
        index = int(voice_input) - 1
        if 0 <= index < len(voices):
            return voices[index]
    elif voice_input in voices:
        return voice_input
    return None

def get_language_from_input(lang_input: str) -> Optional[str]:
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

def ensure_model_server_running() -> bool:
    """Ensure model server is running"""
    from .model_server import is_server_running, ensure_server_running
    return ensure_server_running()

def ensure_voice_server_running(voice: str, lang: str) -> bool:
    """Ensure voice server for given voice/lang is running"""
    from .voice_server import ensure_server_running
    return ensure_server_running(voice, lang)

def kill_all_daemons() -> bool:
    """Kill all TTS-related processes"""
    killed = False
    
    # Find and kill Python processes containing our server names
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.cmdline()
            if any(x in str(cmdline) for x in ['model_server.py', 'voice_server.py']):
                proc.terminate()
                killed = True
                logger.info(f"Terminated process {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return killed