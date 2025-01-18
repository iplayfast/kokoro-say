#!/usr/bin/env python3
"""
Build script to create standalone executable
"""
import os
import sys
import io
import json
import requests
import torch
import urllib.request
import subprocess

def fetch_voices():
    """Fetch and process voice data"""
    voices = [
        "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael", "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis"
    ]
    voices_json = {}
    pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt"
    
    print("Fetching voices...")
    for voice in voices:
        print(f"Downloading {voice}...")
        r = requests.get(pattern.format(voice=voice))
        voice_data = torch.load(io.BytesIO(r.content)).numpy()
        voices_json[voice] = voice_data.tolist()
    
    with open("voices.json", "w") as f:
        json.dump(voices_json, f)

def main():
    # Prepare build directory
    os.makedirs("build", exist_ok=True)
    os.chdir("build")
    
    # Download model and voices
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "kokoro-v0_19.onnx"
    )
    fetch_voices()
    
    # Copy main script
    with open("../say.py", "r") as f:
        with open("say.py", "w") as out:
            out.write(f.read())
    
    # Create spec file
    spec = """
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(['say.py'],
    pathex=[],
    binaries=[],
    datas=[('kokoro-v0_19.onnx', '.'), ('voices.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='say',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    with open("say.spec", "w") as f:
        f.write(spec.strip())
    
    # Build executable
    print("Building executable...")
    subprocess.run(["pyinstaller", "--clean", "say.spec"])
    
    print("\nBuild complete! Executable is at: build/dist/say")

if __name__ == "__main__":
    main()
