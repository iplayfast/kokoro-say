#!/usr/bin/env python3
"""
Example Flask application that integrates with the Kokoro TTS API.
This demonstrates how to use the TTS system in a web application.

Requirements:
- Flask
- requests

Run with:
    python app.py
"""

from flask import Flask, render_template, request, jsonify, send_file
import requests
import tempfile
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# TTS API configuration
TTS_API_URL = "http://localhost:8000"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Get available voices from the TTS API"""
    try:
        response = requests.get(f"{TTS_API_URL}/voices")
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get available languages from the TTS API"""
    try:
        response = requests.get(f"{TTS_API_URL}/languages")
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error fetching languages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak():
    """Generate speech and stream WAV file to client"""
    data = request.json
    
    # Validate input
    if 'text' not in data:
        return jsonify({"error": "Text is required"}), 400
    
    # Create temporary file for the WAV
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    try:
        logger.info(f"Synthesizing speech: {data['text'][:30]}...")
        
        # Call the TTS API
        response = requests.post(
            f"{TTS_API_URL}/synthesize-file",
            json={
                "text": data['text'],
                "voice": data.get('voice', 'af_bella'),
                "language": data.get('language', 'en-us'),
                "speed": float(data.get('speed', 1.0))
            },
            stream=True
        )
        
        response.raise_for_status()
        
        # Save the response content to the temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Speech synthesized successfully, file size: {os.path.getsize(temp_path)} bytes")
        
        # Send the WAV file to the client
        return send_file(
            temp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"speech_{int(time.time())}.wav"
        )
    
    except requests.exceptions.RequestException as e:
        logger.error(f"TTS API error: {e}")
        return jsonify({"error": f"TTS API error: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
    finally:
        # Clean up the temporary file (it will be removed after being sent)
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")

@app.route('/api/speak-server', methods=['POST'])
def speak_server():
    """Generate speech and play it on the server (no audio returned to client)"""
    data = request.json
    
    # Validate input
    if 'text' not in data:
        return jsonify({"error": "Text is required"}), 400
    
    try:
        logger.info(f"Synthesizing speech on server: {data['text'][:30]}...")
        
        # Call the TTS API
        response = requests.post(
            f"{TTS_API_URL}/synthesize",
            json={
                "text": data['text'],
                "voice": data.get('voice', 'af_bella'),
                "language": data.get('language', 'en-us'),
                "speed": float(data.get('speed', 1.0))
            }
        )
        
        response.raise_for_status()
        
        # Return success response
        return jsonify({
            "status": "success",
            "message": "Speech synthesized and playing on server"
        })
    
    except requests.exceptions.RequestException as e:
        logger.error(f"TTS API error: {e}")
        return jsonify({"error": f"TTS API error: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    # Check if TTS API is running
    try:
        health_check = requests.get(f"{TTS_API_URL}/health", timeout=2)
        health_check.raise_for_status()
        logger.info("TTS API is running and healthy")
    except Exception as e:
        logger.warning(f"TTS API may not be running: {e}")
        logger.warning(f"Make sure to start the TTS API with: python api.py")
    
    # Start Flask app
    app.run(debug=True, port=5000)
