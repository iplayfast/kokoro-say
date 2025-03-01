# Kokoro-Say Examples

This directory contains examples showing how to integrate the Kokoro-Say text-to-speech system with different applications and frameworks.

## Flask Web Application Example

The Flask example demonstrates how to create a simple web interface for the Kokoro-Say TTS system. This guide assumes no prior knowledge of Flask.

### What is Flask?

Flask is a lightweight web framework for Python. It allows you to create web applications that can serve HTML pages and handle HTTP requests. In this example, it acts as a bridge between your web browser and the Kokoro-Say TTS system.

### Prerequisites

Before running the example, you need:

1. Python 3.8 or higher installed
2. Kokoro-Say system installed and working
3. Required Python packages:
   ```
   flask
   requests
   ```

You can install the required packages with:
```bash
pip install flask requests
```

### Directory Structure

```
flask_app/
├── app.py            # The Flask application code
└── templates/        # Directory for HTML templates
    └── index.html    # The HTML interface
```

### Running the Example

#### Step 1: Start the Kokoro-Say API Server

First, make sure the Kokoro-Say API server is running:

```bash
# Navigate to the main kokoro-say directory
cd /path/to/kokoro-say

# Start the API server
python api.py
```

This starts the TTS API server on port 8000 by default. Wait until you see a message indicating that the server is running.

#### Step 2: Start the Flask Application

In a new terminal window:

```bash
# Navigate to the flask_app directory
cd /path/to/kokoro-say/doc/examples/flask_app

# Run the Flask application
python app.py
```

You should see output like:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

This means the Flask app is running on port 5000.

#### Step 3: Open the Web Interface

1. Open a web browser and go to: `http://localhost:5000`
2. You should see the Kokoro-Say web interface with:
   - A text input box
   - Voice selection dropdown
   - Language selection dropdown
   - Speed adjustment
   - Three buttons for different playback options

### Using the Interface

#### Text Input
Enter the text you want to convert to speech in the large text area.

#### Voice Selection
Choose from available voices in the dropdown menu.

#### Language Selection
Select the language for speech synthesis.

#### Speed Adjustment
Set a speed value between 0.5 (slower) and 2.0 (faster).

#### Available Actions

The interface offers three ways to use the TTS system:

1. **Speak (Client-side)** - This synthesizes speech and plays it in your browser. The audio is processed on the server but played on your computer.

2. **Download WAV** - This synthesizes speech and downloads the resulting audio as a WAV file that you can save and use later.

3. **Speak (Server-side)** - This synthesizes speech and plays it through the speakers of the computer running the server. No audio is sent to your browser.

### How It Works (No Flask Knowledge Required)

When you click a button:

1. Your browser sends the text and voice settings to the Flask app
2. The Flask app forwards the request to the Kokoro-Say API
3. The API processes the text and generates speech
4. Depending on which button you clicked:
   - Client-side: The audio file is sent back to your browser and played
   - Download: The audio file is sent to your browser for download
   - Server-side: The audio is played on the server's speakers

### Troubleshooting

- **No response from the application**: Make sure both the API server and Flask app are running
- **"Connection refused" error**: The API server isn't running or is on a different port
- **No audio plays**: Check your browser's audio settings and permissions
- **Server-side playback doesn't work**: Make sure the server has audio output configured

## Customizing the Example

You can modify the example to suit your needs:

### Changing the Port

If port 5000 is already in use, you can change it in `app.py`:

```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change 5000 to any available port
```

### Changing the API URL

If your Kokoro-Say API is running on a different host or port, change the `TTS_API_URL` in `app.py`:

```python
TTS_API_URL = "http://localhost:8000"  # Change to your API URL
```

### Adding More Voices or Languages

The example automatically fetches available voices and languages from the API. If you add more voices to the Kokoro-Say system, they will appear in the dropdown menus automatically.

## Creating Your Own Integration

This example shows the basic pattern for integrating with Kokoro-Say:

1. Send a request to the API with text and voice parameters
2. Receive the audio data or play it directly on the server
3. Process the audio as needed (play, save, etc.)

You can use this pattern to integrate Kokoro-Say with any application that can make HTTP requests, not just web applications.

## Other Examples

Check the other directories for examples using different frameworks or languages:

- `/node_app` - Example using Node.js and Express (if available)
- `/python_desktop` - Example desktop application using Python (if available)
