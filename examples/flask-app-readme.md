# Kokoro-Say Flask Web Interface

This is a simple web interface for the Kokoro-Say text-to-speech system built using Flask. This guide explains how the application works and how to use it, even if you've never used Flask before.

## What This Application Does

This web application provides a user-friendly interface for the Kokoro-Say text-to-speech system. It allows you to:

1. Enter text to be converted to speech
2. Select voice, language, and speed options
3. Play audio directly in your browser
4. Download audio as a WAV file
5. Play audio on the server's speakers

## Files Explained

### app.py

This is the main Python file that runs the web application. It does several things:

1. **Creates a web server**: Uses Flask to make a simple web server
2. **Defines routes**: Creates URLs that your browser can access
3. **Handles requests**: Processes form submissions from the web interface
4. **Communicates with the TTS API**: Sends your text to Kokoro-Say and gets audio back

Key parts of this file:

```python
# This creates the web application
app = Flask(__name__)

# This creates a URL at http://localhost:5000/
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

# This creates a URL that handles speech synthesis requests
@app.route('/api/speak', methods=['POST'])
def speak():
    """Generate speech and stream WAV file to client"""
    # Get the data from the form
    data = request.json
    
    # Send the data to the Kokoro-Say API
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
    
    # Send the audio file back to the browser
    return send_file(
        temp_path,
        mimetype='audio/wav',
        as_attachment=True,
        download_name=f"speech_{int(time.time())}.wav"
    )

# This starts the web server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### templates/index.html

This is the HTML file that creates the user interface you see in your browser. It includes:

1. **Form fields**: Text area, dropdowns, and number input for options
2. **Buttons**: For different types of speech synthesis
3. **JavaScript**: Code that runs in your browser to send requests and handle responses

The JavaScript in this file is what makes the interface interactive. When you click a button, it:

1. Collects the text and options you've entered
2. Sends them to the Flask application
3. Receives the audio and either plays it or downloads it

## How the Application Works

When you run `python app.py`, Flask starts a web server on your computer at port 5000. When you open `http://localhost:5000` in your browser, this is what happens:

1. Your browser requests the page from Flask
2. Flask sends back the HTML from `index.html`
3. Your browser displays the page

When you click a button on the page:

1. JavaScript in your browser collects your text and options
2. It sends this data to one of the Flask application's routes (URLs)
3. The Flask application receives the data
4. The Flask application forwards it to the Kokoro-Say API
5. Kokoro-Say generates the speech
6. The Flask application receives the audio
7. The Flask application sends the audio back to your browser
8. Your browser either plays the audio or downloads it

The "Speak (Server-side)" button works a bit differently:

1. The Flask application still forwards your request to Kokoro-Say
2. But instead of sending audio back to your browser, it plays it on the server
3. Your browser only receives a success message

## Running Without Flask Knowledge

To run this application, you don't need to understand Flask. Just follow these steps:

1. Make sure the Kokoro-Say API is running:
   ```bash
   # In the main kokoro-say directory
   python api.py
   ```

2. Run the Flask application:
   ```bash
   # In the flask_app directory
   python app.py
   ```

3. Open `http://localhost:5000` in your web browser

That's it! You're using Flask without needing to understand how it works internally.

## Common Questions

### What is localhost?

"localhost" refers to your own computer. When you access `http://localhost:5000`, you're connecting to a web server running on your own machine.

### What is port 5000?

A port is like a door into your computer. The Flask application opens "door number 5000" to listen for web requests. The Kokoro-Say API uses port 8000 by default.

### Why two servers?

The system uses two servers for separation of concerns:

1. The **Kokoro-Say API** (port 8000) handles the actual text-to-speech conversion
2. The **Flask application** (port 5000) provides a user-friendly web interface

This separation makes the system more flexible. You could replace the Flask interface with something else, and the Kokoro-Say API would still work.

### What if I want to change something?

- To change how the page looks: Edit `templates/index.html`
- To change how requests are processed: Edit `app.py`
- To change the port: Edit the `app.run(debug=True, port=5000)` line in `app.py`

## Troubleshooting

- **"No module named flask"**: Run `pip install flask requests`
- **"Connection refused"**: Make sure the Kokoro-Say API is running
- **"Address already in use"**: Something else is using port 5000. Change the port in `app.py`
- **Page doesn't load**: Make sure the Flask app is running and check the console for errors

## Next Steps

Once you're comfortable with this example, you might want to:

1. Add more features to the web interface
2. Create a more sophisticated application
3. Integrate Kokoro-Say with other systems

The basic pattern of sending requests to the API and handling responses will work with any programming language or framework that can make HTTP requests.
