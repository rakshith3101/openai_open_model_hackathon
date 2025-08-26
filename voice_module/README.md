# Voice Module for Drone Control

This module provides speech recognition capabilities for controlling a drone using natural language commands. It uses OpenAI's Whisper for speech-to-text conversion and includes wake word detection.

## Features

- Real-time speech recognition using Whisper
- Wake word detection (default: "drone")
- Configurable audio parameters
- Thread-safe operation
- Simple callback interface for command processing
- Logging and error handling

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Windows) Install PyAudio:
   ```bash
   pip install pyaudio
   ```
   If you encounter issues, you may need to install the Microsoft Visual C++ Build Tools.

## Usage

```python
from voice_module import VoiceModule

def handle_command(command: str) -> None:
    print(f"Command received: {command}")
    # Add your command processing logic here

# Create and start the voice module
with VoiceModule(wake_word="drone") as voice_module:
    voice_module.start(handle_command)
    
    # Keep the program running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
```

## Configuration

The `VoiceModule` accepts the following parameters:

- `model_size`: Whisper model size (tiny, base, small, medium, large)
- `wake_word`: Word to trigger command processing (default: "drone")
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `chunk_size`: Number of audio frames per buffer (default: 1024)
- `silence_threshold`: Amplitude threshold for silence detection (default: 0.03)
- `silence_duration`: Duration of silence to consider the end of speech (default: 1.5 seconds)

## Testing

Run the test script to try out the voice module:

```bash
python test_voice_module.py
```

Speak commands starting with the wake word (e.g., "drone take off").

## Integration with Drone Controller

To integrate with the drone controller, pass the recognized commands to your controller's command processor in the `handle_command` function.

## Troubleshooting

- If you get audio device errors, check your microphone settings
- Increase the `silence_duration` if commands are being cut off
- Use a smaller `model_size` for faster but less accurate recognition
- Ensure you have sufficient disk space for the Whisper model download
