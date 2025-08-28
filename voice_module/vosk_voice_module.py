"""
Voice Module for Drone Control using Vosk
Handles speech recognition using Vosk and provides a simple interface for voice commands.
"""

import json
import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
from dataclasses import dataclass
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    ERROR = auto()

@dataclass
class VoiceCommand:
    text: str
    confidence: float
    timestamp: float

class VoskVoiceModule:
    """
    Voice module that handles speech recognition using Vosk.
    
    Features:
    - Continuous listening with wake word detection
    - Real-time speech-to-text conversion
    - Callback system for command processing
    - Thread-safe operation
    """
    
    def __init__(
        self,
        model_path: str = "vosk-model-small-en-us",  # Path to the Vosk model
        wake_word: str = "drone",
        sample_rate: int = 16000,
        chunk_size: int = 512,  # Reduced for faster processing
        silence_threshold: float = 0.02,  # More sensitive
        silence_duration: float = 0.8,  # Shorter silence duration
        input_device_index: Optional[int] = None,
    ):
        """Initialize the voice module.
        
        Args:
            model_path: Path to the Vosk model directory
            wake_word: Word to trigger command processing
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of audio frames per buffer
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Duration of silence to consider end of speech (seconds)
            input_device_index: Index of input device to use (None for default)
        """
        self.model_path = model_path
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.input_device_index = input_device_index
        
        # State management
        self.state = VoiceState.IDLE
        self._stop_event = threading.Event()
        self._audio_queue = queue.Queue()
        self._command_callback = None
        
        # Initialize audio components
        self.audio_interface = None
        self.audio_stream = None
        self.model = None
        self.recognizer = None
        
        logger.info(f"VoskVoiceModule initialized")
        
    def _initialize_audio(self):
        """Initialize the audio interface and stream."""
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data."""
        if not self._stop_event.is_set():
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self._audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio data from the queue."""
        silence_frames = 0
        is_speaking = False
        buffer = []
        
        while not self._stop_event.is_set():
            if self.state == VoiceState.LISTENING:
                try:
                    audio_data = self._audio_queue.get(timeout=0.1)  # Reduced timeout for faster response
                    
                    # Convert float32 to int16 for Vosk
                    audio_data_int = (audio_data * 32768).astype(np.int16).tobytes()
                    
                    # Check partial results for faster response
                    if len(buffer) < 4:  # Buffer up to 4 chunks for better recognition
                        buffer.append(audio_data_int)
                        partial = self.recognizer.PartialResult()
                        if partial:
                            partial_result = json.loads(partial)
                            if partial_result.get("partial"):
                                partial_text = partial_result["partial"].lower()
                                if self.wake_word in partial_text:
                                    is_speaking = True
                    else:
                        # Process buffered audio
                        audio_chunk = b"".join(buffer)
                        buffer = []
                        
                        if self.recognizer.AcceptWaveform(audio_chunk):
                            result = json.loads(self.recognizer.Result())
                            
                            if result.get("text"):
                                text = result["text"].lower()
                                
                                if self.wake_word in text:
                                    # Remove wake word and process command
                                    command = text.replace(self.wake_word, "").strip()
                                    if command:  # Only process if there's text after wake word
                                        self._handle_command(command)
                                        is_speaking = False
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    self.state = VoiceState.ERROR
    
    def _handle_command(self, command: str):
        """Process recognized command."""
        if self._command_callback:
            timestamp = time.time()
            voice_command = VoiceCommand(
                text=command,
                confidence=1.0,  # Vosk doesn't provide confidence scores directly
                timestamp=timestamp
            )
            self._command_callback(voice_command)
    
    def start(self, command_callback: Callable[[VoiceCommand], None]) -> None:
        """Start the voice module.
        
        Args:
            command_callback: Function to call when a command is recognized
        """
        if self.state != VoiceState.IDLE:
            logger.warning("Voice module is already running")
            return
            
        try:
            # Load Vosk model
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            self._command_callback = command_callback
            self._stop_event.clear()
            
            # Initialize audio
            self._initialize_audio()
            
            # Start processing thread
            self.state = VoiceState.LISTENING
            self._process_thread = threading.Thread(target=self._process_audio)
            self._process_thread.start()
            
            # Start the audio stream
            self.audio_stream.start_stream()
            
            logger.info("Voice module started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start voice module: {e}")
            self.state = VoiceState.ERROR
            raise
    
    def stop(self) -> None:
        """Stop the voice module."""
        if self.state == VoiceState.IDLE:
            return
            
        self._stop_event.set()
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
        if self.audio_interface:
            self.audio_interface.terminate()
            
        if hasattr(self, '_process_thread'):
            self._process_thread.join()
            
        self.state = VoiceState.IDLE
        logger.info("Voice module stopped")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
