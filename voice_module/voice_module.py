"""
Voice Module for Drone Control
Handles speech recognition using Whisper and provides a simple interface for voice commands.
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import pyaudio
import torch
import whisper
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

class VoiceModule:
    """
    Voice module that handles speech recognition using Whisper.
    
    Features:
    - Continuous listening with wake word detection
    - Real-time speech-to-text conversion
    - Callback system for command processing
    - Thread-safe operation
    """
    
    def __init__(
        self,
        model_size: str = "medium",
        wake_word: str = "drone",
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        silence_threshold: float = 0.03,
        silence_duration: float = 1.5,
        input_device_index: int = 17,
    ):
        """Initialize the voice module.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            wake_word: Word to trigger command processing
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of audio frames per buffer
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Duration of silence to consider the end of speech (seconds)
        """
        self.model_size = model_size
        self.wake_word = wake_word.lower()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        # State management
        self.state = VoiceState.IDLE
        self._stop_event = threading.Event()
        self._audio_queue = queue.Queue()
        self._command_callback = None
        
        # Initialize in start() to handle threading properly
        self.audio_interface = None
        self.audio_stream = None
        self.model = None
        
        logger.info(f"VoiceModule initialized with model: {model_size}")
    
    def start(self, command_callback: Callable[[str], None]) -> None:
        """Start the voice module.
        
        Args:
            command_callback: Function to call when a command is recognized
        """
        if self.state != VoiceState.IDLE:
            logger.warning("Voice module is already running")
            return
            
        self._command_callback = command_callback
        self._stop_event.clear()
        
        try:
            # Initialize Whisper model
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model(self.model_size)
            
            # Initialize audio interface
            self.audio_interface = pyaudio.PyAudio()
            
            # Start the processing thread
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start the recording thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self._set_state(VoiceState.LISTENING)
            logger.info("Voice module started")
            
        except Exception as e:
            self._set_state(VoiceState.ERROR)
            logger.error(f"Failed to start voice module: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the voice module and clean up resources."""
        if self.state == VoiceState.IDLE:
            return
            
        logger.info("Stopping voice module...")
        self._stop_event.set()
        
        # Wait for threads to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        
        # Clean up audio resources
        if hasattr(self, 'audio_stream') and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'audio_interface') and self.audio_interface:
            self.audio_interface.terminate()
        
        self._set_state(VoiceState.IDLE)
        logger.info("Voice module stopped")
    
    def _set_state(self, new_state: VoiceState) -> None:
        """Update the module state and log the change."""
        old_state = self.state
        self.state = new_state
        logger.debug(f"State changed: {old_state.name} -> {new_state.name}")
    
    def _record_audio(self) -> None:
        """Record audio from the microphone and add it to the processing queue."""
        try:
            self.audio_stream = self.audio_interface.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            logger.info("Audio recording started")
            
            # Keep the thread alive until stopped
            while not self._stop_event.is_set():
                time.sleep(0.1)
                
        except Exception as e:
            self._set_state(VoiceState.ERROR)
            logger.error(f"Audio recording error: {e}")
        finally:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            logger.info("Audio recording stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to processing queue if not empty
        if np.abs(audio_data).mean() > self.silence_threshold:
            self._audio_queue.put(audio_data)
        
        return (None, pyaudio.paContinue)
    
    def _process_audio(self) -> None:
        """Process audio chunks from the queue and detect commands."""
        audio_buffer = []
        last_audio_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Get audio chunk with timeout
                try:
                    chunk = self._audio_queue.get(timeout=0.5)
                    audio_buffer.append(chunk)
                    last_audio_time = time.time()
                except queue.Empty:
                    # Check if we've had enough silence to process the audio
                    if audio_buffer and (time.time() - last_audio_time) > self.silence_duration:
                        self._process_audio_buffer(audio_buffer)
                        audio_buffer = []
                    continue
                
                # Process audio if buffer is large enough
                if len(audio_buffer) > 5:  # ~0.3s of audio
                    self._process_audio_buffer(audio_buffer)
                    audio_buffer = []
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                audio_buffer = []
    
    def _process_audio_buffer(self, audio_chunks: list) -> None:
        """Process a buffer of audio chunks and detect commands."""
        if not audio_chunks:
            return
            
        try:
            self._set_state(VoiceState.PROCESSING)
            
            # Convert chunks to single numpy array
            audio_data = np.concatenate(audio_chunks)
            
            # Transcribe audio using Whisper
            result = self.model.transcribe(
                audio_data,
                fp16=torch.cuda.is_available(),
                language="en"
            )
            
            text = result["text"].strip().lower()
            if not text:
                logger.debug("No speech detected")
                return
                
            logger.info(f"Recognized: {text}")
            
            # Check for wake word
            if self.wake_word in text:
                # Remove wake word and any leading/trailing whitespace
                command = text.replace(self.wake_word, "", 1).strip()
                if command and self._command_callback:
                    self._command_callback(command)
            
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
        finally:
            self._set_state(VoiceState.LISTENING)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure resources are cleaned up."""
        self.stop()


def create_voice_module(model_size: str = "base", **kwargs) -> VoiceModule:
    """Create and return a configured VoiceModule instance."""
    return VoiceModule(model_size=model_size, **kwargs)
