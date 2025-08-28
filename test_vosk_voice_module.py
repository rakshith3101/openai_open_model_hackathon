"""
Test script for the VoskVoiceModule.
"""

import logging
import sys
import time
from voice_module.vosk_voice_module import VoskVoiceModule, VoiceCommand, VoiceState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def handle_command(command: VoiceCommand) -> bool:
    """Handle recognized commands."""
    print(f"\n🎤 Command received: {command.text} (confidence: {command.confidence:.2f})")
    
    # Simple command processing
    if "stop" in command.text or "exit" in command.text or "quit" in command.text:
        print("Exiting...")
        return False
    
    return True

def main():
    """Run the voice module test."""
    print("Vosk Voice Module Test")
    print("=====================")
    print("Say 'drone' followed by your command (e.g., 'drone take off')")
    print("Say 'drone stop' to exit\n")
    
    try:
        # Create and start the voice module
        with VoskVoiceModule(
            model_path=r"C:\Users\Rakshith\Documents\ai-workspace\openai_open_model_hackathon\vosk-model-small-en-us",  # You'll need to download this model
            wake_word="drone",
            silence_duration=1.0
        ) as voice_module:
            voice_module.start(handle_command)
            
            # Keep the main thread alive
            while True:
                if voice_module.state == VoiceState.ERROR:
                    print("Error occurred in voice module. Exiting...")
                    break
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nStopping voice module...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
