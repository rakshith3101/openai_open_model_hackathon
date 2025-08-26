"""
Test script for the VoiceModule.

This script demonstrates how to use the VoiceModule to listen for voice commands
and process them.
"""

import logging
import sys
import time
from voice_module.voice_module import VoiceModule

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# def command_callback(command: str) -> None:
#     """Handle recognized commands."""
#     print(f"\n🎤 Command received: {command}")
    
#     # Simple command processing
#     if "stop" in command or "exit" in command or "quit" in command:
#         print("Exiting...")
#         return False
    
#     return True

# def main():
#     """Run the voice module test."""
#     print("Voice Module Test")
#     print("================")
#     print(f"Say 'drone' followed by your command (e.g., 'drone take off')")
#     print("Say 'drone stop' to exit\n")
    
#     try:
#         # Create and start the voice module
#         with VoiceModule(
#             model_size="base",  # Use 'tiny' for faster but less accurate results
#             wake_word="drone",
#             silence_duration=1.0,
#             input_device_index=5
#         ) as voice_module:
            
#             # Keep the main thread alive
#             while True:
#                 try:
#                     # Process commands until we get a stop signal
#                     if not command_callback(input("\nWaiting for command... ")):
#                         break
#                 except KeyboardInterrupt:
#                     print("\nStopping...")
#                     break
#                 except Exception as e:
#                     print(f"Error: {e}")
#                     break
                
#     except KeyboardInterrupt:
#         print("\nExiting...")
#     except Exception as e:
#         print(f"Error: {e}")
#         return 1
    
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())
# In test_voice_module.py, replace the main() function with:

def main():
    """Run the voice module test."""
    print("Voice Module Test")
    print("================")
    print(f"Say 'drone' followed by your command (e.g., 'drone take off')")
    print(f"Say 'drone stop' to exit\n")
    
    def handle_command(command: str) -> bool:
        """Handle recognized commands."""
        print(f"\n🎤 Command received: {command}")
        
        if "stop" in command or "exit" in command or "quit" in command:
            print("Exiting...")
            return False
        
        return True

    try:
        # Create and start the voice module
        voice_module = VoiceModule(
            model_size="small",
            wake_word="drone",
            silence_threshold=0.02,  # Adjust this value
            silence_duration=1.0,    # Adjust this value
            input_device_index=1     # Try different indices (1, 5, or 9)
        )
        
        # Start the voice module with our command handler
        voice_module.start(handle_command)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        voice_module.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())