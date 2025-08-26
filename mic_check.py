import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "test_mic.wav"

p = pyaudio.PyAudio()

# List available input devices
print("Available input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:  # Only show input devices
        print(f"{i}: {dev['name']}")

# Use default input device or specify one
device_index = int(input("Enter device number to use (default is usually 0): ") or 0)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

print("Recording for 5 seconds...")
frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]

stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded data
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Recording saved to {OUTPUT_FILENAME}")