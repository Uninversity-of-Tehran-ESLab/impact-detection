import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 5000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for impact sounds...")

try:
    while True:
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        peak_amplitude = np.abs(audio_data).max()

        if peak_amplitude > THRESHOLD:
            print("Impact detected! Amplitude:", peak_amplitude)
            # TODO: Add code to mark video frame

except KeyboardInterrupt:
    print("Stopping detection.")

finally:
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
