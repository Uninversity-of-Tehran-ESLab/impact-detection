# Internal
from datetime import datetime

# External
import pyaudio
import numpy as np


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def impact_sound_detection(threshold):
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

            if peak_amplitude > threshold:
                print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Impact detected! Amplitude:", peak_amplitude)

    except KeyboardInterrupt:
        print("Stopping detection.")

    finally:
        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
