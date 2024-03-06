import asyncio
import os
import tempfile
import torch
import librosa
import pyaudio
import wave
from flask import Flask, render_template, request

app = Flask(__name__, template_folder=".")

# Load VAD model and utilities
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    onnx=True,
)
_, _, _, VADIterator, _ = utils

VAD_SAMPLING_RATE = 8000
VAD_WINDOW_SIZE_EXAMPLES = 512

# Create VAD iterator
vad_iterator = VADIterator(vad_model, threshold=0.7, sampling_rate=VAD_SAMPLING_RATE)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/record", methods=["POST"])
def record():
    # Create a temporary directory to store the recorded audio
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define audio settings
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 44100
        chunk = 1024
        record_seconds = 5  # Adjust the recording duration as needed

        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        # Open audio stream
        stream = audio.open(format=audio_format,
                            channels=channels,
                            rate=rate,
                            input=True,
                            frames_per_buffer=chunk)

        print("Recording...")

        # Record audio
        frames = []
        for i in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print("Finished recording.")

        # Stop audio stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded audio to a WAV file
        temp_audio_file = os.path.join(temp_dir, "recorded_audio.wav")
        with wave.open(temp_audio_file, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        # Perform voice activity detection
        vad_result = perform_vad(temp_audio_file)

        # Return the result
        if vad_result:
            return "Voice activity detected in the recorded audio."
        else:
            return "No voice activity detected in the recorded audio."

def perform_vad(audio_file_path):
    # Load audio data
    audio_data, _ = librosa.load(audio_file_path, sr=None)

    # Resample to VAD input sampling rate
    audio_data_resampled = librosa.resample(audio_data, 44100, VAD_SAMPLING_RATE)

    # Process the buffer
    speech_detected = False
    for i in range(0, len(audio_data_resampled), VAD_WINDOW_SIZE_EXAMPLES):
        if i + VAD_WINDOW_SIZE_EXAMPLES > len(audio_data_resampled):
            break

        speech_dict = vad_iterator(audio_data_resampled[i: i + VAD_WINDOW_SIZE_EXAMPLES], return_seconds=True)
        if speech_dict:
            speech_detected = True
            break

    return speech_detected

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
