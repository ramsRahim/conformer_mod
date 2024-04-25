import sounddevice as sd
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from utils import *
import time  # Import time module for latency measurements

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone."""
    start_time = time.time()  # Start time for recording
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    elapsed_time = time.time() - start_time
    print(f"Recording time: {elapsed_time:.2f} seconds")
    return audio.flatten()  # Return a 1D numpy array

def audio_to_spectrogram(waveform, sample_rate=16000):
    """Convert audio waveform to spectrogram."""
    start_time = time.time()  # Start time for conversion
    waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension
    transform = MelSpectrogram(sample_rate=sample_rate, n_mels=80, hop_length=160)
    spectrogram = transform(waveform)
    elapsed_time = time.time() - start_time
    print(f"Spectrogram conversion time: {elapsed_time:.2f} seconds")
    return spectrogram

def model_inference(spectrogram, encoder, decoder, char_decoder, gpu=True):
    """Perform inference using the trained model."""
    start_time = time.time()  # Start time for inference
    if gpu:
        spectrogram = spectrogram.cuda()

    encoder.eval()
    decoder.eval()
    text_transform = TextTransform()
    with torch.no_grad():
        outputs = encoder(spectrogram.unsqueeze(0))  # Add batch dimension
        outputs = decoder(outputs)
        inds = char_decoder(outputs)
        predictions = text_transform.int_to_text(inds[0])  # Assuming single output
    elapsed_time = time.time() - start_time
    print(f"Model inference time: {elapsed_time:.2f} seconds")
    return predictions

# Initialize your model components
encoder = ...  # Your encoder model
decoder = ...  # Your decoder model
char_decoder = ...  # Your character decoder function

# Record and process audio
duration = 5  # Duration to record in seconds
sample_rate = 16000  # Sampling rate
start_time = time.time()  # Start time for the entire process
audio = record_audio(duration, sample_rate)
spectrogram = audio_to_spectrogram(audio, sample_rate)

# Perform inference
predicted_text = model_inference(spectrogram, encoder, decoder, char_decoder, gpu=True)
total_elapsed_time = time.time() - start_time
print("Predicted Text:", predicted_text)
print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
