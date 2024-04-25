import torchaudio

torchaudio.set_audio_backend('soundfile')
try:
    waveform, sample_rate = torchaudio.load('../Data/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac')
    print("Audio loaded successfully:", waveform.shape, sample_rate)
except Exception as e:
    print("Failed to load audio:", e)
