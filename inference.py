import sounddevice as sd
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from utils import *
import time  # Import time module for latency measurements
import argparse  # Import argparse for command line arguments
from torch.cuda.amp import autocast, GradScaler
from model import ConformerEncoder, LSTMDecoder

parser = argparse.ArgumentParser("conformer")

parser.add_argument('--gpu', type=int, default=0, help='gpu device id (optional)')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--report_freq', type=int, default=100, help='training objective report frequency')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='../model/model_best.pt', help='path to save the model')
parser.add_argument('--use_amp', action='store_true', default=False, help='use mixed precision to train')
parser.add_argument('--attention_heads', type=int, default=4, help='number of heads to use for multi-head attention')
parser.add_argument('--d_input', type=int, default=80, help='dimension of the input (num filter banks)')
parser.add_argument('--d_encoder', type=int, default=144, help='dimension of the encoder')
parser.add_argument('--d_decoder', type=int, default=320, help='dimension of the decoder')
parser.add_argument('--encoder_layers', type=int, default=16, help='number of conformer blocks in the encoder')
parser.add_argument('--decoder_layers', type=int, default=1, help='number of decoder layers')
parser.add_argument('--conv_kernel_size', type=int, default=31, help='size of kernel for conformer convolution blocks')
parser.add_argument('--feed_forward_expansion_factor', type=int, default=4, help='expansion factor for conformer feed forward blocks')
parser.add_argument('--feed_forward_residual_factor', type=int, default=.5, help='residual factor for conformer feed forward blocks')
parser.add_argument('--dropout', type=float, default=.1, help='dropout factor for conformer model')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='model weight decay (corresponds to L2 regularization)')
parser.add_argument('--variational_noise_std', type=float, default=.0001, help='std of noise added to model weights for regularization')


args = parser.parse_args()


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
    waveform = torch.tensor(waveform).unsqueeze(0)  # Convert to tensor and add batch dimension
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)

    # Normalize the spectrogram
    spectrogram = transform(waveform).squeeze(0).transpose(0, 1)

    # Pad the spectrogram to have shape (1, 1, T, F)
    spectrogram = nn.utils.rnn.pad_sequence(spectrogram, batch_first=True)

    elapsed_time = time.time() - start_time
    print(f"Spectrogram conversion time: {elapsed_time:.2f} seconds")
    return spectrogram.squeeze(0)  # Remove the batch dimension

def load_checkpoint(checkpoint_path, encoder, decoder, optimizer=None, scheduler=None):
    # Load the saved checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Restore the model and optimizer states
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['valid_loss']

def model_inference(spectrogram,char_decoder,gpu=True):
    """Perform inference using the trained model."""
    start_time = time.time()  # Start time for inference
    if gpu:
        spectrogram = spectrogram.cuda()

    text_transform = TextTransform()
    with torch.no_grad():
        outputs = encoder(spectrogram.unsqueeze(0))
        outputs = decoder(outputs)
        inds = char_decoder(outputs)
        predictions = text_transform.int_to_text(inds[0])  # Assuming single output
    elapsed_time = time.time() - start_time
    print(f"Model inference time: {elapsed_time:.2f} seconds")
    return predictions

# Set the device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Initialize your model components
encoder = ConformerEncoder(
                      d_input=args.d_input,
                      d_model=args.d_encoder,
                      num_layers=args.encoder_layers,
                      conv_kernel_size=args.conv_kernel_size, 
                      dropout=args.dropout,
                      feed_forward_residual_factor=args.feed_forward_residual_factor,
                      feed_forward_expansion_factor=args.feed_forward_expansion_factor,
                      num_heads=args.attention_heads)

decoder = LSTMDecoder(
                  d_encoder=args.d_encoder, 
                  d_decoder=args.d_decoder, 
                  num_layers=args.decoder_layers)
char_decoder = GreedyCharacterDecoder().eval()

# Load the trained model weights
epoch, valid_loss = load_checkpoint(args.model_path, encoder, decoder)
encoder = encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
decoder = decoder.to('cuda' if torch.cuda.is_available() else 'cpu')
encoder.eval()
decoder.eval()
# Record and process audio
duration = 5  # Duration to record in seconds
sample_rate = 16000  # Sampling rate
start_time = time.time()  # Start time for the entire process
audio = record_audio(duration, sample_rate)
spectrogram = audio_to_spectrogram(audio, sample_rate)

# Perform inference
predicted_text = model_inference(spectrogram, char_decoder)
total_elapsed_time = time.time() - start_time
print("Predicted Text:", predicted_text)
print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
