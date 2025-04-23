import torch
import torchaudio
from torch import nn
from model2 import SpeechModel  # Import the model class from your training script
import sys

# Load the trained model
def load_model(model_path, input_dim=13):
    model = SpeechModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Extract MFCC from input audio
def extract_mfcc(file_path, sample_rate=16000, n_mfcc=13):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = transform(waveform).squeeze(0)  # shape: [n_mfcc, time]
    
    # Pad or truncate to fixed length
    fixed_length = 300
    if mfcc.shape[1] > fixed_length:
        mfcc = mfcc[:, :fixed_length]
    else:
        pad_amount = fixed_length - mfcc.shape[1]
        mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))

    return mfcc.unsqueeze(0)  # add batch dimension

# Evaluate the input audio file
def evaluate(model, mfcc_tensor):
    with torch.no_grad():
        output = model(mfcc_tensor)
        score = output.item()
        return score

# Main function
if __name__ == "__main__":
    # Example usage: python evaluate_audio.py sample.wav
    if len(sys.argv) < 2:
        print("Usage: python evaluate_audio.py <audio_file.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = "mfcc_model.pth"

    model = load_model(model_path)
    mfcc_tensor = extract_mfcc(audio_path)
    score = evaluate(model, mfcc_tensor)

    print(f"Pronunciation Score: {score:.4f}")
