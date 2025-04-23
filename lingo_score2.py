import torch
import torchaudio
import torchaudio.transforms as T
from lingo_model2 import SpeechModel
from torch import nn
import os

model = SpeechModel(input_dim=13)  
model_path = os.path.join("model", "phrase001_speech_model.pth")  
model.load_state_dict(torch.load(model_path))
model.eval()


transform = T.MFCC(sample_rate=16000, n_mfcc=13)


def extract_mfcc(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    mfcc = transform(waveform).squeeze(0)
    mfcc = mfcc[:, :300] if mfcc.size(1) > 300 else torch.nn.functional.pad(mfcc, (0, 300 - mfcc.size(1)))
    return mfcc.unsqueeze(0) 

def predict(audio_path):
    mfcc = extract_mfcc(audio_path)
    with torch.no_grad():
        output = model(mfcc)  
        prediction = output.item()  
    return prediction

audio_input = "p225_p225_001.wav"  
predicted_label = predict(audio_input)
print(f"Predicted label: {predicted_label}")
