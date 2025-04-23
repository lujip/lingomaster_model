import torch
import torchaudio
import torchaudio.transforms as T
from lingo_model2 import SpeechModel 

model_path = "model/phrase001_speech_model.pth"  
model = SpeechModel(input_dim=13) 
model.load_state_dict(torch.load(model_path))
model.eval() 


def preprocess_audio(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True) 


    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = mfcc_transform(waveform) 
    return mfcc.squeeze(0)  


def predict(model, mfcc):
    mfcc_tensor = mfcc.unsqueeze(0)  
    output = model(mfcc_tensor) 
    score = output.item() * 100 
    return score


#audio_input = "Recording (3).wav" 
audio_input = "p225_p225_001.py" 


mfcc = preprocess_audio(audio_input)
predicted_score = predict(model, mfcc)

print(f"Predicted pronunciation score: {predicted_score:.2f}%")
