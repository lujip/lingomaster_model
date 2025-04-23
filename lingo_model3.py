import torch
import torchaudio
import torchaudio.transforms as T
from lingo_model2 import SpeechModel
import os
import torch.nn.functional as F


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

def predict(audio_input):
    waveform, sample_rate = torchaudio.load(audio_input)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    transform = T.MFCC(sample_rate=16000, n_mfcc=13)
    mfcc = transform(waveform) 

   
    mfcc = mfcc.squeeze(0).unsqueeze(0) 

    
    model.eval()
    with torch.no_grad():
        output = model(mfcc)

    return output.item()



def calculate_similarity(predicted_score, target_score=1.0):

    similarity = 100 * (1 - abs(predicted_score - target_score))  
    similarity = max(0, min(100, similarity))  
    return similarity

def calculate_cosine_similarity(predicted_score, target_score):
    predicted_tensor = torch.tensor([predicted_score])
    target_tensor = torch.tensor([target_score])
    cosine_similarity = F.cosine_similarity(predicted_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
    similarity_percentage = (cosine_similarity.item() + 1) * 50  
    return similarity_percentage

def calculate_scaled_similarity(predicted_score, min_score=0.0, max_score=1.0):
    normalized = (predicted_score - min_score) / (max_score - min_score)
    similarity = max(0, min(100, normalized * 100))
    return similarity



#audio_input = "Recording (5).wav" 
#audio_input = "p225_p225_001.wav"
#audio_input = "p226_p226_001.wav"
#audio_input = "p229_p229_012.wav"
predicted_score = predict(audio_input)
raw_sim = calculate_similarity(predicted_score)

print(f"Raw predicted score: {predicted_score}")
print(f"Predicted similarity score: {raw_sim:.2f}%")

cos_sim = calculate_cosine_similarity(predicted_score, target_score=1)
print(f"Predicted similarity (cosine): {cos_sim:.2f}%")

similarity = calculate_scaled_similarity(predicted_score)
print(f"Predicted similarity (scaled): {similarity:.2f}%")
blended_sim = (raw_sim + cos_sim) / 2
print(f"Predicted similarity (blended): {blended_sim:.2f}%")
