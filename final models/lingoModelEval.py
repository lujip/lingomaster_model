import torch
import torchaudio
import torchaudio.transforms as T
import os
import torch.nn.functional as F
from lingo_model2 import SpeechModel

# Fixed input
audio_file = r'D:\Code research\AI Model\Lingomaster_Model\p226_p226_001.wav'
selected_word = 'phrase001'

# Transforms and VAD
transform = T.MFCC(sample_rate=16000, n_mfcc=13)
vad = T.Vad(sample_rate=16000)

def preprocess_waveform(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = vad(waveform) 
    return waveform

def predict(audio_input, model):
    waveform = preprocess_waveform(audio_input)
    mfcc = transform(waveform)
    mfcc = mfcc.squeeze(0).unsqueeze(0)

    if mfcc.abs().sum() == 0:
        print("⚠️ Empty or silent input detected. Skipping prediction.")
        return 0.0

    model.eval()
    with torch.no_grad():
        output = model(mfcc)
    return output.item()

def calculate_similarity(predicted_score, target_score=1.0):
    similarity = 100 * (1 - abs(predicted_score - target_score))
    return max(0, min(100, similarity))

def calculate_cosine_similarity(predicted_score, target_score):
    predicted_tensor = torch.tensor([predicted_score])
    target_tensor = torch.tensor([target_score])
    cosine_similarity = F.cosine_similarity(predicted_tensor.unsqueeze(0), target_tensor.unsqueeze(0))
    return (cosine_similarity.item() + 1) * 50

def calculate_scaled_similarity(predicted_score, min_score=0.0, max_score=1.0):
    normalized = (predicted_score - min_score) / (max_score - min_score)
    similarity = max(0, min(100, normalized * 100))
    return similarity

# Load model
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
model_path = os.path.join(base_dir, "model", f"{selected_word}_speech_model.pth")

if not os.path.exists(model_path):
    print(f"Model for '{selected_word}' not found at {model_path}")
else:
    model = SpeechModel(input_dim=13)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    predicted_score = predict(audio_file, model)
    similarity = calculate_similarity(predicted_score)
    cosine_sim = calculate_cosine_similarity(predicted_score, target_score=1)
    scale_sim = calculate_scaled_similarity(predicted_score)
    blended_sim = 0.8 * similarity + 0.2 * cosine_sim

    result = {
        "predicted_score": predicted_score,
        "similarity": similarity,
        "scaled_similarity": scale_sim,
        "blended_similarity": blended_sim,
    }

    print("Evaluation result:")
    for key, value in result.items():
        print(f"{key}: {value:.2f}")
