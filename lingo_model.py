import os
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import random
import librosa.effects
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
import soundfile as sf

PHRASE_NAME = "phrase001" 
DATASET_PATH = f"D:/Code research/Dataset-Sorted/{PHRASE_NAME}/"
MODEL_SAVE_PATH = f"models/{PHRASE_NAME}.pth"


def time_stretch(audio, rate=1.1):
    return librosa.effects.time_stretch(audio, rate)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    audio += noise_factor * noise
    return audio

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def extract_features(file_path, max_len=50, augment=False):
    waveform, sample_rate = torchaudio.load(file_path)
    audio_numpy = waveform.numpy().flatten()

    trimmed_audio, _ = librosa.effects.trim(audio_numpy, top_db=20)

    if augment:
        if random.random() < 0.5:
            trimmed_audio = pitch_shift(trimmed_audio, sample_rate) 
        if random.random() < 0.5:
            trimmed_audio = time_stretch(trimmed_audio)
        if random.random() < 0.5:
            trimmed_audio = add_noise(trimmed_audio)


    trimmed_audio = trimmed_audio / np.max(np.abs(trimmed_audio))

    mfcc = librosa.feature.mfcc(y=trimmed_audio, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=trimmed_audio, sr=sample_rate)
    pitch, _ = librosa.piptrack(y=trimmed_audio, sr=sample_rate)
    spectrogram = librosa.feature.melspectrogram(y=trimmed_audio, sr=sample_rate)

    def pad_or_truncate(feature, max_len):
        current_len = feature.shape[1]
        if current_len < max_len:
            pad_width = max_len - current_len
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        else:
            feature = feature[:, :max_len]
        return feature

    mfcc = pad_or_truncate(mfcc, max_len)
    chroma = pad_or_truncate(chroma, max_len)
    pitch = pad_or_truncate(pitch, max_len)
    spectrogram = pad_or_truncate(spectrogram, max_len)

    features = np.hstack([mfcc.flatten(), chroma.flatten(), pitch.flatten(), spectrogram.flatten()])
    return features


class SpeechDataset(Dataset):
    def __init__(self, dataset_path, augment=False):
        self.file_paths = []
        self.labels = []
        self.augment = augment

        for file in os.listdir(dataset_path):
            if file.endswith(".wav"):
                self.file_paths.append(os.path.join(dataset_path, file))
                label = 1 if "correct" in file.lower() else 0
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feature = extract_features(self.file_paths[idx], augment=self.augment)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(feature, dtype=torch.float32), label


class SpeechModel(nn.Module):
    def __init__(self, input_size):
        super(SpeechModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x 


def train():
    augment = True  
    dataset = SpeechDataset(DATASET_PATH, augment=augment)
    sample_feature, _ = dataset[0]
    feature_size = len(sample_feature)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SpeechModel(feature_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze(dim=1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                val_outputs = model(batch_features).squeeze(dim=1)
                val_loss += criterion(val_outputs, batch_labels).item()

        if epoch % 10 == 0 or epoch == 49:
            print(f"[{PHRASE_NAME}] Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
