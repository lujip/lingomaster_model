import os
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

#improtanttt
# using 13 mfcc weight,   low = 13,  normal or ave 40 - 60, 128 is mel spectrogram

# MFCC extraction
class MFCCDataset(Dataset):
    def __init__(self, root_folder, sample_rate=16000, n_mfcc=13, target_folder=None):
        self.file_paths = []
        self.labels = []
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
        self.vad = T.Vad(sample_rate=sample_rate)  

        
        phrase_folders = [target_folder] if target_folder else sorted(os.listdir(root_folder))
        for label_idx, phrase in enumerate(phrase_folders):
            phrase_folder = os.path.join(root_folder, phrase)
            for file in os.listdir(phrase_folder):
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(phrase_folder, file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        
        if sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

       
        waveform = self.vad(waveform)

        if waveform.numel() == 0:
            print(f"⚠️ Skipping empty waveform after VAD: {audio_path}")
            return torch.zeros((self.n_mfcc, 300)), -1 

        mfcc = self.transform(waveform).squeeze(0)

        mfcc = mfcc[:, :300] if mfcc.size(1) > 300 else torch.nn.functional.pad(mfcc, (0, 300 - mfcc.size(1)))

        label = self.labels[idx]
        return mfcc, label


#neural network model using Conv1D
class SpeechModel(nn.Module):
    def __init__(self, input_dim=13):
        super(SpeechModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  
        
        #use sigmoid or softmax for classification tasks, test #23

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#training loop
def train(model, dataloader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0

        for mfcc, label in dataloader:
            if (label == -1).any():
                continue 

            mfcc, label = mfcc.to(device), label.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(mfcc)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            valid_batches += 1

        if valid_batches > 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/valid_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - ⚠️ No valid batches to train on.")


#remember to change to sigmoid
if __name__ == "__main__":
    data_dir = "D:/Code research/To be Trained"
    num_phrases = 24

    for i in range(1, num_phrases + 1):
        target_folder = f"phrase{i:03d}"  
        print(f"\n🎯 Training on: {target_folder}")

        dataset = MFCCDataset(data_dir, target_folder=target_folder)
        if len(dataset) == 0:
            print(f"⚠️ Skipping {target_folder} (no .wav files found)")
            continue

        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = SpeechModel(input_dim=13)
        train(model, dataloader, epochs=20, lr=0.001)

        model_save_path = os.path.join("model", f"{target_folder}_speech_model.pth")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Model saved to {model_save_path}")
