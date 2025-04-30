import os
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

class MFCCDataset(Dataset):
    def __init__(self, root_folder, sample_rate=16000, n_mfcc=13):
        self.file_paths = []
        self.labels = []
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)

        phrase_folders = sorted(os.listdir(root_folder))
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

        mfcc = self.transform(waveform).squeeze(0)
        mfcc = mfcc[:, :300] if mfcc.size(1) > 300 else torch.nn.functional.pad(mfcc, (0, 300 - mfcc.size(1)))
        label = self.labels[idx]
        return mfcc, label


class SpeechModel(nn.Module):
    def __init__(self, input_dim=13):
        super(SpeechModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, dataloader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for mfcc, label in dataloader:
            mfcc, label = mfcc.to(device), label.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(mfcc)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    data_dir = "D:/Code research/Dataset-Sorted"
    dataset = MFCCDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SpeechModel(input_dim=13)
    train(model, dataloader, epochs=10, lr=0.001)

    torch.save(model.state_dict(), "speech_model.pth")
    print("✅ Model saved to speech_model.pth")
