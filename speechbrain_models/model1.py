import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from speechbrain.processing.features import KaldiMFCC
from speechbrain.utils.data_utils import get_all_files


# Dataset class for MFCC extraction using KaldiMFCC
class MFCCDataset(Dataset):
    def __init__(self, root_folder, sample_rate=16000, n_mfcc=13, target_folder=None):
        self.file_paths = []
        self.labels = []
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        # Initialize KaldiMFCC feature extractor from SpeechBrain
        self.mfcc_extractor = KaldiMFCC(
            sample_rate=self.sample_rate, 
            num_ceps=self.n_mfcc,
            melkwargs={"n_mels": 23, "window_size": 25, "window_stride": 10, "window": "hamming"}
        )

        # Use only the specified folder if provided, otherwise use all folders
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
        
        # Resample if the sample rate doesn't match
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Extract MFCC using SpeechBrain's KaldiMFCC extractor
        mfcc = self.mfcc_extractor.encode_audio(waveform).squeeze(0)  # shape: (n_mfcc, time)
        
        # Pad or trim to match the required size (e.g., 300 time steps)
        mfcc = mfcc[:, :300] if mfcc.size(1) > 300 else torch.nn.functional.pad(mfcc, (0, 300 - mfcc.size(1)))

        label = self.labels[idx]
        return mfcc, label


# Neural network model using Conv1D
class SpeechModel(nn.Module):
    def __init__(self, input_dim=13):
        super(SpeechModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Use sigmoid or softmax for classification tasks

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training loop using SpeechBrain's optimizers
def train(model, dataloader, epochs=50, lr=0.001):
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


# Main execution
if __name__ == "__main__":
    data_dir = "D:/Code research/Dataset-Sorted"
    target_folder = "phrase003"  # Specify which folder to run, e.g., "phrase001"
    
    # Use custom dataset loading
    dataset = MFCCDataset(data_dir, target_folder=target_folder)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize the model
    model = SpeechModel(input_dim=13)
    
    # Train the model
    train(model, dataloader, epochs=10, lr=0.001)

    # Save the trained model
    model_save_path = os.path.join("model", f"{target_folder}_speech_model.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")
