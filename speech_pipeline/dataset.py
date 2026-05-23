import os
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
from pathlib import Path
import sys

# Import our data manager
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project.utils.data_manager import get_data_path

class TESSSpeechDataset(Dataset):
    def __init__(self, target_sr=16000):
        self.root_path = get_data_path()
        if not self.root_path:
            raise FileNotFoundError("Dataset path could not be resolved.")

        self.target_sr = target_sr
        self.file_paths = []
        self.labels = []

        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'pleasant_surprised': 5, 'sad': 6
        }
        self._parse_dataset()

    def _parse_dataset(self):
        all_files = list(self.root_path.rglob("*.wav"))
        print(f"Found {len(all_files)} total audio files. Parsing labels & deduplicating...")

        seen_filenames = set()
        for file_path in all_files:
            filename = file_path.stem.lower()

            if filename in seen_filenames:
                continue

            parts = filename.split('_')
            emotion = parts[-1]

            if emotion == 'ps' or 'surprise' in emotion:
                emotion = 'pleasant_surprised'

            if emotion in self.emotion_map:
                self.file_paths.append(str(file_path))
                self.labels.append(self.emotion_map[emotion])
                seen_filenames.add(filename)
            else:
                print(f"Warning: Could not parse emotion for {filename}")
        print(f"Successfully loaded {len(self.labels)} unique, valid samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return waveform_tensor, label_tensor

if __name__ == "__main__":
    print("Testing the TESS Dataset Loader...")
    dataset = TESSSpeechDataset()
    if len(dataset) > 0:
        audio, label = dataset[0]
        print(f"First sample shape: {audio.shape} | Label: {label}")