import os
import torch
import librosa
from torch.utils.data import Dataset
from pathlib import Path
from transformers import DistilBertTokenizer
import sys

# Import our data manager
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project.utils.data_manager import get_data_path

class MultimodalTESSDataset(Dataset):
    def __init__(self, target_sr=16000, max_length=16):
        """
        Synchronized Data Loader for Early/Late Fusion.
        Returns Audio, Text Input IDs, Text Attention Mask, and Label in a single tuple.
        """
        self.root_path = get_data_path()
        if not self.root_path:
            raise FileNotFoundError("Dataset path could not be resolved.")

        self.target_sr = target_sr
        self.max_length = max_length
        print("Loading DistilBERT Tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.file_paths = []
        self.texts = []
        self.labels = []

        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'pleasant_surprised': 5, 'sad': 6
        }

        self._parse_dataset()

    def _parse_dataset(self):
        all_files = list(self.root_path.rglob("*.wav"))
        print(f"Parsing Multimodal Data (Speech + Text) from {len(all_files)} files...")

        seen_filenames = set()
        for file_path in all_files:
            filename = file_path.stem.lower()
            if filename in seen_filenames:
                continue

            parts = filename.split('_')
            if len(parts) >= 3:
                target_word = parts[1]
                emotion = parts[-1]

                if emotion == 'ps' or 'surprise' in emotion:
                    emotion = 'pleasant_surprised'

                if emotion in self.emotion_map:
                    # Sync 1: The Audio Path
                    self.file_paths.append(str(file_path))
                    # Sync 2: The Text Transcript
                    self.texts.append(f"Say the word {target_word}")
                    # Sync 3: The Label
                    self.labels.append(self.emotion_map[emotion])
                    seen_filenames.add(filename)

        print(f"Successfully loaded {len(self.labels)} perfectly synced multimodal samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # --- 1. Speech Modality ---
        audio_path = self.file_paths[idx]
        waveform, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        audio_tensor = torch.tensor(waveform, dtype=torch.float32)

        # --- 2. Text Modality ---
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # --- 3. Ground Truth Label ---
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        # Return ALL elements required for both models
        return audio_tensor, input_ids, attention_mask, label_tensor

# --- Sanity Check ---
if __name__ == "__main__":
    print("Testing Multimodal Dataset Loader...")
    dataset = MultimodalTESSDataset()
    if len(dataset) > 0:
        audio, ids, mask, label = dataset[0]
        print("\n--- Sync Verification ---")
        print(f"Audio Tensor Shape: {audio.shape}")
        print(f"Text IDs Shape: {ids.shape}")
        print(f"Label: {label}")
        print("Data is locked and synchronized.")