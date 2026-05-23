import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import DistilBertTokenizer
import sys

# Import our data manager
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project.utils.data_manager import get_data_path

class TESSTextDataset(Dataset):
    def __init__(self, max_length=16):
        self.root_path = get_data_path()
        if not self.root_path:
            raise FileNotFoundError("Dataset path could not be resolved.")

        self.texts = []
        self.labels = []

        # We use DistilBERT's tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length

        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'pleasant_surprised': 5, 'sad': 6
        }
        self._parse_dataset()

    def _parse_dataset(self):
        all_files = list(self.root_path.rglob("*.wav"))
        print(f"Parsing Text transcripts from {len(all_files)} audio filenames...")

        seen_filenames = set()
        for file_path in all_files:
            filename = file_path.stem.lower()

            if filename in seen_filenames:
                continue

            parts = filename.split('_')

            # The TESS format is usually: [Speaker]_[Word]_[Emotion].wav
            # e.g., OAF_back_angry -> Word is 'back', Emotion is 'angry'
            if len(parts) >= 3:
                target_word = parts[1]
                emotion = parts[-1]

                if emotion == 'ps' or 'surprise' in emotion:
                    emotion = 'pleasant_surprised'

                if emotion in self.emotion_map:
                    # RECONSTRUCT THE TEXT TRANSCRIPT
                    transcript = f"Say the word {target_word}"

                    self.texts.append(transcript)
                    self.labels.append(self.emotion_map[emotion])
                    seen_filenames.add(filename)

        print(f"Successfully generated {len(self.labels)} text transcripts.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text [cite: 36]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Squeeze removes the batch dimension added by tokenizer
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label_tensor

if __name__ == "__main__":
    print("Testing the TESS Text Dataset Loader...")
    dataset = TESSTextDataset()
    if len(dataset) > 0:
        input_ids, mask, label = dataset[0]
        print(f"First transcript text: {dataset.texts[0]}")
        print(f"Input IDs shape: {input_ids.shape} | Label: {label}")