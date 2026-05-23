import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add root to sys.path to import our sub-models
sys.path.append(str(Path(__file__).resolve().parents[1]))
from project.speech_pipeline.model import TESSSpeechModel
from project.text_pipeline.model import TESSTextModel

class LateFusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(LateFusionModel, self).__init__()

        print("Initializing Sub-Models for Late Fusion...")
        # 1. Initialize models
        self.speech_model = TESSSpeechModel(freeze_hubert=True)
        self.text_model = TESSTextModel(freeze_bert=True)

        # 2. Load pre-trained weights
        try:
            self.speech_model.load_state_dict(torch.load("models/speech_pipeline_weights.pth", weights_only=True))
            self.text_model.load_state_dict(torch.load("models/text_pipeline_weights.pth", weights_only=True))
            print("Successfully loaded pre-trained Speech and Text weights.")
        except FileNotFoundError:
            print("CRITICAL WARNING: Pre-trained weights not found! Meta-classifier will train on untrained embeddings.")

        # 3. Freeze sub-models entirely!
        # We only want to train the fusion head, not destroy our pre-trained backbones.
        for param in self.speech_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

        # 4. The Meta-Classifier
        # 7 speech logits + 7 text logits = 14 inputs
        self.meta_classifier = nn.Sequential(
            nn.Linear(14, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, audio, input_ids, attention_mask):
        # We use torch.no_grad() for the sub-models to save GPU memory during the forward pass
        with torch.no_grad():
            # Speech branch (ensure sub-model is in eval mode so dropout is off)
            self.speech_model.eval()
            speech_logits = self.speech_model(audio)

            # Text branch
            self.text_model.eval()
            text_logits = self.text_model(input_ids, attention_mask)

        # Concatenate along the feature dimension
        fused_features = torch.cat((speech_logits, text_logits), dim=1)

        # Final classification
        output = self.meta_classifier(fused_features)
        return output

if __name__ == "__main__":
    print("Testing Fusion Architecture...")
    dummy_audio = torch.randn(2, 16000)
    dummy_ids = torch.randint(0, 1000, (2, 16))
    dummy_mask = torch.ones(2, 16)

    model = LateFusionModel()
    output = model(dummy_audio, dummy_ids, dummy_mask)
    print(f"Output shape: {output.shape} (Expected: [2, 7])")