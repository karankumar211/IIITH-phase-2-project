import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from project.speech_pipeline.dataset import TESSSpeechDataset
from project.speech_pipeline.model import TESSSpeechModel
from project.speech_pipeline.train import pad_collate

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load the exact same dataset split
    full_dataset = TESSSpeechDataset(target_sr=16000)

    # We must seed the split so it perfectly matches the train.py split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)

    # 2. Load the trained model weights
    print("Loading saved model weights...")
    model = TESSSpeechModel(freeze_hubert=True).to(device)
    try:
        model.load_state_dict(torch.load("models/speech_pipeline_weights.pth"))
    except FileNotFoundError:
        print("CRITICAL: Could not find trained weights. Run train.py first.")
        return

    model.eval() # Disable dropout and batch norm

    all_preds = []
    all_targets = []

    print("\n--- Starting Evaluation ---")
    with torch.no_grad(): # Don't calculate gradients during testing (saves memory)
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 3. Generate Metrics (Required for Report)
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprised', 'Sad']

    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=emotion_classes))

    # 4. Save Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    plt.title('Speech Pipeline: Confusion Matrix')
    plt.savefig('Results/speech_confusion_matrix.png')
    print("Confusion matrix saved to Results/speech_confusion_matrix.png")

if __name__ == "__main__":
    test_model()