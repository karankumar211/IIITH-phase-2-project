import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from project.fusion_pipeline.dataset import MultimodalTESSDataset
from project.fusion_pipeline.model import LateFusionModel
from project.fusion_pipeline.train import multimodal_collate

def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Fusion Model on device: {device}")

    # 1. Load Data (Strict Seed)
    full_dataset = MultimodalTESSDataset(target_sr=16000, max_length=16)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=multimodal_collate)

    # 2. Load Model
    model = LateFusionModel().to(device)
    try:
        model.load_state_dict(torch.load("models/fusion_pipeline_weights.pth", weights_only=True))
        print("Loaded Fusion Weights Successfully.")
    except FileNotFoundError:
        print("CRITICAL: Run train.py first to generate weights.")
        return

    model.eval()

    all_preds = []
    all_targets = []
    all_features = [] # To store the 14-D vectors for t-SNE

    print("\n--- Running Inference ---")
    with torch.no_grad():
        for audios, input_ids, masks, targets in val_loader:
            audios, input_ids = audios.to(device), input_ids.to(device)
            masks, targets = masks.to(device), targets.to(device)

            # Forward pass through sub-models to get features
            speech_logits = model.speech_model(audios)
            text_logits = model.text_model(input_ids, masks)

            # The 14-D latent space
            fused_features = torch.cat((speech_logits, text_logits), dim=1)

            # Final output
            outputs = model.meta_classifier(fused_features)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_features.extend(fused_features.cpu().numpy())

    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']

    # 3. Print Metrics
    print("\n--- Final Multimodal Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=emotion_classes))

    # 4. Generate t-SNE Visualization
    print("\nGenerating t-SNE Visualization (This might take a few seconds)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(np.array(all_features))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=[emotion_classes[i] for i in all_targets],
        palette=sns.color_palette("hsv", 7),
        legend="full", alpha=0.8
    )
    plt.title("t-SNE Latent Space Visualization of Fused Multimodal Features")
    plt.savefig('Results/tsne_visualization.png')
    print("t-SNE plot saved to Results/tsne_visualization.png")

if __name__ == "__main__":
    test_and_visualize()