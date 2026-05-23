import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

from project.fusion_pipeline.dataset import MultimodalTESSDataset
from project.fusion_pipeline.model import LateFusionModel

# --- THE PADDING PUZZLE SOLVED ---
def multimodal_collate(batch):
    # Unpack the batch
    audios = [item[0] for item in batch]
    input_ids = [item[1] for item in batch]
    attention_masks = [item[2] for item in batch]
    labels = [item[3] for item in batch]

    # Pad ONLY the audio. Everything else is already uniform length.
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    input_ids_stacked = torch.stack(input_ids)
    attention_masks_stacked = torch.stack(attention_masks)
    labels_stacked = torch.stack(labels)

    return audios_padded, input_ids_stacked, attention_masks_stacked, labels_stacked

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using compute device: {device}")

    print("Loading Multimodal Dataset...")
    full_dataset = MultimodalTESSDataset(target_sr=16000, max_length=16)

    # CRITICAL: Same Seed as Speech & Text pipelines
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # We use our custom multimodal_collate here
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=multimodal_collate)

    print("Initializing Fusion Model...")
    model = LateFusionModel().to(device)

    criterion = nn.CrossEntropyLoss()
    # Notice we only optimize the meta_classifier parameters. Training is lightning fast.
    optimizer = optim.Adam(model.meta_classifier.parameters(), lr=5e-3)

    epochs = 5
    print("\n--- Starting Meta-Classifier Training ---")

    for epoch in range(epochs):
        model.train() # This only affects the meta_classifier since backbones are frozen
        running_loss = 0.0

        for batch_idx, (audios, input_ids, masks, targets) in enumerate(train_loader):
            audios = audios.to(device)
            input_ids = input_ids.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Pass all modalities to the model
            outputs = model(audios, input_ids, masks)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 40 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        print(f"-> Epoch {epoch+1} Average Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/fusion_pipeline_weights.pth")
    print("\nTraining Complete! Model saved to models/fusion_pipeline_weights.pth")

if __name__ == "__main__":
    train_model()