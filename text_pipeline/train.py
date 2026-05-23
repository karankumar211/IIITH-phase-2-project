import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from project.text_pipeline.dataset import TESSTextDataset
from project.text_pipeline.model import TESSTextModel

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using compute device: {device}")

    print("Loading Text Dataset...")
    full_dataset = TESSTextDataset(max_length=16)

    # CRITICAL: Synchronized Data Split
    # We use manual_seed(42) so the train/val split matches the speech pipeline EXACTLY.
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Notice we don't need a custom collate_fn here!
    # DistilBertTokenizer already padded everything to max_length=16.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("Initializing Text Model...")
    model = TESSTextModel(freeze_bert=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=2e-4)

    epochs = 3 # Kept short because the text is emotionally neutral
    print("\n--- Starting Text Training ---")
    print("EXPECTATION: Loss will likely not drop significantly. Accuracy will be low (~14%).")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (input_ids, attention_mask, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        print(f"-> Epoch {epoch+1} Average Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained weights for the Fusion pipeline
    torch.save(model.state_dict(), "models/text_pipeline_weights.pth")
    print("\nTraining Complete! Model saved to models/text_pipeline_weights.pth")

if __name__ == "__main__":
    train_model()