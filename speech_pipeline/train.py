import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from project.speech_pipeline.dataset import TESSSpeechDataset
from project.speech_pipeline.model import TESSSpeechModel

def pad_collate(batch):
    audios = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    labels_stacked = torch.stack(labels)
    return audios_padded, labels_stacked

def train_model():
    # This line ensures we use the free Colab GPU!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using compute device: {device}")

    print("Loading Dataset...")
    full_dataset = TESSSpeechDataset(target_sr=16000)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # We use a conservative batch size of 8 to avoid Colab OOM errors
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)

    print("Initializing Model...")
    model = TESSSpeechModel(freeze_hubert=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

    epochs = 5
    print("\n--- Starting Training ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        print(f"-> Epoch {epoch+1} Average Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained weights to Google Drive so they aren't lost when Colab closes!
    torch.save(model.state_dict(), "models/speech_pipeline_weights.pth")
    print("\nTraining Complete! Model saved to models/speech_pipeline_weights.pth")

if __name__ == "__main__":
    train_model()