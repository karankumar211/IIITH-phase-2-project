import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

from project.text_pipeline.dataset import TESSTextDataset
from project.text_pipeline.model import TESSTextModel

def test_text_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Text Model on device: {device}")

    # 1. Load Data with STRICT SEED (Must match Speech and Fusion!)
    full_dataset = TESSTextDataset(max_length=16)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 2. Load Model
    print("Loading saved text model weights...")
    model = TESSTextModel(freeze_bert=True).to(device)
    try:
        # Note: weights_only=True is best practice for PyTorch 2.0+ security
        model.load_state_dict(torch.load("../models/text_pipeline_weights.pth", weights_only=True))
    except FileNotFoundError:
        # Fallback path just in case they run it from the root directory
        try:
            model.load_state_dict(torch.load("models/text_pipeline_weights.pth", weights_only=True))
        except FileNotFoundError:
            print("CRITICAL: Could not find trained weights. Run train.py first.")
            return
        
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("\n--- Running Inference (Expecting ~14% Accuracy) ---")
    with torch.no_grad():
        for input_ids, masks, targets in val_loader:
            input_ids, masks = input_ids.to(device), masks.to(device)
            targets = targets.to(device)
            
            outputs = model(input_ids, masks)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
    
    # zero_division=0 prevents warnings when a model completely fails to predict certain classes
    print("\n--- Text Pipeline Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=emotion_classes, zero_division=0))

if __name__ == "__main__":
    test_text_model()