import torch
import torch.nn as nn
from transformers import DistilBertModel

class TESSTextModel(nn.Module):
    def __init__(self, num_classes=7, freeze_bert=True):
        super(TESSTextModel, self).__init__()

        print("Loading pre-trained DistilBERT base...")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # DistilBERT outputs 768-dimensional embeddings
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # High dropout because we expect the text to be confusing
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Extract features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # We take the hidden states of the last layer
        hidden_states = outputs.last_hidden_state

        # Global Average Pooling: We average the token embeddings across the sequence
        # to get a single vector representing the whole sentence
        pooled_output = torch.mean(hidden_states, dim=1)

        # Output probabilities (logits)
        logits = self.classifier(pooled_output)
        return logits

if __name__ == "__main__":
    print("Testing Text Model Architecture...")
    dummy_input_ids = torch.randint(0, 1000, (2, 16))
    dummy_mask = torch.ones(2, 16)

    model = TESSTextModel()
    output = model(dummy_input_ids, dummy_mask)
    print(f"Output shape: {output.shape}")