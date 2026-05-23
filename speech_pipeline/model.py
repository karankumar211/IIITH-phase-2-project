import torch
import torch.nn as nn
from transformers import HubertModel

class TESSSpeechModel(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=256, num_lstm_layers=2, freeze_hubert=True):
        super(TESSSpeechModel, self).__init__()

        print("Loading pre-trained HuBERT base...")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        if freeze_hubert:
            for param in self.hubert.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs.last_hidden_state

        lstm_out, (hn, cn) = self.lstm(hidden_states)

        final_forward = hn[-2, :, :]
        final_backward = hn[-1, :, :]
        final_hidden = torch.cat((final_forward, final_backward), dim=1)

        logits = self.classifier(final_hidden)
        return logits

if __name__ == "__main__":
    print("Testing Speech Model Architecture...")
    dummy_input = torch.randn(2, 16000)
    model = TESSSpeechModel()
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")