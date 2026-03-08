import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        batch, seq_len, C, H, W = x.shape
        feats = []
        for t in range(seq_len):
            f = self.cnn(x[:, t])
            f = self.flatten(f)
            f = self.fc1(f)
            feats.append(f)
        seq_features = torch.stack(feats, dim=1)
        lstm_out, _ = self.lstm(seq_features)
        return self.classifier(lstm_out[:, -1, :])