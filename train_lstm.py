import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
from model import CNNLSTMModel
import random

# --- Sir, GLOBAL CONFIGURATION ---
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/sequences"  # Sir, this was the missing definition
MODEL_SAVE_PATH = "models/cnn_lstm_theft_model.pth"

class TheftDataset(Dataset):
    def __init__(self, data_dir, files=None):
        self.data_dir = data_dir
        # If no files provided, load all from directory
        self.files = files if files is not None else [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        # Label 1 = Shoplifting, Label 0 = Normal
        label = int(self.files[idx].split('_')[0])
        return torch.from_numpy(np.load(file_path)).float(), torch.tensor(label).long()

def train():
    # 1. Manual File Splitting
    if not os.path.exists(DATA_DIR):
        print(f"Sir, ERROR: The directory {DATA_DIR} does not exist.")
        return

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    if len(all_files) == 0:
        print("Sir, no .npy sequence files found in the data directory.")
        return

    random.shuffle(all_files)
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]

    # 2. Calculate Weights for Balancing (Training set only)
    train_labels = [int(f.split('_')[0]) for f in train_files]
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    # Check if we have both classes
    if len(unique_labels) < 2:
        print("Sir, warning: Only one class found in the training set. Balancing skipped.")
        sampler = None
    else:
        class_weights = 1. / counts
        sample_weights = np.array([class_weights[t] for t in train_labels])
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))

    # 3. Create DataLoaders
    train_ds = TheftDataset(DATA_DIR, files=train_files)
    val_ds = TheftDataset(DATA_DIR, files=val_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Setup
    model = CNNLSTMModel(num_classes=2).to(DEVICE)
    # Shoplifting (1) is weighted higher (2.5) to fix the 'blindness'
    loss_weights = torch.tensor([1.0, 2.5]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0
    print(f"Sir, Retraining System... (Train: {len(train_files)}, Val: {len(val_files)})")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for sequences, labels in loop:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for seqs, lbls in val_loader:
                seqs, lbls = seqs.to(DEVICE), lbls.to(DEVICE)
                preds = model(seqs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
        
        accuracy = 100 * correct / len(val_ds) if len(val_ds) > 0 else 0
        print(f"Sir, Validation Accuracy: {accuracy:.2f}%")
        
        if accuracy >= best_acc:
            best_acc = accuracy
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ Best Model Saved: {accuracy:.2f}%")

if __name__ == "__main__":
    train()