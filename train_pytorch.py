"""PyTorch training script for subcultural identity neural network."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from dataloader.dataset import MyDataset
from model.pytorch_model import create_model, save_model
from config.path import RESULTS
from config.tasks import TASKS

# Configuration
EPOCHS = 100
BATCH_SIZE = 40
LEARNING_RATE = 0.001
INPUT_DIMS, OUTPUT_DIMS = 5, 13
TASK = 'default'
OUTPUT_FOLDER = 'tmp_pytorch'
DEVICE = 'cpu'  # Use 'cuda' if GPU available

if __name__ == '__main__':
    print("Training PyTorch Model for Subcultural Identity Assessment")
    print("=" * 60)

    # Set device
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    print(f"Using device: {DEVICE}")

    # ---------- 1. Load data ----------
    print("\n1. Loading data...")
    ds = MyDataset(data_file='dataset300.csv').load(task=TASK)
    _, train_x, train_y, _, test_x, test_y = ds.prepare(train_ratio=0.8)

    print(f"   Train set: {train_x.shape}")
    print(f"   Test set: {test_x.shape}")

    # Convert to PyTorch tensors
    train_x = torch.FloatTensor(train_x).to(DEVICE)
    train_y = torch.FloatTensor(train_y).to(DEVICE)
    test_x = torch.FloatTensor(test_x).to(DEVICE)
    test_y = torch.FloatTensor(test_y).to(DEVICE)

    # Create data loaders
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------- 2. Create model ----------
    print("\n2. Creating model...")
    model = create_model(INPUT_DIMS, OUTPUT_DIMS)
    model = model.to(DEVICE)

    print(f"   Model architecture:")
    print(f"   Input: {INPUT_DIMS} features")
    print(f"   Hidden: 32 → 32 units (ReLU + Dropout 0.2)")
    print(f"   Output: {OUTPUT_DIMS} predictions")

    # ---------- 3. Setup training ----------
    print("\n3. Setting up training...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------- 4. Training loop ----------
    print(f"\n4. Training for {EPOCHS} epochs...")
    print("-" * 60)

    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(test_x)
            val_loss = criterion(val_outputs, test_y).item()
            val_losses.append(val_loss)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

    # ---------- 5. Save best model ----------
    print("\n5. Saving model...")
    model.load_state_dict(best_model_state)

    results_folder = os.path.join(RESULTS, OUTPUT_FOLDER, f'pytorch-{EPOCHS}epochs')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    model_path = os.path.join(results_folder, 'model.pt')
    save_model(model, model_path)

    print(f"   ✓ Model saved: {model_path}")
    print(f"   ✓ Best validation loss: {best_loss:.6f}")

    # ---------- 6. Test predictions ----------
    print("\n6. Testing predictions...")
    model.eval()
    with torch.no_grad():
        sample_input = test_x[:5]
        predictions = model(sample_input)
        print(f"   Sample predictions shape: {predictions.shape}")
        print(f"   ✓ Model is ready for inference!")

    print("\n" + "=" * 60)
    print("Training completed successfully! 🎉")
    print(f"Model saved to: {results_folder}/")
