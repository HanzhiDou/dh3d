import os
import sys
import warnings
# Enable MPS fallback for Mac users automatically
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress the specific MPS fallback warning
warnings.filterwarnings("ignore", message="The operator 'aten::slow_conv3d_forward' is not currently supported on the MPS backend")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from hologram_dataset_classic import HologramDatasetClassic, GRID_SIZE, Z_SIZE
from unet3d_model import ConcatDeepResUNet3D
from losses import BalancedHybridLoss
from torchsummary import summary


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_MODEL_PATH = "unet3d_dih_256_x.pth"

BATCH_SIZE = 2
EPOCHS = 5

if __name__ == "__main__":
    # Create Datasets
    num_samples_train = 800
    num_samples_val = 10

    # Note that the number of particles is fixed in this model training
    train_set = HologramDatasetClassic(num_samples=num_samples_train, num_particles=100, is_train=True)
    val_set = HologramDatasetClassic(num_samples=num_samples_val, num_particles=60, is_train=False) # Val is clean

    # Create Dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for val

    # Optimizer & Loss
    model = ConcatDeepResUNet3D().to(DEVICE)
    summary(model, input_size=(1, Z_SIZE, GRID_SIZE, GRID_SIZE))

    scale_vector = torch.tensor([10.0, 1.0, 1.0]).to(DEVICE)
    criterion = BalancedHybridLoss(scale_vector,pos_weight=500.0).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []

    print("\nStarting Model Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for dirty, clean, _, coords in train_loader:
            # print('dirty:', dirty.shape)
            # print('clearn:', clean.shape)
            # print('coords:', coords.shape)
            dirty = dirty.to(DEVICE)
            clean = clean.to(DEVICE)
            coords = coords.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(dirty)
            loss = criterion(outputs, clean, coords)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss)

        # Validation step: Check performance on CLEAN data
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_dirty, v_clean, _, v_coords in val_loader:
                v_dirty = v_dirty.to(DEVICE)
                v_clean = v_clean.to(DEVICE)
                v_coords = v_coords.to(DEVICE)
                v_outputs = model(v_dirty)
                val_loss += criterion(v_outputs, v_clean, v_coords).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch} | Train: {epoch_loss/len(train_loader):.4f} | Val: {val_loss/len(val_loader):.4f}")
        torch.save(model.state_dict(), f"active_model_{epoch}.pth")
        print(f"Model saved to active_model_{epoch}.pth")


    # Save the trained model
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"Model saved to {OUTPUT_MODEL_PATH}")
