import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.dataset import HarmonicDataset
from src.ensemble_model import EnsembleHarmonicModel
from src import config
import os
import numpy as np

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = HarmonicDataset(root_dir='data')

    # Calculate Class Weights
    y_labels = np.array(dataset.labels)
    n_pos = np.sum(y_labels == 1.0)
    n_neg = np.sum(y_labels == 0.0)

    # Weight = Number of Negatives / Number of Positives
    # If n_pos is 0, default to 1
    weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    # pos_weight = torch.tensor([weight_val], dtype=torch.float).to(device)
    print(f"Class Distribution: YES={n_pos}, NO={n_neg}, Pos Weight={weight_val:.2f}")

    # Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")

    # Initialize Ensemble
    ensemble = EnsembleHarmonicModel(device=device)

    # Train
    # We pass weight_val as the positive weight for both Linear (BCE) and XGBoost (scale_pos_weight)
    ensemble.fit(train_loader, val_loader, pos_weight=weight_val)

    # Save
    model_dir = 'models/ensemble'
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    ensemble.save(model_dir)
    print(f"Ensemble models saved to {model_dir}")

if __name__ == "__main__":
    train()
