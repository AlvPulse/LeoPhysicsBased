import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.dataset import HarmonicDataset
from src.models import LinearHarmonicModel, GNNEventDetector
from src import config
import os
import numpy as np

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = HarmonicDataset(root_dir='data')

    # Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")

    linear_model = LinearHarmonicModel().to(device)
    gnn_model = GNNEventDetector().to(device)

    opt_linear = optim.Adam(linear_model.parameters(), lr=config.LEARNING_RATE)
    opt_gnn = optim.Adam(gnn_model.parameters(), lr=config.LEARNING_RATE)

    criterion = nn.BCELoss()

    for epoch in range(config.EPOCHS):
        linear_model.train()
        gnn_model.train()

        train_loss_lin = 0
        train_loss_gnn = 0
        train_acc_lin = 0
        train_acc_gnn = 0

        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.unsqueeze(1)

            # Linear Train
            lin_feat = batch.linear_features
            if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)

            opt_linear.zero_grad()
            out_linear = linear_model(lin_feat)
            loss_linear = criterion(out_linear, labels)
            loss_linear.backward()
            opt_linear.step()

            train_loss_lin += loss_linear.item()
            train_acc_lin += ((out_linear > 0.5) == labels).sum().item()

            # GNN Train
            opt_gnn.zero_grad()
            out_gnn = gnn_model(batch.x, batch.edge_index, batch.batch)
            loss_gnn = criterion(out_gnn, labels)
            loss_gnn.backward()
            opt_gnn.step()

            train_loss_gnn += loss_gnn.item()
            train_acc_gnn += ((out_gnn > 0.5) == labels).sum().item()

        # Validation
        linear_model.eval()
        gnn_model.eval()

        val_acc_lin = 0
        val_acc_gnn = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                labels = batch.y.unsqueeze(1)

                lin_feat = batch.linear_features
                if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)

                out_linear = linear_model(lin_feat)
                val_acc_lin += ((out_linear > 0.5) == labels).sum().item()

                out_gnn = gnn_model(batch.x, batch.edge_index, batch.batch)
                val_acc_gnn += ((out_gnn > 0.5) == labels).sum().item()

        # Metrics
        t_acc_lin = train_acc_lin / len(train_dataset)
        t_acc_gnn = train_acc_gnn / len(train_dataset)
        v_acc_lin = val_acc_lin / len(val_dataset)
        v_acc_gnn = val_acc_gnn / len(val_dataset)

        print(f"Epoch {epoch+1:02d} | "
              f"Linear (Train/Val Acc): {t_acc_lin:.2f}/{v_acc_lin:.2f} | "
              f"GNN (Train/Val Acc): {t_acc_gnn:.2f}/{v_acc_gnn:.2f}")

    # Save
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(linear_model.state_dict(), 'models/linear_model.pth')
    torch.save(gnn_model.state_dict(), 'models/gnn_model.pth')
    print("Models saved.")

if __name__ == "__main__":
    train()
