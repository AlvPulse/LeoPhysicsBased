import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.dataset import HarmonicDataset
from src.models import LinearHarmonicModel
# GNNEventDetector removed
from src import config
import os
import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib

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
    pos_weight = torch.tensor([weight_val], dtype=torch.float).to(device)
    print(f"Class Distribution: YES={n_pos}, NO={n_neg}, Pos Weight={weight_val:.2f}")

    # Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Loaders for Neural Networks
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")

    # --- 1. Linear Model (Legacy/Recall) ---
    linear_model = LinearHarmonicModel().to(device)
    opt_linear = optim.Adam(linear_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training Linear Model...")
    for epoch in range(config.EPOCHS):
        linear_model.train()
        train_loss_lin = 0
        train_acc_lin = 0

        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.unsqueeze(1)

            lin_feat = batch.linear_features
            if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)

            opt_linear.zero_grad()
            out_linear = linear_model(lin_feat)
            loss_linear = criterion(out_linear, labels)
            loss_linear.backward()
            opt_linear.step()

            train_loss_lin += loss_linear.item()
            train_acc_lin += ((out_linear > 0) == labels).sum().item()

        # Validation
        linear_model.eval()
        val_acc_lin = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                labels = batch.y.unsqueeze(1)
                lin_feat = batch.linear_features
                if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)
                out_linear = linear_model(lin_feat)
                val_acc_lin += ((out_linear > 0) == labels).sum().item()

        t_acc_lin = train_acc_lin / len(train_dataset)
        v_acc_lin = val_acc_lin / len(val_dataset)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Linear (Train/Val Acc): {t_acc_lin:.2f}/{v_acc_lin:.2f}")

    # --- 2. XGBoost Classifier (Precision) ---
    print("\nPreparing data for XGBoost Classifier...")

    def extract_numpy_data(loader):
        X_list = []
        y_list = []
        for batch in loader:
            # Extract classifier features
            feats = batch.classifier_features
            # Ensure it's (N, FeatureDim)
            if feats.dim() == 3: feats = feats.squeeze(1)

            X_list.append(feats.cpu().numpy())
            y_list.append(batch.y.cpu().numpy())

        if not X_list:
            return np.array([]), np.array([])

        return np.vstack(X_list), np.hstack(y_list)

    X_train, y_train = extract_numpy_data(train_loader)
    X_val, y_val = extract_numpy_data(val_loader)

    print(f"Features extracted. X_train shape: {X_train.shape}")

    # Initialize XGBoost Classifier
    # scale_pos_weight handles class imbalance (neg/pos)
    clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        scale_pos_weight=weight_val,
        eval_metric='logloss'
    )

    print("Training XGBoost Classifier...")
    clf.fit(X_train, y_train)

    # Validation Metrics
    print("\n--- Classifier Validation Metrics ---")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=['NO', 'YES']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(linear_model.state_dict(), 'models/linear_model.pth')

    # Save the XGBoost model
    # joblib works for sklearn-like wrappers
    joblib.dump(clf, 'models/classifier_model.pkl')

    print("Models saved.")

if __name__ == "__main__":
    train()
