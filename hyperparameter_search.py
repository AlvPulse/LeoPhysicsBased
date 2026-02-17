import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.dataset import HarmonicDataset
from src.ensemble_model import EnsembleHarmonicModel
from src import config
import os
import numpy as np
import itertools
from sklearn.metrics import f1_score, recall_score, precision_score
import xgboost as xgb

def evaluate(ensemble, loader, device):
    all_preds = []
    all_labels = []

    ensemble.linear_model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            labels = batch.y.cpu().numpy()

            # Linear Features
            lin_feat = batch.linear_features
            if lin_feat.dim() == 3: lin_feat = lin_feat.squeeze(1)

            # XGB Features
            xgb_feat = batch.classifier_features
            if xgb_feat.dim() == 3: xgb_feat = xgb_feat.squeeze(1)
            xgb_feat_np = xgb_feat.cpu().numpy()

            preds = []
            for i in range(len(labels)):
                l_vec = lin_feat[i]
                x_vec = xgb_feat_np[i]

                _, _, p_meta = ensemble.predict_proba(l_vec, x_vec)
                preds.append(1 if p_meta > 0.5 else 0)

            all_preds.extend(preds)
            all_labels.extend(labels)

    return f1_score(all_labels, all_preds), recall_score(all_labels, all_preds), precision_score(all_labels, all_preds)

def hyperparameter_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    dataset = HarmonicDataset(root_dir='data')

    # Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Class Weights
    y_labels = np.array(dataset.labels)
    n_pos = np.sum(y_labels == 1.0)
    n_neg = np.sum(y_labels == 0.0)
    weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class Distribution: YES={n_pos}, NO={n_neg}, Pos Weight={weight_val:.2f}")

    # Define Hyperparameter Grid
    param_grid = {
        'learning_rate_linear': [0.001, 0.01],
        'epochs_linear': [10, 20],
        'xgb_n_estimators': [50, 100],
        'xgb_max_depth': [3, 5],
        'xgb_learning_rate': [0.1, 0.2]
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    print(f"Starting Grid Search with {len(combinations)} combinations...")

    best_score = 0.0
    best_params = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\nEvaluating Combo {i+1}/{len(combinations)}: {params}")

        # Initialize Ensemble with specific XGB params
        xgb_params = {
            'n_estimators': params['xgb_n_estimators'],
            'max_depth': params['xgb_max_depth'],
            'learning_rate': params['xgb_learning_rate'],
            'random_state': 42,
            'eval_metric': 'logloss'
        }

        ensemble = EnsembleHarmonicModel(device=device, xgb_params=xgb_params)

        # Train
        ensemble.fit(
            train_loader,
            val_loader,
            pos_weight=weight_val,
            lr_linear=params['learning_rate_linear'],
            epochs_linear=params['epochs_linear']
        )

        # Evaluate
        f1, recall, precision = evaluate(ensemble, val_loader, device)
        print(f"Result -> F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        if f1 > best_score:
            best_score = f1
            best_params = params
            print(">>> New Best Score!")

    print("\n" + "="*50)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*50)
    print(f"Best F1 Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    hyperparameter_search()
