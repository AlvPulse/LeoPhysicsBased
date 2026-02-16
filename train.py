import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader, Subset
from src import dataset, harmonic_detection, models, config, signal_processing, feature_extraction
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if isinstance(model, models.LinearHarmonicModel):
                # Linear model uses linear_x
                outputs = model(batch.linear_x)
            else:
                # GNN model uses graph data
                outputs = model(batch.x, batch.edge_index, batch.batch)

            # Outputs are logits. Sigmoid > 0.5 is equivalent to Logits > 0
            predicted = (outputs > 0.0).float()

            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())

    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return acc, prec, rec, f1

def calculate_pos_weight(dataset_subset):
    num_pos = 0
    total = len(dataset_subset)
    indices = dataset_subset.indices
    labels = dataset_subset.dataset.labels

    for idx in indices:
        if labels[idx] == 1:
            num_pos += 1

    num_neg = total - num_pos
    if num_pos > 0:
        return torch.tensor([num_neg / num_pos], device=device)
    return None

def get_subset_file_and_label(dataset_obj, idx):
    """Helper to get file path and label from a Subset or Dataset."""
    if isinstance(dataset_obj, Subset):
        real_idx = dataset_obj.indices[idx]
        return dataset_obj.dataset.file_list[real_idx], dataset_obj.dataset.labels[real_idx]
    else:
        return dataset_obj.file_list[idx], dataset_obj.labels[idx]

def random_search_optimization(full_dataset_obj, num_iterations=10):
    best_f1 = 0.0
    best_params = {}

    print(f"Starting Random Search Optimization ({num_iterations} iterations)...")

    # Use a larger subset for hyperparameter search to avoid overfitting/instability
    # Since dataset is cached, we can afford more.
    # Actually, for HYPERPARAMETER search on Detection Logic, we can't use the cached dataset features!
    # Because changing params changes the features.
    # So we MUST re-process raw audio.
    # To keep it fast, we use a small but representative subset.

    # Generate subset indices ONCE
    indices = list(range(len(full_dataset_obj)))
    random.shuffle(indices)
    search_subset_indices = indices[:40] # Use 40 files for search

    # Split this subset into train/val for the temporary linear model
    split = int(0.8 * len(search_subset_indices))
    train_indices = search_subset_indices[:split]
    val_indices = search_subset_indices[split:]

    # Extract raw audio paths/labels once
    subset_data = []
    for idx in search_subset_indices:
        filepath, label = get_subset_file_and_label(full_dataset_obj, idx)
        audio, fs = signal_processing.load_audio(filepath)
        if audio is None: continue
        # Pre-compute STFT/Peaks (independent of harmonic params)
        f, psd_db = signal_processing.compute_psd(audio, fs)
        nf = signal_processing.estimate_noise_floor(psd_db)
        peaks = signal_processing.find_significant_peaks(f, psd_db, nf)
        subset_data.append({'peaks': peaks, 'label': label, 'idx': idx})

    train_data = [d for d in subset_data if d['idx'] in train_indices]
    val_data = [d for d in subset_data if d['idx'] in val_indices]

    for i in range(num_iterations):
        # 1. Sample Hyperparameters
        params = {
            'tolerance': random.uniform(0.05, 0.25),
            'snr_threshold': random.uniform(1.0, 8.0),
            'power_threshold': random.uniform(-80.0, -40.0),
            'weights': [random.uniform(0.1, 1.0) for _ in range(3)] # [w_snr, w_pwr, w_drift]
        }

        # 2. Extract Features with these params (FAST - only harmonic detection)
        def extract_feats_from_precomputed(data_list):
            feats = []
            lbls = []
            for item in data_list:
                # Apply CUSTOM params
                candidates = harmonic_detection.detect_harmonics_iterative(item['peaks'], config_params=params)
                f_vec = harmonic_detection.extract_linear_features(candidates)
                feats.append(f_vec)
                lbls.append(item['label'])
            return np.array(feats), np.array(lbls)

        X_train, y_train = extract_feats_from_precomputed(train_data)
        X_val, y_val = extract_feats_from_precomputed(val_data)

        if len(X_train) == 0 or len(X_val) == 0: continue

        X_t = torch.tensor(X_train, dtype=torch.float).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float).unsqueeze(1).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float).to(device)

        # 3. Train Small Linear Model
        # This is just to evaluate if the FEATURES produced by these detection params are good.
        model = models.LinearHarmonicModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            outputs = model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

        # 4. Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_v)
            preds = (outputs > 0.0).float().cpu().numpy().flatten()

        val_f1 = f1_score(y_val, preds, zero_division=0)
        val_rec = recall_score(y_val, preds, zero_division=0)

        print(f"Iter {i+1}: F1={val_f1:.2f} (Rec={val_rec:.2f}) | Params={params}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    print(f"Best Params found: F1={best_f1:.2f}")
    if not best_params: # Fallback if no improvement
         print("No params found (F1=0), using defaults.")
         best_params = {
            'tolerance': config.TOLERANCE,
            'snr_threshold': config.HARMONIC_MIN_SNR,
            'power_threshold': config.HARMONIC_MIN_POWER,
            'weights': config.QUALITY_WEIGHTS
         }
    return best_params

def train_final_models(train_loader, val_loader, pos_weight=None):
    print("Training Final Models...")
    if pos_weight is not None:
        print(f"Using pos_weight: {pos_weight.item():.2f}")

    # LINEAR MODEL
    linear_model = models.LinearHarmonicModel().to(device)
    optimizer = optim.Adam(linear_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training Linear Model...")
    for epoch in range(30): # Increased epochs slightly
        linear_model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = linear_model(batch.linear_x)
            loss = criterion(outputs, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            acc, prec, rec, f1 = evaluate_model(linear_model, val_loader)
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f} | Val F1={f1:.2f}, Rec={rec:.2f}, Acc={acc:.2f}")

    # GNN MODEL
    gnn_model = models.GNNEventDetector().to(device)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)

    print("Training GNN Model...")
    for epoch in range(30):
        gnn_model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = gnn_model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            acc, prec, rec, f1 = evaluate_model(gnn_model, val_loader)
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f} | Val F1={f1:.2f}, Rec={rec:.2f}, Acc={acc:.2f}")

    return linear_model, gnn_model

def main():
    set_seed()

    # 1. Load Dataset (This uses DEFAULT params for initial loading)
    # Note: We need the raw file paths for Random Search, which dataset provides.
    # The 'cache=True' in HarmonicDataset will cache the DEFAULT features.
    # But Random Search will re-process audio with NEW params.
    # This is fine. The cached features are used for the FINAL training.

    print("Loading Datasets...")
    full_dataset = dataset.HarmonicDataset(mode='train', split_ratio=0.8, cache=True)
    if len(full_dataset) == 0:
        print("Dataset is empty. Run generate_data.py first.")
        return

    # 2. Hyperparameter Optimization
    # We pass the full_dataset object, but the function extracts raw audio paths from it
    # and processes a subset manually.
    best_params = random_search_optimization(full_dataset, num_iterations=10)

    # Save Best Params
    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/best_config.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("Saved best_config.json")

    # 3. Apply Best Params to Global Config
    print("Applying optimized parameters...")
    if 'tolerance' in best_params: config.TOLERANCE = best_params['tolerance']
    if 'snr_threshold' in best_params: config.HARMONIC_MIN_SNR = best_params['snr_threshold']
    if 'power_threshold' in best_params: config.HARMONIC_MIN_POWER = best_params['power_threshold']
    if 'weights' in best_params: config.QUALITY_WEIGHTS = best_params['weights']

    # 4. RE-LOAD Dataset with NEW params?
    # Since HarmonicDataset processes features in __getitem__ (or _load_item) based on global config,
    # and we just changed global config, we MUST clear the cache or re-create the dataset
    # to ensure the final training uses the OPTIMIZED features.

    print("Reloading dataset with optimized parameters...")
    full_dataset = dataset.HarmonicDataset(mode='train', split_ratio=0.8, cache=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Calculate pos_weight for imbalanced learning
    pos_weight = calculate_pos_weight(train_dataset)

    # 5. Prepare DataLoaders
    train_loader = GeoDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 6. Train Final Models
    linear_model, gnn_model = train_final_models(train_loader, val_loader, pos_weight=pos_weight)

    # Save Models
    torch.save(linear_model.state_dict(), 'models/linear_model.pth')
    torch.save(gnn_model.state_dict(), 'models/gnn_model.pth')
    print("Saved models to models/")

if __name__ == "__main__":
    main()
