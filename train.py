import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from torch_geometric.loader import DataLoader as GeoDataLoader
from src import dataset, harmonic_detection, models, config, signal_processing

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
    correct = 0
    total = 0
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
            correct += (predicted == batch.y.unsqueeze(1)).sum().item()
            total += batch.y.size(0)

    return correct / total if total > 0 else 0.0

def get_subset_file_and_label(dataset_obj, idx):
    """Helper to get file path and label from a Subset or Dataset."""
    if isinstance(dataset_obj, torch.utils.data.Subset):
        real_idx = dataset_obj.indices[idx]
        return dataset_obj.dataset.file_list[real_idx], dataset_obj.dataset.labels[real_idx]
    else:
        return dataset_obj.file_list[idx], dataset_obj.labels[idx]

def random_search_optimization(train_dataset, val_dataset, num_iterations=10):
    best_acc = 0.0
    best_params = {}

    print(f"Starting Random Search Optimization ({num_iterations} iterations)...")

    # Pre-select a fixed subset of indices for validation speed
    train_subset_indices = random.sample(range(len(train_dataset)), min(20, len(train_dataset)))
    val_subset_indices = random.sample(range(len(val_dataset)), min(20, len(val_dataset)))

    for i in range(num_iterations):
        # 1. Sample Hyperparameters
        params = {
            'tolerance': random.uniform(0.05, 0.25),
            'snr_threshold': random.uniform(1.0, 8.0),
            'power_threshold': random.uniform(-80.0, -40.0),
            'weights': [random.uniform(0.1, 1.0) for _ in range(3)] # [w_snr, w_pwr, w_drift]
        }

        # 2. Extract Features with these params
        train_features = []
        train_labels = []

        for idx in train_subset_indices:
            filepath, label = get_subset_file_and_label(train_dataset, idx)

            # Load & Process
            audio, fs = signal_processing.load_audio(filepath)
            if audio is None: continue

            f, psd_db = signal_processing.compute_psd(audio, fs)
            nf = signal_processing.estimate_noise_floor(psd_db)
            peaks = signal_processing.find_significant_peaks(f, psd_db, nf)

            # Use CUSTOM params
            candidates = harmonic_detection.detect_harmonics_iterative(peaks, config_params=params)
            feats = harmonic_detection.extract_linear_features(candidates)

            train_features.append(feats)
            train_labels.append(label)

        if not train_features: continue

        X = torch.tensor(np.array(train_features), dtype=torch.float).to(device)
        y = torch.tensor(np.array(train_labels), dtype=torch.float).unsqueeze(1).to(device)

        # 3. Train Small Linear Model
        model = models.LinearHarmonicModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # Use BCEWithLogitsLoss
        pos_weight = None # For small subset optimization, keep simple
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(20): # Increased epochs for better convergence
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 4. Evaluate on Validation Subset
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
             for idx in val_subset_indices:
                filepath, label = get_subset_file_and_label(val_dataset, idx)

                audio, fs = signal_processing.load_audio(filepath)
                if audio is None: continue
                f, psd_db = signal_processing.compute_psd(audio, fs)
                nf = signal_processing.estimate_noise_floor(psd_db)
                peaks = signal_processing.find_significant_peaks(f, psd_db, nf)

                candidates = harmonic_detection.detect_harmonics_iterative(peaks, config_params=params)
                feats = harmonic_detection.extract_linear_features(candidates)

                input_tensor = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)
                pred_logits = model(input_tensor)
                predicted = (pred_logits > 0.0).float().item()
                if predicted == label:
                    val_correct += 1
                val_total += 1

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Iter {i+1}: Acc={val_acc:.2f} | Params={params}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params

    print(f"Best Params found: Acc={best_acc:.2f}")
    if not best_params: # Fallback if no improvement
         print("No params found (acc=0), using defaults.")
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
    for epoch in range(20):
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

        val_acc = evaluate_model(linear_model, val_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Val Acc={val_acc:.2f}")

    # GNN MODEL
    gnn_model = models.GNNEventDetector().to(device)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)

    print("Training GNN Model...")
    for epoch in range(20):
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

        val_acc = evaluate_model(gnn_model, val_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Val Acc={val_acc:.2f}")

    return linear_model, gnn_model

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

def main():
    set_seed()

    # 1. Load Dataset
    print("Loading Datasets...")
    full_dataset = dataset.HarmonicDataset(mode='train', split_ratio=0.8)
    if len(full_dataset) == 0:
        print("Dataset is empty. Run generate_data.py first.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Calculate pos_weight for imbalanced learning
    pos_weight = calculate_pos_weight(train_dataset)

    # 2. Hyperparameter Optimization
    best_params = random_search_optimization(train_dataset, val_dataset, num_iterations=10)

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

    # 4. Prepare DataLoaders
    train_loader = GeoDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 5. Train Final Models
    linear_model, gnn_model = train_final_models(train_loader, val_loader, pos_weight=pos_weight)

    # Save Models
    torch.save(linear_model.state_dict(), 'models/linear_model.pth')
    torch.save(gnn_model.state_dict(), 'models/gnn_model.pth')
    print("Saved models to models/")

if __name__ == "__main__":
    main()
