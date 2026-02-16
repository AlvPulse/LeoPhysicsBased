import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from src import config, signal_processing, harmonic_detection

class HarmonicData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'linear_x':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'linear_x':
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

class HarmonicDataset(Dataset):
    def __init__(self, data_dir='data', mode='train', split_ratio=0.8, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.file_list = []
        self.labels = []

        # Load file paths
        yes_dir = os.path.join(data_dir, 'yes')
        no_dir = os.path.join(data_dir, 'no')

        yes_files = [os.path.join(yes_dir, f) for f in os.listdir(yes_dir) if f.endswith('.wav')]
        no_files = [os.path.join(no_dir, f) for f in os.listdir(no_dir) if f.endswith('.wav')]

        all_files = yes_files + no_files
        all_labels = [1] * len(yes_files) + [0] * len(no_files)

        # Shuffle
        combined = list(zip(all_files, all_labels))
        np.random.seed(42) # Reproducibility
        np.random.shuffle(combined)

        # Split
        split_idx = int(len(combined) * split_ratio)
        if mode == 'train':
            data = combined[:split_idx]
        else:
            data = combined[split_idx:]

        self.file_list, self.labels = zip(*data)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        label = self.labels[idx]

        # 1. Load Audio
        audio, fs = signal_processing.load_audio(filepath)
        if audio is None:
            return None

        # 2. Compute Spectrogram & Peaks (Take average or max over time?)
        # For simple classification, let's take the frame with max energy
        # Or just compute PSD over the whole file (Welch)
        f, psd_db = signal_processing.compute_psd(audio, fs)
        nf = signal_processing.estimate_noise_floor(psd_db)
        peaks = signal_processing.find_significant_peaks(f, psd_db, nf)

        # 3. Harmonic Detection (For Linear Model Features)
        candidates = harmonic_detection.detect_harmonics_iterative(peaks)
        linear_features = harmonic_detection.extract_linear_features(candidates)

        # 4. Graph Construction (For GNN)
        # Nodes: [freq_norm, snr_norm, pwr_norm]
        node_features = []
        for p in peaks:
            f_norm = p['freq'] / config.MAX_FREQ
            snr_norm = min(max(p['snr'], 0), 50) / 50.0
            pwr_norm = min(max(p['power'] + 100, 0), 100) / 100.0
            node_features.append([f_norm, snr_norm, pwr_norm])

        if not node_features:
            # Handle empty peaks case (silence)
            # Create a dummy node to avoid GNN crash
            node_features = [[0.0, 0.0, 0.0]]
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            # Fully connected graph
            num_nodes = len(node_features)
            edge_index = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index.append([i, j])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            if edge_index.numel() == 0: # Single node case resulted in no edges
                 edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.float)

        # Linear Features as Tensor
        # Add batch dimension so PyG treats it as a graph-level attribute
        linear_x = torch.tensor(linear_features, dtype=torch.float).unsqueeze(0)

        return HarmonicData(x=x, edge_index=edge_index, y=y, linear_x=linear_x)
