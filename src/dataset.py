import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from src import config, signal_processing, harmonic_detection, feature_extraction
from tqdm import tqdm

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
    def __init__(self, data_dir='data', mode='train', split_ratio=0.8, transform=None, cache=True):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.cache = cache
        self.data_cache = {} # Dict: idx -> Data

        self.file_list = []
        self.labels = []

        # Load file paths
        yes_dir = os.path.join(data_dir, 'yes')
        no_dir = os.path.join(data_dir, 'no')

        # Check if dirs exist
        if not os.path.exists(yes_dir) or not os.path.exists(no_dir):
            print(f"Warning: Data directories not found in {data_dir}")
            return

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

        if not data:
            self.file_list, self.labels = [], []
        else:
            self.file_list, self.labels = zip(*data)

        if self.cache and len(self.file_list) > 0:
            print(f"Pre-loading {len(self.file_list)} files into memory...")
            for i in tqdm(range(len(self.file_list))):
                self._load_item(i)

    def __len__(self):
        return len(self.file_list)

    def _load_item(self, idx):
        if self.cache and idx in self.data_cache:
            return self.data_cache[idx]

        filepath = self.file_list[idx]
        label = self.labels[idx]

        # 1. Load Audio
        audio, fs = signal_processing.load_audio(filepath)
        if audio is None:
            # Fallback for corrupted audio
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
        # Use the consistent graph construction method
        gnn_data = feature_extraction.build_gnn_data(peaks, label)

        # If the graph is empty (no peaks), build_gnn_data returns dummy node/edges
        # handled inside build_gnn_data

        # Linear Features as Tensor
        # Add batch dimension so PyG treats it as a graph-level attribute
        linear_x = torch.tensor(linear_features, dtype=torch.float).unsqueeze(0)

        # Create the custom data object
        data = HarmonicData(
            x=gnn_data.x,
            edge_index=gnn_data.edge_index,
            y=gnn_data.y,
            linear_x=linear_x
        )

        if self.cache:
            self.data_cache[idx] = data

        return data

    def __getitem__(self, idx):
        return self._load_item(idx)
