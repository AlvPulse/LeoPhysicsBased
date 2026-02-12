import numpy as np
import torch
from torch_geometric.data import Data
from src import config, harmonic_detection, signal_processing
import os
from torch.utils.data import Dataset

class HarmonicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = []
        self.labels = []

        # Load YES samples
        yes_dir = os.path.join(root_dir, 'yes')
        if os.path.exists(yes_dir):
            for f in os.listdir(yes_dir):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(yes_dir, f))
                    self.labels.append(1.0)

        # Load NO samples
        no_dir = os.path.join(root_dir, 'no')
        if os.path.exists(no_dir):
            for f in os.listdir(no_dir):
                if f.endswith('.wav'):
                    self.files.append(os.path.join(no_dir, f))
                    self.labels.append(0.0)

        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        filepath = self.files[idx]
        label = self.labels[idx]

        # Load and process
        audio, fs = signal_processing.load_audio(filepath)
        if audio is None:
            return self._get_dummy()

        f, psd = signal_processing.compute_psd(audio, fs)
        nf = signal_processing.estimate_noise_floor(psd)
        peaks = signal_processing.find_significant_peaks(f, psd, nf)

        # --- Prepare GNN Data ---
        # Nodes: Peaks
        # Features: [freq_norm, power_norm, snr_norm]
        # Edges: Harmonic relations

        if not peaks:
             # Handle empty peaks case
            x = torch.zeros((1, 3), dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            linear_features = torch.zeros(20, dtype=torch.float)
        else:
            # Normalize features for NN
            freqs = np.array([p['freq'] for p in peaks])
            powers = np.array([p['power'] for p in peaks])
            snrs = np.array([p['snr'] for p in peaks])

            # Simple normalization
            f_norm = freqs / config.MAX_FREQ
            p_norm = (powers + 100) / 100 # Approx -100 to 0 -> 0 to 1
            s_norm = snrs / 50 # Approx 0 to 50 -> 0 to 1

            x = torch.tensor(np.column_stack((f_norm, p_norm, s_norm)), dtype=torch.float)

            # Build edges
            edge_src = []
            edge_dst = []

            for i in range(len(peaks)):
                for j in range(len(peaks)):
                    if i == j: continue

                    fi = peaks[i]['freq']
                    fj = peaks[j]['freq']

                    # Ensure fj > fi (harmonic relationship)
                    if fi >= fj: continue

                    ratio = fj / fi
                    harmonic_idx = round(ratio)
                    drift = abs(ratio - harmonic_idx)

                    # Connect if it looks like a harmonic
                    if harmonic_idx > 1 and drift < config.TOLERANCE:
                        edge_src.append(i)
                        edge_dst.append(j)

            if edge_src:
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # --- Prepare Linear Data ---
            candidates = harmonic_detection.detect_harmonics_iterative(peaks)
            linear_vec = harmonic_detection.extract_linear_features(candidates)
            linear_features = torch.tensor(linear_vec, dtype=torch.float)

        gnn_data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
        gnn_data.linear_features = linear_features.unsqueeze(0)

        self.cache[idx] = gnn_data
        return gnn_data

    def _get_dummy(self):
        data = Data(x=torch.zeros((1,3)), edge_index=torch.zeros((2,0)), y=torch.tensor([0.0]))
        data.linear_features = torch.zeros(1, 20)
        return data
