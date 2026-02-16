import numpy as np
import torch
from torch_geometric.data import Data
from src import config, harmonic_detection, signal_processing, feature_extraction
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
            return self._get_dummy(label)

        # 1. Compute STFT, Peaks, and Spectral Features
        # Unpack 5 values now
        f, t, Pxx_db, peaks_per_frame, spectral_features = signal_processing.compute_spectrogram_and_peaks(audio, fs)

        # 2. Track Harmonics
        # Pass spectral_features to track_harmonics
        tracks = harmonic_detection.track_harmonics(peaks_per_frame, t, spectral_features)

        # 3. Select Best Track
        if not tracks:
            # If no tracks found, return dummy (empty) data
            return self._get_dummy(label)

        # Sort by score descending (already done by track_harmonics but to be safe)
        tracks.sort(key=lambda x: x['max_score'], reverse=True)
        best_track = tracks[0]

        # 4. Extract Features from Best Candidate (Snapshot)
        # best_track has 'best_candidate', which has 'harmonics'
        best_candidate = best_track.get('best_candidate')
        if not best_candidate:
             return self._get_dummy(label)

        # GNN Data: Build graph from the harmonics of the best candidate
        gnn_data = feature_extraction.build_gnn_data(best_candidate['harmonics'], label=label)

        # Linear Features: Flattened vector from the best candidate
        linear_vec = feature_extraction.extract_linear_features(best_candidate)
        gnn_data.linear_features = torch.tensor(linear_vec, dtype=torch.float).unsqueeze(0)

        # Classifier Features: Comprehensive vector for Gradient Boosting
        classifier_vec = feature_extraction.extract_classifier_features(best_track)
        gnn_data.classifier_features = torch.tensor(classifier_vec, dtype=torch.float).unsqueeze(0)

        self.cache[idx] = gnn_data
        return gnn_data

    def _get_dummy(self, label):
        data = Data(x=torch.zeros((1,3)), edge_index=torch.zeros((2,0)), y=torch.tensor([label], dtype=torch.float))
        data.linear_features = torch.zeros(1, 20)
        # 31 features: 11 global/spectral + 20 harmonic (10*2)
        data.classifier_features = torch.zeros(1, 31)
        return data
