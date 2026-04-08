import numpy as np
import torch
from torch_geometric.data import Data
from src import config, harmonic_detection

def build_gnn_data(peaks, label=None):
    """
    Constructs a PyTorch Geometric Data object from a list of peaks.
    peaks: List of dicts {'freq': f, 'power': p, 'snr': snr, ...}
    label: Optional label (float) for training.
    """
    if not peaks:
        x = torch.zeros((1, 3), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        freqs = np.array([p['freq'] for p in peaks])
        powers = np.array([p['power'] for p in peaks])
        snrs = np.array([p['snr'] for p in peaks])

        # Normalize features
        f_norm = freqs / config.MAX_FREQ
        # Power is typically negative dB, e.g. -60 to -20. +100 shifts to 40-80. /100 -> 0.4-0.8
        p_norm = (powers + 100) / 100.0
        # SNR is typically 0 to 50 dB. /50 -> 0-1
        s_norm = snrs / 50.0

        x = torch.tensor(np.column_stack((f_norm, p_norm, s_norm)), dtype=torch.float)

        # Build edges based on harmonic relationships
        edge_src = []
        edge_dst = []

        for i in range(len(peaks)):
            for j in range(len(peaks)):
                if i == j: continue

                fi = peaks[i]['freq']
                fj = peaks[j]['freq']

                # Directed edge from lower frequency (source/fundamental) to higher (harmonic)
                if fi >= fj: continue

                ratio = fj / fi
                harmonic_idx = round(ratio)
                drift = abs(ratio - harmonic_idx)

                # Connect if it looks like a harmonic (e.g. 2nd, 3rd, ...)
                if harmonic_idx > 1 and drift < config.TOLERANCE:
                    edge_src.append(i)
                    edge_dst.append(j)

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create Data object
    y_tensor = torch.tensor([label], dtype=torch.float) if label is not None else None
    data = Data(x=x, edge_index=edge_index, y=y_tensor)

    return data

def extract_linear_features(track_or_candidate):
    """
    Wrapper for harmonic_detection.extract_linear_features.
    We might move the logic here eventually, but for now we delegate to keep compatibility.
    """
    return harmonic_detection.extract_linear_features(track_or_candidate)
