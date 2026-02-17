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

def extract_classifier_features(track, num_harmonics=10):
    """
    Extracts a comprehensive feature vector for classifier models (e.g. XGBoost/GradientBoosting).
    
    Args:
        track (dict): The track dictionary containing persistence info and best_candidate.
        num_harmonics (int): Number of harmonics to include features for.

    Returns:
        np.array: Feature vector of shape (11 + num_harmonics * 2,)
    """
    # Base features
    # 0. Base Frequency (Normalized)
    # 1. Max Score
    # 2. Persistence
    # 3. Harmonic Count
    # 4. Average Drift
    
    # Spectral Features (New)
    # 5. Centroid
    # 6. Flux
    # 7. Flatness
    # 8. RMS
    # 9. PAPR
    # 10. Delta RMS

    # Harmonic Features (11...)
    
    feature_dim = 11 + (num_harmonics * 2)
    vec = np.zeros(feature_dim, dtype=np.float32)

    if not track:
        return vec
    
    best_candidate = track.get('best_candidate', {})
    if not best_candidate:
        # Fallback if track is actually a candidate itself (legacy support)
        if 'harmonics' in track:
            best_candidate = track
        else:
            return vec

    # Global Track Features
    base_freq = track.get('freq', best_candidate.get('base_freq', 0.0))
    vec[0] = base_freq / config.MAX_FREQ

    vec[1] = track.get('max_score', best_candidate.get('score', 0.0))
    vec[2] = track.get('persistence', 0) # 0 if just a candidate
    
    harmonics = best_candidate.get('harmonics', [])
    vec[3] = len(harmonics)
    
    avg_drift = 0.0
    if harmonics:
        avg_drift = np.mean([h.get('drift', 0.0) for h in harmonics])
    vec[4] = avg_drift

    # Spectral Features from 'best_frame_features'
    spectral = track.get('best_frame_features', {})
    
    # 5. Centroid (Normalized)
    vec[5] = spectral.get('centroid', 0.0) / config.MAX_FREQ
    
    # 6. Flux
    vec[6] = spectral.get('flux', 0.0)
    
    # 7. Flatness
    vec[7] = spectral.get('flatness', 0.0)
    
    # 8. RMS
    vec[8] = spectral.get('rms', 0.0)

    # 9. PAPR
    vec[9] = spectral.get('papr', 0.0)

    # 10. Delta RMS
    vec[10] = spectral.get('delta_rms', 0.0)

    # Harmonic Specific Features (SNR, Power)
    # Harmonics are typically 1-indexed (Fundamental is 1)
    # We map harmonic index i to vector index 11 + (i-1)*2
    for h in harmonics:
        idx = h.get('harmonic_index', 0)
        if 1 <= idx <= num_harmonics:
            vec_idx = 11 + (idx - 1) * 2
            
            # SNR
            snr = h.get('snr', 0.0)
            vec[vec_idx] = min(max(snr, 0), 50) / 50.0
            
            # Power
            pwr = h.get('power', -100.0)
            vec[vec_idx+1] = min(max(pwr + 100, 0), 100) / 100.0
            
    return vec
