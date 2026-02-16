import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import argparse

from src import config, signal_processing, harmonic_detection, models

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
FILENAME = 'data/yes/sample_000.wav' # Default file
WINDOW_DURATION = config.WINDOW_DURATION
STEP_SIZE = config.STEP_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    # Load Config
    if os.path.exists('models/best_config.json'):
        with open('models/best_config.json', 'r') as f:
            params = json.load(f)
            print(f"Loaded optimized params: {params}")
            if 'tolerance' in params: config.TOLERANCE = params['tolerance']
            if 'snr_threshold' in params: config.HARMONIC_MIN_SNR = params['snr_threshold']
            if 'power_threshold' in params: config.HARMONIC_MIN_POWER = params['power_threshold']
            if 'weights' in params: config.QUALITY_WEIGHTS = params['weights']
    else:
        print("Using default config.")

    # Load Models
    linear_model = models.LinearHarmonicModel().to(device)
    gnn_model = models.GNNEventDetector().to(device)

    if os.path.exists('models/linear_model.pth'):
        try:
            linear_model.load_state_dict(torch.load('models/linear_model.pth', map_location=device))
            linear_model.eval()
        except RuntimeError as e:
            print(f"Warning: Could not load linear_model.pth (architecture mismatch?): {e}")
    else:
        print("Warning: linear_model.pth not found")

    if os.path.exists('models/gnn_model.pth'):
        try:
            gnn_model.load_state_dict(torch.load('models/gnn_model.pth', map_location=device))
            gnn_model.eval()
        except RuntimeError as e:
            print(f"Warning: Could not load gnn_model.pth (architecture mismatch?): {e}")
    else:
        print("Warning: gnn_model.pth not found")

    return linear_model, gnn_model

def run_analysis(filename=None, output_image='analysis_result.png'):
    if filename is None:
        filename = FILENAME

    if not os.path.exists(filename):
        # Try finding any wav file in data/yes
        if os.path.exists('data/yes'):
            files = [f for f in os.listdir('data/yes') if f.endswith('.wav')]
            if files:
                filename = os.path.join('data/yes', files[0])
            else:
                print("No wav files found.")
                return
        else:
             print(f"File {filename} not found.")
             return

    print(f"Analyzing {filename}...")
    linear_model, gnn_model = load_models()

    # Load Audio
    fs, audio = wavfile.read(filename)
    if len(audio.shape) > 1: audio = audio[:, 0]
    audio = audio.astype(np.float32)
    # Normalize
    audio /= np.max(np.abs(audio))

    duration = len(audio) / fs
    if duration < WINDOW_DURATION:
        print(f"Warning: File duration {duration:.2f}s is shorter than window {WINDOW_DURATION}s.")
        return

    frames = int((duration - WINDOW_DURATION) / STEP_SIZE) + 1

    history = {
        't': [],
        'p_base': [],
        'p_lin': [],
        'p_gnn': [],
        'best_harmonics': [] # List of best harmonic series per frame
    }

    print(f"Processing {frames} frames...")

    for frame in range(frames):
        start_t = frame * STEP_SIZE
        end_t = start_t + WINDOW_DURATION

        # Slice Audio
        s_idx, e_idx = int(start_t*fs), int(end_t*fs)
        slice_audio = audio[s_idx:e_idx]
        if len(slice_audio) < 2048: continue

        # 1. Signal Processing
        f, psd_db = signal_processing.compute_psd(slice_audio, fs)
        nf = signal_processing.estimate_noise_floor(psd_db)
        peaks = signal_processing.find_significant_peaks(f, psd_db, nf)

        # 2. Harmonic Detection
        candidates = harmonic_detection.detect_harmonics_iterative(peaks)
        best_cand = candidates[0] if candidates else None

        # 3. Model Inference
        # Baseline Score (Normalized)
        score_base = 0.0
        if best_cand:
            score_base = min(best_cand['score'] / 5.0, 1.0)

        # Linear Model
        feat_vec = harmonic_detection.extract_linear_features(candidates)
        feat_tensor = torch.tensor(feat_vec, dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_lin = torch.sigmoid(linear_model(feat_tensor)).item()

        # GNN Model
        node_feats = []
        if not peaks:
            node_feats = [[0.0, 0.0, 0.0]]
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            for p in peaks:
                f_norm = p['freq'] / config.MAX_FREQ
                snr_norm = min(max(p['snr'], 0), 50) / 50.0
                pwr_norm = min(max(p['power'] + 100, 0), 100) / 100.0
                node_feats.append([f_norm, snr_norm, pwr_norm])

            edge_index = []
            num_nodes = len(node_feats)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j: edge_index.append([i, j])
            if not edge_index: edge_index = [[0], [0]]
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        x = torch.tensor(node_feats, dtype=torch.float).to(device)
        edge_index = edge_index.to(device)
        batch = torch.zeros(len(node_feats), dtype=torch.long).to(device)

        with torch.no_grad():
            prob_gnn = torch.sigmoid(gnn_model(x, edge_index, batch)).item()

        # Update History
        history['t'].append(start_t + WINDOW_DURATION/2)
        history['p_base'].append(score_base)
        history['p_lin'].append(prob_lin)
        history['p_gnn'].append(prob_gnn)

        # Store best harmonic info for visualization later (e.g., specific frame)
        history['best_harmonics'].append(best_cand)

    # --- Generate Static Report ---
    print("Generating report...")
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2)

    # 1. Spectrogram
    ax_spec = fig.add_subplot(gs[0, :])
    f_spec, t_spec, Sxx = signal.spectrogram(audio, fs, nperseg=1024)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    ax_spec.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='inferno')
    ax_spec.set_title(f"Spectrogram: {os.path.basename(filename)}")
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_xlabel("Time (s)")

    # 2. Probability History
    ax_prob = fig.add_subplot(gs[1, :])
    ax_prob.plot(history['t'], history['p_base'], label='Baseline', color='green', alpha=0.7)
    ax_prob.plot(history['t'], history['p_lin'], label='Linear Model', color='blue', alpha=0.7)
    ax_prob.plot(history['t'], history['p_gnn'], label='GNN Model', color='red', alpha=0.7)
    ax_prob.set_title("Event Probability over Time")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim(-0.1, 1.1)
    ax_prob.legend()
    ax_prob.grid(True, alpha=0.3)

    # 3. Snapshot of Peak Detection (Middle of the file or highest probability frame)
    # Find frame with max GNN probability
    best_frame_idx = np.argmax(history['p_gnn']) if history['p_gnn'] else 0
    t_snapshot = history['t'][best_frame_idx]

    # Re-process that frame to get PSD
    start_t = t_snapshot - WINDOW_DURATION/2
    end_t = start_t + WINDOW_DURATION
    s_idx, e_idx = int(start_t*fs), int(end_t*fs)
    slice_audio = audio[s_idx:e_idx]
    f_snap, psd_snap = signal_processing.compute_psd(slice_audio, fs)
    nf_snap = signal_processing.estimate_noise_floor(psd_snap)

    ax_psd = fig.add_subplot(gs[2, 0])
    ax_psd.plot(f_snap, psd_snap, color='navy', lw=1, label='PSD')
    ax_psd.plot(f_snap, nf_snap, color='gray', linestyle='--', label='Noise Floor')

    # Get stored candidate for this frame
    cand = history['best_harmonics'][best_frame_idx]
    if cand:
        h_f = [h['freq'] for h in cand['harmonics']]
        h_p = [h.get('peak_power', h['power']) for h in cand['harmonics']]
        ax_psd.scatter(h_f, h_p, color='lime', marker='*', s=100, zorder=5, label='Harmonics')

    ax_psd.set_title(f"Snapshot at t={t_snapshot:.2f}s (Max Prob)")
    ax_psd.set_xlim(0, config.MAX_FREQ)
    ax_psd.legend()

    # 4. Text Summary
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis('off')
    summary_text = (
        f"File: {os.path.basename(filename)}\n"
        f"Duration: {duration:.2f}s\n\n"
        f"Max Prob (GNN): {max(history['p_gnn']):.4f}\n"
        f"Max Prob (Linear): {max(history['p_lin']):.4f}\n"
        f"Max Prob (Base): {max(history['p_base']):.4f}\n\n"
        f"Avg Prob (GNN): {np.mean(history['p_gnn']):.4f}\n"
    )
    ax_text.text(0.1, 0.5, summary_text, fontsize=12, va='center')

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Analysis saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to wav file')
    parser.add_argument('--output', type=str, default='analysis_result.png', help='Output image path')
    args = parser.parse_args()

    run_analysis(args.file, args.output)
