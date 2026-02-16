import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from scipy import signal
import argparse

from src import config, signal_processing, harmonic_detection, models, feature_extraction

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
FILENAME = 'data/yes/sample_000.wav' # Default file
WINDOW_DURATION = config.WINDOW_DURATION
STEP_SIZE = config.STEP_SIZE
REFRESH_INTERVAL = 100 # ms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    # Load Config
    if os.path.exists('models/best_config.json'):
        with open('models/best_config.json', 'r') as f:
            try:
                params = json.load(f)
                print(f"Loaded optimized params: {params}")
                if 'tolerance' in params: config.TOLERANCE = params['tolerance']
                if 'snr_threshold' in params: config.HARMONIC_MIN_SNR = params['snr_threshold']
                if 'power_threshold' in params: config.HARMONIC_MIN_POWER = params['power_threshold']
                if 'weights' in params: config.QUALITY_WEIGHTS = params['weights']
            except json.JSONDecodeError:
                print("Warning: best_config.json is corrupted.")
    else:
        print("Using default config.")

    # Load Models
    linear_model = models.LinearHarmonicModel().to(device)
    gnn_model = models.GNNEventDetector().to(device)

    if os.path.exists('models/linear_model.pth'):
        try:
            linear_model.load_state_dict(torch.load('models/linear_model.pth', map_location=device))
            linear_model.eval()
        except Exception as e:
            print(f"Warning: Could not load linear_model.pth: {e}")
    else:
        print("Warning: linear_model.pth not found")

    if os.path.exists('models/gnn_model.pth'):
        try:
            gnn_model.load_state_dict(torch.load('models/gnn_model.pth', map_location=device))
            gnn_model.eval()
        except Exception as e:
            print(f"Warning: Could not load gnn_model.pth: {e}")
    else:
        print("Warning: gnn_model.pth not found")

    return linear_model, gnn_model

def run_analysis(filename=None):
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
    try:
        fs, audio = wavfile.read(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if len(audio.shape) > 1: audio = audio[:, 0]
    audio = audio.astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    duration = len(audio) / fs
    if duration < WINDOW_DURATION:
        print(f"Warning: File duration {duration:.2f}s is shorter than window {WINDOW_DURATION}s.")
        return

    # Pre-calc Spectrogram for background
    f_spec, t_spec, Sxx = signal.spectrogram(audio, fs, nperseg=1024)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # --- Setup Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if hasattr(fig.canvas, 'manager'):
        fig.canvas.manager.set_window_title('Harmonic Event Detector')
    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    # 1. Probability History Plot
    ax_prob = axes[0, 0]
    line_prob_base, = ax_prob.plot([], [], color='green', lw=2, label='Baseline (Heuristic)')
    line_prob_lin, = ax_prob.plot([], [], color='blue', lw=2, label='Linear Model')
    line_prob_gnn, = ax_prob.plot([], [], color='red', lw=2, label='GNN Model')

    ax_prob.set_title("Event Probability (Persistent Tracks)", fontweight='bold')
    ax_prob.set_ylim(-0.1, 1.1)
    ax_prob.set_xlim(0, duration)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Time (s)")
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(loc='upper right')

    # 2. Spectrogram
    ax_spec = axes[0, 1]
    ax_spec.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='inferno')
    cursor_line = ax_spec.axvline(x=0, color='cyan', linestyle='--')
    ax_spec.set_title("Spectrogram", fontweight='bold')
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_ylim(0, config.MAX_FREQ * 1.1) # Show slightly more than max freq

    # 3. Instantaneous PSD & Detections
    ax_psd = axes[1, 0]
    line_psd, = ax_psd.plot([], [], color='navy', lw=1.5, label='Instantaneous PSD')
    line_nf, = ax_psd.plot([], [], color='gray', linestyle='--', alpha=0.7, label='Noise Floor')
    scatter_peaks = ax_psd.scatter([], [], color='red', s=50, zorder=5, label='Peaks')
    scatter_harmonics = ax_psd.scatter([], [], color='lime', marker='*', s=100, zorder=6, label='Tracked Harmonics')

    ax_psd.set_title("Instantaneous PSD & Detections", fontweight='bold')
    ax_psd.set_xlim(0, config.MAX_FREQ)
    ax_psd.set_ylim(-100, 0)
    ax_psd.grid(alpha=0.3)
    ax_psd.legend(loc='upper right')

    # 4. LIVE TABLE
    ax_table = axes[1, 1]
    ax_table.axis('off')
    ax_table.set_title(f"Harmonic Analysis (Best Candidate)", fontweight='bold')

    col_labels = ["Harmonic", "Freq (Hz)", "Power (dB)", "SNR (dB)"]
    table_data = [["-", "-", "-", "-"] for _ in range(10)]
    the_table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1, 1.5)

    # State
    history = {
        't': [],
        'p_base': [],
        'p_lin': [],
        'p_gnn': []
    }

    frames = int((duration - WINDOW_DURATION) / STEP_SIZE) + 1

    def init():
        return line_prob_base, line_prob_lin, line_prob_gnn, line_psd, scatter_peaks, cursor_line

    def update(frame):
        start_t = frame * STEP_SIZE
        end_t = start_t + WINDOW_DURATION
        if end_t > duration: return init()

        # Slice Audio
        s_idx, e_idx = int(start_t*fs), int(end_t*fs)
        slice_audio = audio[s_idx:e_idx]
        if len(slice_audio) < 2048: return init()

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
            # Simple normalization of score (heuristic)
            score_base = min(best_cand['score'] / 5.0, 1.0)

        # Linear Model
        feat_vec = harmonic_detection.extract_linear_features(candidates)
        feat_tensor = torch.tensor(feat_vec, dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            # Apply sigmoid since model outputs logits
            prob_lin = torch.sigmoid(linear_model(feat_tensor)).item()

        # GNN Model
        # Construct graph
        gnn_data = feature_extraction.build_gnn_data(peaks)
        x = gnn_data.x.to(device)
        edge_index = gnn_data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

        with torch.no_grad():
            # Apply sigmoid since model outputs logits
            prob_gnn = torch.sigmoid(gnn_model(x, edge_index, batch)).item()

        # Update History
        history['t'].append(start_t + WINDOW_DURATION/2)
        history['p_base'].append(score_base)
        history['p_lin'].append(prob_lin)
        history['p_gnn'].append(prob_gnn)

        # Update Plots
        line_prob_base.set_data(history['t'], history['p_base'])
        line_prob_lin.set_data(history['t'], history['p_lin'])
        line_prob_gnn.set_data(history['t'], history['p_gnn'])

        cursor_line.set_xdata([start_t + WINDOW_DURATION/2])

        line_psd.set_data(f, psd_db)
        line_nf.set_data(f, nf)

        if peaks:
            p_f = [p['freq'] for p in peaks]
            p_p = [p.get('peak_power', p['power']) for p in peaks]
            scatter_peaks.set_offsets(np.c_[p_f, p_p])
        else:
            scatter_peaks.set_offsets(np.empty((0, 2)))

        if best_cand:
            h_f = [h['freq'] for h in best_cand['harmonics']]
            h_p = [h.get('peak_power', h['power']) for h in best_cand['harmonics']]
            scatter_harmonics.set_offsets(np.c_[h_f, h_p])

            # Update Table
            for i, h in enumerate(best_cand['harmonics']):
                if i >= 10: break
                the_table[i+1, 0].get_text().set_text(f"#{h['harmonic_index']}")
                the_table[i+1, 1].get_text().set_text(f"{h['freq']:.1f}")
                the_table[i+1, 2].get_text().set_text(f"{h['power']:.1f}")
                the_table[i+1, 3].get_text().set_text(f"{h['snr']:.1f}")

            # Clear rest of table
            for k in range(len(best_cand['harmonics']), 10):
                for c in range(4): the_table[k+1, c].get_text().set_text("-")
        else:
            scatter_harmonics.set_offsets(np.empty((0, 2)))
            for k in range(10):
                for c in range(4): the_table[k+1, c].get_text().set_text("-")

        return line_prob_base, line_prob_lin, line_prob_gnn, line_psd, scatter_peaks, cursor_line

    anim = FuncAnimation(fig, update, frames=range(frames),
                         init_func=init, blit=False, interval=REFRESH_INTERVAL, repeat=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to wav file')
    args = parser.parse_args()

    run_analysis(args.file)
