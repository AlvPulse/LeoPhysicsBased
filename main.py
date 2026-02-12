import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch
from matplotlib.animation import FuncAnimation
import os
import sys
from torch_geometric.data import Data, Batch

# Import modules
from src import config, signal_processing, harmonic_detection, baseline_model
from src.models import LinearHarmonicModel, GNNEventDetector

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Default file
FILENAME = 'data/yes/Autel_Evo_II_20.wav'

# Allow file selection via CLI args if provided
if len(sys.argv) > 1:
    FILENAME = sys.argv[1]

WINDOW_DURATION = config.WINDOW_DURATION
STEP_SIZE = config.STEP_SIZE
REFRESH_INTERVAL = 100

def run_analysis():
    print(f"Analyzing {FILENAME}...")
    if not os.path.exists(FILENAME):
        print(f"File not found: {FILENAME}")
        return

    # Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        linear_model = LinearHarmonicModel()
        linear_model.load_state_dict(torch.load('models/linear_model.pth', map_location=device))
        linear_model.to(device)
        linear_model.eval()

        gnn_model = GNNEventDetector()
        gnn_model.load_state_dict(torch.load('models/gnn_model.pth', map_location=device))
        gnn_model.to(device)
        gnn_model.eval()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Ensure you ran train.py.")
        return

    # Load Audio
    audio, fs = signal_processing.load_audio(FILENAME)
    if audio is None: return

    duration = len(audio) / fs

    # Pre-calc Spectrogram for visualization
    f_spec, t_spec, Sxx = signal_processing.signal.spectrogram(audio, fs, nperseg=config.N_FFT)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # --- Setup Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.canvas.manager.set_window_title('Harmonic Event Detector - Multi-Model Comparison')
    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    # 1. Probability History Plot
    ax_prob = axes[0, 0]
    line_prob1, = ax_prob.plot([], [], color='green', lw=2, label='Baseline (Heuristic)')
    line_prob2, = ax_prob.plot([], [], color='blue', lw=2, label='Linear Model')
    line_prob3, = ax_prob.plot([], [], color='red', lw=2, label='GNN Model')

    ax_prob.set_title("Event Probability", fontweight='bold')
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
    ax_spec.set_ylim(0, config.MAX_FREQ * 1.2)

    # 3. PSD Curve
    ax_psd = axes[1, 0]
    line_psd, = ax_psd.plot([], [], color='navy', lw=1.5, label='PSD')
    line_nf, = ax_psd.plot([], [], color='gray', linestyle='--', alpha=0.7, label='Noise Floor')
    scatter_peaks = ax_psd.scatter([], [], color='red', s=50, zorder=5, label='Peaks')
    scatter_harmonics = ax_psd.scatter([], [], color='lime', s=80, marker='*', zorder=10, label='Harmonics (Iterative)')

    ax_psd.set_title("PSD & Detections", fontweight='bold')
    ax_psd.set_xlim(0, config.MAX_FREQ)
    # Set ylim dynamically or fixed? Fixed is safer for animation
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
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)

    # Store history
    history_t = []
    prob1_y = []
    prob2_y = []
    prob3_y = []

    def init():
        line_prob1.set_data([], [])
        line_prob2.set_data([], [])
        line_prob3.set_data([], [])
        line_psd.set_data([], [])
        line_nf.set_data([], [])
        scatter_peaks.set_offsets(np.empty((0, 2)))
        scatter_harmonics.set_offsets(np.empty((0, 2)))
        cursor_line.set_xdata([0])
        return line_prob1, line_prob2, line_prob3, line_psd, line_nf, scatter_peaks, scatter_harmonics, cursor_line

    def update(frame):
        start_t = frame * STEP_SIZE
        end_t = start_t + WINDOW_DURATION
        if end_t > duration: return init()

        # Slice Audio
        s_idx, e_idx = int(start_t*fs), int(end_t*fs)
        slice_audio = audio[s_idx:e_idx]
        if len(slice_audio) < config.N_FFT: return init()

        # Update Cursor
        cursor_line.set_xdata([start_t + WINDOW_DURATION/2])

        # --- PROCESS ---
        f, psd = signal_processing.compute_psd(slice_audio, fs)
        nf = signal_processing.estimate_noise_floor(psd)
        peaks = signal_processing.find_significant_peaks(f, psd, nf)

        # Update PSD Plot
        line_psd.set_data(f, psd)
        line_nf.set_data(f, nf)

        if peaks:
            pf = [p['freq'] for p in peaks]
            pp = [p['power'] for p in peaks]
            scatter_peaks.set_offsets(np.c_[pf, pp])
        else:
            scatter_peaks.set_offsets(np.empty((0, 2)))

        # --- METHOD 1: BASELINE HEURISTIC ---
        score1, base_freq1 = baseline_model.detect_baseline_heuristic(peaks)

        # --- METHOD 2 & 3 PREP: ITERATIVE SEARCH ---
        candidates = harmonic_detection.detect_harmonics_iterative(peaks)

        best_candidate = None
        if candidates:
            best_candidate = candidates[0]

            # Plot Harmonics
            hf = [h['freq'] for h in best_candidate['harmonics']]
            hp = [h['power'] for h in best_candidate['harmonics']]
            scatter_harmonics.set_offsets(np.c_[hf, hp])

            # Update Table
            for r in range(10):
                for c in range(4):
                    the_table[r+1, c].get_text().set_text("-")

            for i, h in enumerate(best_candidate['harmonics']):
                if i >= 10: break
                the_table[i+1, 0].get_text().set_text(f"#{h['harmonic_index']}")
                the_table[i+1, 1].get_text().set_text(f"{h['freq']:.1f}")
                the_table[i+1, 2].get_text().set_text(f"{h['power']:.1f}")
                the_table[i+1, 3].get_text().set_text(f"{h['snr']:.1f}")
        else:
            scatter_harmonics.set_offsets(np.empty((0, 2)))
            for r in range(10):
                for c in range(4):
                    the_table[r+1, c].get_text().set_text("-")

        # --- PREPARE DATA FOR MODELS ---
        if not peaks:
             x = torch.zeros((1, 3), dtype=torch.float)
             edge_index = torch.zeros((2, 0), dtype=torch.long)
             linear_vec = np.zeros(20, dtype=np.float32)
        else:
            freqs = np.array([p['freq'] for p in peaks])
            powers = np.array([p['power'] for p in peaks])
            snrs = np.array([p['snr'] for p in peaks])

            f_norm = freqs / config.MAX_FREQ
            p_norm = (powers + 100) / 100
            s_norm = snrs / 50

            x = torch.tensor(np.column_stack((f_norm, p_norm, s_norm)), dtype=torch.float)

            edge_src = []
            edge_dst = []
            for i in range(len(peaks)):
                for j in range(len(peaks)):
                    if i == j: continue
                    fi = peaks[i]['freq']
                    fj = peaks[j]['freq']
                    if fi >= fj: continue
                    ratio = fj / fi
                    harmonic_idx = round(ratio)
                    drift = abs(ratio - harmonic_idx)
                    if harmonic_idx > 1 and drift < config.TOLERANCE:
                        edge_src.append(i)
                        edge_dst.append(j)

            if edge_src:
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Use shared feature extraction logic
            linear_vec = harmonic_detection.extract_linear_features(candidates)

        # --- METHOD 2: LINEAR ---
        with torch.no_grad():
            lin_input = torch.tensor(linear_vec, dtype=torch.float).unsqueeze(0).to(device)
            score2 = linear_model(lin_input).item()

        # --- METHOD 3: GNN ---
        with torch.no_grad():
            gnn_batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device)
            score3 = gnn_model(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch).item()

        # Update Plots
        history_t.append(start_t)
        prob1_y.append(score1)
        prob2_y.append(score2)
        prob3_y.append(score3)

        line_prob1.set_data(history_t, prob1_y)
        line_prob2.set_data(history_t, prob2_y)
        line_prob3.set_data(history_t, prob3_y)

        print(f"Time: {start_t:.1f}s | Scores -> Baseline: {score1:.2f}, Linear: {score2:.2f}, GNN: {score3:.2f}")

        return line_prob1, line_prob2, line_prob3, line_psd, line_nf, scatter_peaks, scatter_harmonics, cursor_line, the_table

    print(f"Starting Analysis on {FILENAME}")
    anim = FuncAnimation(fig, update, frames=range(int((duration-WINDOW_DURATION)/STEP_SIZE)),
                         init_func=init, blit=False, interval=REFRESH_INTERVAL)
    plt.show()

if __name__ == "__main__":
    run_analysis()
