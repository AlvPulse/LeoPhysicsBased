import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
import os
import argparse
from torch_geometric.data import Data, Batch

# Import modules
from src import config, signal_processing, harmonic_detection, baseline_model, feature_extraction
from src.models import LinearHarmonicModel, GNNEventDetector


# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

# Default file
TEST_FILENAME = 'data/yes/DJI_Tello_TT_93.wav'



WINDOW_DURATION = config.WINDOW_DURATION
STEP_SIZE = config.STEP_SIZE
REFRESH_INTERVAL = 100



def run_analysis():
    parser = argparse.ArgumentParser(description='Harmonic Event Detector - Visualization')
    parser.add_argument('filename', nargs='?', default=TEST_FILENAME, help='Path to WAV file')


    args = parser.parse_args()
    FILENAME = args.filename

    if not os.path.exists(FILENAME):
        print(f"File not found: {FILENAME}")
        return

    print(f"Analyzing {FILENAME}...")

    # Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        linear_model = LinearHarmonicModel()
        linear_model.load_state_dict(torch.load('models/linear_model.pth', map_location=device))
        linear_model.to(device)
        linear_model.eval()

    #     gnn_model = GNNEventDetector()
    #     gnn_model.load_state_dict(torch.load('models/gnn_model.pth', map_location=device))
    #     gnn_model.to(device)
    #     gnn_model.eval()
    #     print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Ensure you ran train.py.")
        return

    # Load Audio
    audio, fs = signal_processing.load_audio(FILENAME)
    if audio is None: return

    duration = len(audio) / fs

    # 1. Compute STFT and Peaks (Full File)
    print("Computing STFT and Peaks...")
    f, t, Pxx_db, peaks_per_frame = signal_processing.compute_spectrogram_and_peaks(audio, fs)

    # 2. Track Harmonics (Persistence)
    print("Tracking Harmonics...")
    tracks = harmonic_detection.track_harmonics(peaks_per_frame, t)

    # 3. Calculate Probabilities Time Series
    num_frames = len(t)
    prob_baseline = np.zeros(num_frames)
    prob_linear = np.zeros(num_frames)
    prob_gnn = np.zeros(num_frames)

    # Map tracks to time series
    # We only visualize the BEST track's probability for simplicity,
    # or we could sum/max them if multiple exist (though rare with current logic).
    if tracks:
        # Sort tracks by score (descending)
        tracks.sort(key=lambda x: x['max_score'], reverse=True)
        best_track = tracks[0]
        start_frame = best_track['start_frame']
        end_frame = best_track['last_seen']

        # Get peaks from best snapshot
        best_frame_idx = best_track.get('best_frame_idx', 0)
        best_peaks = peaks_per_frame[best_frame_idx] if best_frame_idx < len(peaks_per_frame) else []

        # Calculate scores
        score1, _ = baseline_model.detect_baseline_heuristic(best_peaks)

        # Retrieve Best Candidate Snapshot
        best_candidate = best_track.get('best_candidate')

        score2 = 0.0
        score3 = 0.0

        if best_candidate:
            # Linear Model
            linear_vec = feature_extraction.extract_linear_features(best_candidate)
            with torch.no_grad():
                lin_input = torch.tensor(linear_vec, dtype=torch.float).unsqueeze(0).to(device)
                score2 = torch.sigmoid(linear_model(lin_input)).item()

            # GNN Model
            # gnn_data = feature_extraction.build_gnn_data(best_candidate['harmonics'])
            # with torch.no_grad():
            #     gnn_batch = Batch.from_data_list([gnn_data]).to(device)
            #     score3 = torch.sigmoid(gnn_model(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch)).item()

        # Assign to time series
        # We fill from start to end of the track
        prob_baseline[start_frame:end_frame+1] = score1
        prob_linear[start_frame:end_frame+1] = score2
        prob_gnn[start_frame:end_frame+1] = score3

        print(f"Best Track Found: {best_track['freq']:.1f}Hz, Frames {start_frame}-{end_frame}, Score: {best_track['max_score']:.2f}")

    # --- Setup Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.canvas.manager.set_window_title('Harmonic Event Detector - Multi-Model Comparison')
    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    # 1. Probability History Plot
    ax_prob = axes[0, 0]
    line_prob1, = ax_prob.plot([], [], color='green', lw=2, label='Baseline (Heuristic)')
    line_prob2, = ax_prob.plot([], [], color='blue', lw=2, label='Linear Model')
    #line_prob3, = ax_prob.plot([], [], color='red', lw=2, label='GNN Model')

    ax_prob.set_title("Event Probability (Persistent Tracks)", fontweight='bold')
    ax_prob.set_ylim(-0.1, 1.1)
    ax_prob.set_xlim(0, duration)
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xlabel("Time (s)")
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(loc='upper right')

    # 2. Spectrogram
    ax_spec = axes[0, 1]
    # Pxx_db is (Freq, Time). t is time array. f is freq array.
    # pcolormesh needs coordinates for corners, so we might need to adjust shapes or assume centers.
    # shading='gouraud' usually handles it well.
    c = ax_spec.pcolormesh(t, f, Pxx_db, shading='gouraud', cmap='inferno')
    cursor_line = ax_spec.axvline(x=0, color='cyan', linestyle='--')
    ax_spec.set_title("Spectrogram", fontweight='bold')
    ax_spec.set_ylabel("Freq (Hz)")
    ax_spec.set_ylim(0, config.MAX_FREQ * 1.2)
    # plt.colorbar(c, ax=ax_spec)

    # 3. PSD Curve (Instantaneous)
    ax_psd = axes[1, 0]
    line_psd, = ax_psd.plot([], [], color='navy', lw=1.5, label='Instantaneous PSD')
    line_nf, = ax_psd.plot([], [], color='gray', linestyle='--', alpha=0.7, label='Noise Floor')
    scatter_peaks = ax_psd.scatter([], [], color='red', s=50, zorder=5, label='Peaks')
    scatter_harmonics = ax_psd.scatter([], [], color='lime', s=80, marker='*', zorder=10, label='Tracked Harmonics')

    ax_psd.set_title("Instantaneous PSD & Detections", fontweight='bold')
    ax_psd.set_xlim(0, config.MAX_FREQ)
    ax_psd.set_ylim(-100, 0) # Fixed range
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
        #line_prob3.set_data([], [])
        line_psd.set_data([], [])
        line_nf.set_data([], [])
        scatter_peaks.set_offsets(np.empty((0, 2)))
        scatter_harmonics.set_offsets(np.empty((0, 2)))
        cursor_line.set_xdata([0])
        return line_prob1, line_prob2, line_psd, line_nf, scatter_peaks, scatter_harmonics, cursor_line

    def update(frame_idx):
        if frame_idx >= len(t): return init()

        current_time = t[frame_idx]

        # Update Probabilities (History up to now)
        # We plot the pre-calculated curves up to current_time
        valid_indices = t <= current_time
        line_prob1.set_data(t[valid_indices], prob_baseline[valid_indices])
        line_prob2.set_data(t[valid_indices], prob_linear[valid_indices])
        #line_prob3.set_data(t[valid_indices], prob_gnn[valid_indices])

        # Update Spectrogram Cursor
        cursor_line.set_xdata([current_time])

        # Update PSD
        psd_frame = Pxx_db[:, frame_idx]
        nf_frame = signal_processing.estimate_noise_floor(psd_frame)
        peaks = peaks_per_frame[frame_idx]

        line_psd.set_data(f, psd_frame)
        line_nf.set_data(f, nf_frame)

        if peaks:
            pf = [p['freq'] for p in peaks]
            pp = [p['power'] for p in peaks]
            scatter_peaks.set_offsets(np.c_[pf, pp])
        else:
            scatter_peaks.set_offsets(np.empty((0, 2)))

        # Update Harmonics & Table
        # Check if we are inside a track
        active_track = None
        if tracks:
            # Simple check for the best track
            bt = tracks[0]
            if bt['start_frame'] <= frame_idx <= bt['last_seen']:
                active_track = bt
        
        candidates = [] # Initialize candidates to prevent UnboundLocalError
        if active_track:
            # We want to show the harmonics for THIS frame, but track stores 'best_candidate' from BEST frame.
            # We can run detect_harmonics_iterative for this frame's peaks to find the matching candidate.
            candidates = harmonic_detection.detect_harmonics_iterative(peaks)
            # Find the one matching track freq
            match = None
            for c in candidates:
                if abs(c['base_freq'] - active_track['freq']) < (active_track['freq'] * config.TOLERANCE):
                    match = c
                    break

            if match:
                hf = [h['freq'] for h in match['harmonics']]
                hp = [h['power'] for h in match['harmonics']]
                scatter_harmonics.set_offsets(np.c_[hf, hp])

                # Update Table
                for r in range(10):
                    for c in range(4):
                        the_table[r+1, c].get_text().set_text("-")

                for i, h in enumerate(match['harmonics']):
                    if i >= 10: break
                    the_table[i+1, 0].get_text().set_text(f"#{h['harmonic_index']}")
                    the_table[i+1, 1].get_text().set_text(f"{h['freq']:.1f}")
                    the_table[i+1, 2].get_text().set_text(f"{h['power']:.1f}")
                    the_table[i+1, 3].get_text().set_text(f"{h['snr']:.1f}")
            else:
                scatter_harmonics.set_offsets(np.empty((0, 2)))
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
            score2 = torch.sigmoid(linear_model(lin_input)).item()

        # --- METHOD 3: GNN ---
        with torch.no_grad():
            gnn_batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device)
        #    score3 = torch.sigmoid(gnn_model(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch)).item()

        # Update Plots
        history_t.append(current_time)
        prob1_y.append(score1)
        prob2_y.append(score2)
       # prob3_y.append(score3)

        line_prob1.set_data(history_t, prob1_y)
        line_prob2.set_data(history_t, prob2_y)
        #line_prob3.set_data(history_t, prob3_y)

        print(f"Time: {current_time:.1f}s | Scores -> Baseline: {score1:.2f}, Linear: {score2:.2f}, GNN: {score3:.2f}")

        return line_prob1, line_prob2, line_psd, line_nf, scatter_peaks, scatter_harmonics, cursor_line, the_table

    print(f"Starting Analysis on {FILENAME}")
    

    # Interval: roughly map frame step to real time or faster
    # frame_step = (N_FFT - Overlap) / fs = 1024 / 44100 ~= 23ms
    interval_ms = 50 # Slower than real time for visualization

    anim = FuncAnimation(fig, update, frames=range(len(t)),
                         init_func=init, blit=False, interval=interval_ms)
    plt.show()
    #return line_prob1, line_prob2, line_prob3, line_psd, line_nf, scatter_peaks, scatter_harmonics, cursor_line, the_table
    return line_prob1, line_prob2,  line_psd, line_nf, scatter_peaks, scatter_harmonics, cursor_line, the_table

if __name__ == "__main__":
    run_analysis()
