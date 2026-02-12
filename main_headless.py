import numpy as np
import torch
import os
import glob
from torch_geometric.data import Data, Batch

# Import modules
from src import config, signal_processing, harmonic_detection
from src.models import LinearHarmonicModel, GNNEventDetector

def process_file(filepath, linear_model, gnn_model, device):
    """
    Process a single file and return probabilities.
    """
    # Load Audio
    audio, fs = signal_processing.load_audio(filepath)
    if audio is None: return None

    # Process Signal (Full duration for this example)
    f, psd = signal_processing.compute_psd(audio, fs)
    nf = signal_processing.estimate_noise_floor(psd)
    peaks = signal_processing.find_significant_peaks(f, psd, nf)

    # --- METHOD 1: HEURISTIC ---
    candidates = harmonic_detection.detect_harmonics_iterative(peaks)

    score1 = 0.0
    best_candidate_freq = 0.0

    if candidates:
        best_candidate = candidates[0]
        # Normalize score roughly
        score1 = min(best_candidate['score'] / 500.0, 1.0)
        best_candidate_freq = best_candidate['base_freq']

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

        linear_vec = np.zeros(20, dtype=np.float32)
        if candidates:
            best = candidates[0]
            for h in best['harmonics']:
                idx = h['harmonic_index']
                if idx <= 10:
                    vec_idx = (idx - 1) * 2
                    linear_vec[vec_idx] = h['snr'] / 50.0
                    linear_vec[vec_idx+1] = (h['power'] + 100) / 100.0

    # --- METHOD 2: LINEAR ---
    with torch.no_grad():
        lin_input = torch.tensor(linear_vec, dtype=torch.float).unsqueeze(0).to(device)
        score2 = linear_model(lin_input).item()

    # --- METHOD 3: GNN ---
    with torch.no_grad():
        gnn_batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device)
        score3 = gnn_model(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch).item()

    return {
        'filename': os.path.basename(filepath),
        'heuristic_prob': score1,
        'linear_prob': score2,
        'gnn_prob': score3,
        'base_freq': best_candidate_freq
    }

def main():
    print("Starting Headless Analysis...")

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
    except Exception as e:
        print(f"Error loading models: {e}. Run train.py first.")
        return

    # Find Files
    yes_files = glob.glob("data/yes/*.wav")
    no_files = glob.glob("data/no/*.wav")
    all_files = yes_files + no_files

    print(f"Found {len(all_files)} files.")
    print(f"{'Filename':<20} | {'Heuristic':<10} | {'Linear':<10} | {'GNN':<10} | {'Freq (Hz)':<10} | {'True Label'}")
    print("-" * 80)

    correct_gnn = 0
    total = 0

    for fpath in all_files:
        res = process_file(fpath, linear_model, gnn_model, device)
        if res:
            is_yes = "yes" in fpath
            label = "YES" if is_yes else "NO"

            # Simple thresholding for accuracy check
            pred_gnn = res['gnn_prob'] > 0.5
            if pred_gnn == is_yes:
                correct_gnn += 1
            total += 1

            # Print only a subset to avoid clutter, or all? Let's print first 5 of each and last 5
            # Actually, let's print mismatches mostly
            if (pred_gnn != is_yes) or (total % 10 == 0):
                print(f"{res['filename']:<20} | {res['heuristic_prob']:<10.2f} | {res['linear_prob']:<10.2f} | {res['gnn_prob']:<10.2f} | {res['base_freq']:<10.1f} | {label}")

    print("-" * 80)
    print(f"GNN Accuracy on this set: {correct_gnn}/{total} ({correct_gnn/total*100:.1f}%)")

if __name__ == "__main__":
    main()
