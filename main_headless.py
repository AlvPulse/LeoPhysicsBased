import numpy as np
import torch
import os
import glob
from torch_geometric.data import Data, Batch

# Import modules
from src import config, signal_processing, harmonic_detection, baseline_model
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

    # --- METHOD 1: BASELINE HEURISTIC ---
    score1, base_freq1 = baseline_model.detect_baseline_heuristic(peaks)

    # --- METHOD 2 & 3 PREP: ITERATIVE SEARCH ---
    candidates = harmonic_detection.detect_harmonics_iterative(peaks)

    linear_vec = np.zeros(20, dtype=np.float32)
    best_candidate_freq = 0.0

    if candidates:
        best_candidate = candidates[0]
        best_candidate_freq = best_candidate['base_freq']

        for h in best_candidate['harmonics']:
            idx = h['harmonic_index']
            if idx <= 10:
                vec_idx = (idx - 1) * 2
                linear_vec[vec_idx] = h['snr'] / 50.0
                linear_vec[vec_idx+1] = (h['power'] + 100) / 100.0

    # --- METHOD 2: LINEAR ---
    with torch.no_grad():
        lin_input = torch.tensor(linear_vec, dtype=torch.float).unsqueeze(0).to(device)
        score2 = torch.sigmoid(linear_model(lin_input)).item()

    # --- METHOD 3: GNN ---
    if not peaks:
         x = torch.zeros((1, 3), dtype=torch.float)
         edge_index = torch.zeros((2, 0), dtype=torch.long)
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

    with torch.no_grad():
        gnn_batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device)
        score3 = torch.sigmoid(gnn_model(gnn_batch.x, gnn_batch.edge_index, gnn_batch.batch)).item()

    return {
        'filename': os.path.basename(filepath),
        'baseline_prob': score1,
        'linear_prob': score2,
        'gnn_prob': score3,
        'base_freq': best_candidate_freq if best_candidate_freq > 0 else base_freq1
    }

def print_confusion_matrix(tp, fp, tn, fn, model_name):
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
    print(f"TP: {tp:<4} | FP: {fp:<4}")
    print(f"FN: {fn:<4} | TN: {tn:<4}")

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
    # Also check debug_data
    debug_files = glob.glob("debug_data/yes/*.wav")

    all_files = yes_files + no_files + debug_files

    print(f"Found {len(all_files)} files.")
    print(f"{'Filename':<20} | {'Base(H)':<10} | {'Linear':<10} | {'GNN':<10} | {'Freq (Hz)':<10} | {'True Label'}")
    print("-" * 80)

    # Metrics Accumulators
    metrics = {
        'Baseline': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'Linear': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'GNN': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    }

    for fpath in all_files:
        res = process_file(fpath, linear_model, gnn_model, device)
        if res:
            is_yes = "yes" in fpath or "Autel" in fpath # Assuming Autel in debug_data/yes is YES
            label = "YES" if is_yes else "NO"

            # Update Metrics (Threshold 0.5)
            # Baseline
            p = res['baseline_prob'] > 0.5
            if is_yes and p: metrics['Baseline']['tp'] += 1
            elif is_yes and not p: metrics['Baseline']['fn'] += 1
            elif not is_yes and p: metrics['Baseline']['fp'] += 1
            elif not is_yes and not p: metrics['Baseline']['tn'] += 1

            # Linear
            p = res['linear_prob'] > 0.5
            if is_yes and p: metrics['Linear']['tp'] += 1
            elif is_yes and not p: metrics['Linear']['fn'] += 1
            elif not is_yes and p: metrics['Linear']['fp'] += 1
            elif not is_yes and not p: metrics['Linear']['tn'] += 1

            # GNN
            p = res['gnn_prob'] > 0.5
            if is_yes and p: metrics['GNN']['tp'] += 1
            elif is_yes and not p: metrics['GNN']['fn'] += 1
            elif not is_yes and p: metrics['GNN']['fp'] += 1
            elif not is_yes and not p: metrics['GNN']['tn'] += 1

            # Print
            # To avoid spamming, print if incorrect or every 10th
            is_correct_gnn = (res['gnn_prob'] > 0.5) == is_yes
            if not is_correct_gnn or (metrics['GNN']['tp'] + metrics['GNN']['tn']) % 10 == 0:
                print(f"{res['filename']:<20} | {res['baseline_prob']:<10.2f} | {res['linear_prob']:<10.2f} | {res['gnn_prob']:<10.2f} | {res['base_freq']:<10.1f} | {label}")

    print("-" * 80)

    for m_name, m_vals in metrics.items():
        print_confusion_matrix(m_vals['tp'], m_vals['fp'], m_vals['tn'], m_vals['fn'], m_name)

if __name__ == "__main__":
    main()
