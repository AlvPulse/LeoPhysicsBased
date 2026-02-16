import numpy as np
import torch
import os
import glob
import joblib
import xgboost as xgb
from torch_geometric.data import Data, Batch

# Import modules
from src import config, signal_processing, harmonic_detection, baseline_model, feature_extraction
from src.models import LinearHarmonicModel
# GNNEventDetector removed

def process_file(filepath, linear_model, clf, device):
    """
    Process a single file and return probabilities using persistence tracking.
    """
    # Load Audio
    audio, fs = signal_processing.load_audio(filepath)
    if(len(audio)<config.N_FFT):
        return None
    if audio is None: return None

    # 1. Compute STFT, Peaks, and Spectral Features
    f, t, Pxx_db, peaks_per_frame, spectral_features = signal_processing.compute_spectrogram_and_peaks(audio, fs)

    # 2. Track Harmonics (Persistence)
    tracks = harmonic_detection.track_harmonics(peaks_per_frame, t, spectral_features)

    # Default values (assume NO event)
    baseline_prob = 0.0
    linear_prob = 0.0
    clf_prob = 0.0
    best_freq = 0.0

    if tracks:
        # Sort by score descending (already done by track_harmonics but to be safe)
        tracks.sort(key=lambda x: x['max_score'], reverse=True)
        best_track = tracks[0]
        best_freq = best_track['freq']

        # Get peaks from the best frame of this track for analysis
        best_frame_idx = best_track.get('best_frame_idx', 0)

        # Safety check for frame index
        if best_frame_idx >= len(peaks_per_frame):
            best_frame_idx = 0

        best_peaks = peaks_per_frame[best_frame_idx]

        # --- METHOD 1: BASELINE HEURISTIC ---
        # Run heuristic on the best frame
        baseline_prob, _ = baseline_model.detect_baseline_heuristic(best_peaks)

        # Retrieve Best Candidate Snapshot
        best_candidate = best_track.get('best_candidate')

        if best_candidate:
            # --- METHOD 2: LINEAR ---
            linear_vec = feature_extraction.extract_linear_features(best_candidate)
            with torch.no_grad():
                lin_input = torch.tensor(linear_vec, dtype=torch.float).unsqueeze(0).to(device)
                linear_prob = torch.sigmoid(linear_model(lin_input)).item()

            # --- METHOD 3: CLASSIFIER (XGBoost) ---
            classifier_vec = feature_extraction.extract_classifier_features(best_track)
            # Sklearn/XGBoost expects (N_samples, N_features)
            # classifier_vec is 1D, so reshape to (1, -1)
            clf_prob = clf.predict_proba(classifier_vec.reshape(1, -1))[0][1]

    # # If no tracks found, probs remain 0.0 (correct for noise files)

    return {
        'filename': os.path.basename(filepath),
        'baseline_prob': baseline_prob,
        'linear_prob': linear_prob,
        'clf_prob': clf_prob,
        'base_freq': best_freq
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
    print("Starting Headless Analysis (Persistence Enabled)...")

    # Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        linear_model = LinearHarmonicModel()
        linear_model.load_state_dict(torch.load('models/linear_model.pth', map_location=device))
        linear_model.to(device)
        linear_model.eval()

        # Load Sklearn/XGBoost Model
        clf = joblib.load('models/classifier_model.pkl')

    except Exception as e:
        print(f"Error loading models: {e}. Run train.py first.")
        # Proceeding without models might crash later, but user should have run train.
        return

    # Find Files
    yes_files = glob.glob("data/yes/*.wav")
    no_files = glob.glob("data/no/*.wav")
    # Also check debug_data
    debug_files = glob.glob("debug_data/yes/*.wav")

    all_files = yes_files + no_files + debug_files

    print(f"Found {len(all_files)} files.")
    print(f"{'Filename':<25} | {'Base(H)':<10} | {'Linear':<10} | {'Classif':<10} | {'Freq (Hz)':<10} | {'True Label'}")
    print("-" * 90)

    # Metrics Accumulators
    metrics = {
        'Baseline': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'Linear': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'Classifier': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    }

    for fpath in all_files:
        res = process_file(fpath, linear_model, clf, device)
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

            # Classifier
            p = res['clf_prob'] > 0.5
            if is_yes and p: metrics['Classifier']['tp'] += 1
            elif is_yes and not p: metrics['Classifier']['fn'] += 1
            elif not is_yes and p: metrics['Classifier']['fp'] += 1
            elif not is_yes and not p: metrics['Classifier']['tn'] += 1

            # Print
            # Print if incorrect or every 10th
            is_correct_clf = (res['clf_prob'] > 0.5) == is_yes
            if not is_correct_clf or (metrics['Classifier']['tp'] + metrics['Classifier']['tn']) % 10 == 0:
                print(f"{res['filename']:<25} | {res['baseline_prob']:<10.2f} | {res['linear_prob']:<10.2f} | {res['clf_prob']:<10.2f} | {res['base_freq']:<10.1f} | {label}")

    print("-" * 90)

    for m_name, m_vals in metrics.items():
        print_confusion_matrix(m_vals['tp'], m_vals['fp'], m_vals['tn'], m_vals['fn'], m_name)

if __name__ == "__main__":
    main()
