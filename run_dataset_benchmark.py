import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.signal_processing import load_audio, compute_psd, find_significant_peaks
from src.harmonic_scoring import (
    compute_fixed_grid_scores, 
    harmonic_product_spectrum, 
    cepstrum_scoring, 
    dp_harmonic_tracking
)
from src import config


def get_chunk_scores_matrix(audio, fs, method, f0_axis):
    """
    Splits audio into 1-second chunks and returns a (T, F) scoring matrix.
    """
    chunk_size = int(fs * 1.0) # 1-second chunking
    num_chunks = max(1, len(audio) // chunk_size)
    
    chunk_scores_matrix = np.zeros((num_chunks, len(f0_axis)))

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        
        if len(chunk) < chunk_size // 2:
            continue # skip heavily truncated final chunks

        # Compute PSD
        f, psd_db = compute_psd(chunk, fs)
        psd_mag = 10 ** (psd_db / 10)  # Linear PSD
        
        if method in ["hps", "cepstrum"]:
            if method == "hps":
                freqs, scores = harmonic_product_spectrum(psd_mag, f)
            elif method == "cepstrum":
                freqs, scores = cepstrum_scoring(psd_mag, f)
                
            # Interpolate to the standardized f0 axis
            chunk_scores_matrix[i] = np.interp(f0_axis, freqs, scores, left=0, right=0)
            
        else:
            peaks = find_significant_peaks(f, psd_db, noise_floor_db=None, max_peaks=20)
            peak_freqs = np.array([p['freq'] for p in peaks])
            peak_mags = np.array([10**(p['power']/10) for p in peaks])
            
            _, scores = compute_fixed_grid_scores(
                peak_freqs, peak_mags, 
                method=method, 
                fmin=80, fmax=2000, resolution=5.0
            )
            chunk_scores_matrix[i] = scores
            
    return chunk_scores_matrix


def run_dataset_benchmark(yes_dir="data/yes", no_dir="data/no"):
    """
    1. Loads dataset
    2. Iterates over evaluation methods
    3. Extracts 1-sec chunks and evaluates with/without DP History
    4. Plots ROC curves and compares runtime
    """
    yes_files = glob.glob(os.path.join(yes_dir, "*.wav"))
    no_files = glob.glob(os.path.join(no_dir, "*.wav"))
    
    all_files = [(f, 1) for f in yes_files] + [(f, 0) for f in no_files]
    
    if len(all_files) == 0:
        print(f"No WAV files found in {yes_dir} or {no_dir}. Please populate dataset.")
        return

    print(f"Running benchmark on {len(yes_files)} Positive and {len(no_files)} Negative files...\n")

    f0_axis = np.arange(80, 2000, 5.0)
    methods = ["sum", "weighted", "log", "lattice", "hps", "cepstrum"]
    
    plt.figure(figsize=(12, 9))
    plt.title('ROC Curve: Harmonic Scoring Methods (1s Chunks)', fontsize=14, fontweight='bold')
    
    results = {}

    for method in methods:
        for use_dp in [False, True]:
            method_name = f"{method.upper()}" + (" + DP Tracker" if use_dp else "")
            
            y_true = []
            y_scores = []
            
            start_time = time.time()
            
            for filepath, label in all_files:
                audio, fs = load_audio(filepath, config.SAMPLE_RATE)
                if audio is None: continue
                
                # Generate Chunk Scores Matrix
                scores_matrix = get_chunk_scores_matrix(audio, fs, method, f0_axis)
                
                # Track-Before-Detect (DP Tracker)
                if use_dp:
                    final_scores = dp_harmonic_tracking(scores_matrix, decay=0.8, transition_width=2)
                else:
                    final_scores = scores_matrix
                    
                # Aggregate: Max score identified in the clip
                clip_score = np.max(final_scores) if len(final_scores) > 0 else 0.0
                
                y_true.append(label)
                y_scores.append(clip_score)
                
            elapsed_ms_per_file = (time.time() - start_time) / len(all_files) * 1000
            
            # Compute Metrics
            # Handle NaNs or identical scores safely
            y_scores = np.nan_to_num(y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Save results
            results[method_name] = {'auc': roc_auc, 'time_ms': elapsed_ms_per_file}
            
            # Plot ROC
            linestyle = '-' if not use_dp else '--'
            linewidth = 1.5 if not use_dp else 2.5
            plt.plot(fpr, tpr, lw=linewidth, linestyle=linestyle, 
                     label=f'{method_name} (AUC = {roc_auc:.3f})')
            
            print(f"{method_name:<25} | AUC: {roc_auc:.4f} | Avg File Time: {elapsed_ms_per_file:.1f}ms")

    # Finalize plot formatting
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plot_path = "roc_benchmark.png"
    plt.savefig(plot_path)
    print(f"\nROC Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Adjust the directory paths if needed
    run_dataset_benchmark(yes_dir="data/yes", no_dir="data/no")