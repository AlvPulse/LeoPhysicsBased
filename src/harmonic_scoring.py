# harmonic_scoring_with_benchmarks.py

import numpy as np
from scipy.signal import find_peaks
import time
from dataclasses import dataclass


# ============================================================
# 0️⃣ Helper: generate F0 candidates from detected peaks
# ============================================================

def generate_f0_candidates(peak_freqs,
                           fmin=80,
                           fmax=2000,
                           max_harmonic=8):
    candidates = []
    for f in peak_freqs:
        for k in range(1, max_harmonic + 1):
            f0 = f / k
            if fmin <= f0 <= fmax:
                candidates.append(f0)
    return np.unique(np.array(candidates))


# ============================================================
# 1️⃣ Peak alignment helper
# ============================================================

def peak_match_score(f_pred, peak_freqs, peak_mags, tol):
    diff = np.abs(peak_freqs - f_pred)
    idx = np.argmin(diff)
    if diff[idx] < tol:
        return peak_mags[idx]
    return 0


# ============================================================
# 2️⃣ Classic harmonic sum
# ============================================================

def harmonic_sum_peaks(peak_freqs,
                       peak_mags,
                       f0_candidates,
                       num_harmonics=6,
                       tol=10):
    scores = []
    for f0 in f0_candidates:
        s = 0
        for k in range(1, num_harmonics + 1):
            f_pred = k * f0
            s += peak_match_score(f_pred, peak_freqs, peak_mags, tol)
        scores.append(s)
    return np.array(scores)


# ============================================================
# 3️⃣ Weighted harmonic sum
# ============================================================

def weighted_harmonic_sum_peaks(peak_freqs,
                                peak_mags,
                                f0_candidates,
                                num_harmonics=6,
                                tol=10):
    scores = []
    for f0 in f0_candidates:
        s = 0
        for k in range(1, num_harmonics + 1):
            f_pred = k * f0
            s += peak_match_score(f_pred, peak_freqs, peak_mags, tol) / k
        scores.append(s)
    return np.array(scores)


# ============================================================
# 4️⃣ Log harmonic sum (robust to noise)
# ============================================================

def log_harmonic_sum_peaks(peak_freqs,
                           peak_mags,
                           f0_candidates,
                           num_harmonics=6,
                           tol=10):
    scores = []
    log_m = np.log(np.maximum(peak_mags, 1e-12))
    for f0 in f0_candidates:
        s = 0
        for k in range(1, num_harmonics + 1):
            f_pred = k * f0
            diff = np.abs(peak_freqs - f_pred)
            idx = np.argmin(diff)
            if diff[idx] < tol:
                s += log_m[idx]
        scores.append(s)
    return np.array(scores)


# ============================================================
# 5️⃣ NEW METHOD: Harmonic Lattice Voting
# ============================================================
# This is a powerful technique used in low-SNR radar/sonar.
# It accumulates evidence from all peak-to-peak harmonic ratios.
# At low SNR it often outperforms all others.

def harmonic_lattice_voting(peak_freqs,
                            peak_mags,
                            fmin=80,
                            fmax=2000,
                            num_harmonics=8,
                            resolution=1.0,
                            ratio_tol=0.015):
    """
    Build a lattice of harmonic ratios:
       if two peaks are at f_i, f_j and f_j ≈ (k/m)*f_i → vote for f0 = f_i/m.
    """

    f0_axis = np.arange(fmin, fmax, resolution)
    scores = np.zeros_like(f0_axis)

    N = len(peak_freqs)
    if N < 2:
        return f0_axis, scores

    for i in range(N):
        fi = peak_freqs[i]
        mi = peak_mags[i]

        for j in range(i + 1, N):
            fj = peak_freqs[j]
            mj = peak_mags[j]

            ratio = fj / fi

            for m in range(1, num_harmonics + 1):
                for k in range(1, num_harmonics + 1):
                    expected = k / m

                    if abs(ratio - expected) < ratio_tol:
                        f0 = fi / m
                        if fmin <= f0 <= fmax:
                            idx = int((f0 - fmin) / resolution)
                            scores[idx] += (mi + mj) * 0.5

    return f0_axis, scores


# ============================================================
# 6️⃣ Unified wrapper
# ============================================================

def compute_f0_scores_all(
    peak_freqs,
    peak_mags,
    fmin=80,
    fmax=2000,
    method="weighted",
    num_harmonics=6,
    tol=10
):

    # Lattice voting uses no f0 candidates → handle separately
    if method == "lattice":
        return harmonic_lattice_voting(
            peak_freqs,
            peak_mags,
            fmin,
            fmax,
            num_harmonics=num_harmonics
        )

    # Otherwise: generate f0 candidates
    f0_candidates = generate_f0_candidates(
        peak_freqs, fmin, fmax, max_harmonic=num_harmonics
    )

    if len(f0_candidates) == 0:
        return np.array([]), np.array([])

    if method == "sum":
        scores = harmonic_sum_peaks(
            peak_freqs, peak_mags, f0_candidates,
            num_harmonics=num_harmonics, tol=tol
        )

    elif method == "weighted":
        scores = weighted_harmonic_sum_peaks(
            peak_freqs, peak_mags, f0_candidates,
            num_harmonics=num_harmonics, tol=tol
        )

    elif method == "log":
        scores = log_harmonic_sum_peaks(
            peak_freqs, peak_mags, f0_candidates,
            num_harmonics=num_harmonics, tol=tol
        )

    else:
        raise ValueError("Unknown method.")

    return f0_candidates, scores


# ============================================================
# 7️⃣ BENCHMARKING SUITE
# ============================================================

@dataclass
class BenchmarkResult:
    method: str
    time_ms: float
    f0_est: float
    error_hz: float


def synthetic_harmonic_signal(f0, sr, duration, harmonics=6, snr_db=0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        x += (1 / k) * np.sin(2 * np.pi * k * f0 * t)

    # Add Gaussian noise for SNR
    sig_power = np.mean(x ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(x))

    return x + noise


def compute_peaks(signal, sr, n_fft=4096):
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    # Peak detection
    idx, props = find_peaks(
        spec,
        prominence=np.max(spec)*0.01,
        distance=3,
        height=np.max(spec)*0.005
    )

    return freqs[idx], spec[idx]


def benchmark_methods(f0_true=800, sr=32000, snr_db=0):
    # Generate synthetic signal
    x = synthetic_harmonic_signal(f0_true, sr, 0.05, snr_db=snr_db)
    peak_freqs, peak_mags = compute_peaks(x, sr)

    methods = ["sum", "weighted", "log", "lattice"]
    results = []

    for m in methods:
        t0 = time.time()
        f0_axis, scores = compute_f0_scores_all(
            peak_freqs, peak_mags, fmin=80, fmax=2000, method=m
        )
        dt = (time.time() - t0) * 1000

        if len(scores) == 0:
            f0_est = None
            err = None
        else:
            f0_est = f0_axis[np.argmax(scores)]
            err = abs(f0_est - f0_true)

        results.append(BenchmarkResult(m, dt, f0_est, err))

    return results


# ============================================================
# 8️⃣ Run benchmark example
# ============================================================

if __name__ == "__main__":
    results = benchmark_methods(f0_true=800, snr_db=0)

    print("Benchmark results at SNR = 0 dB:")
    for r in results:
        print(f"{r.method:<10s}   time={r.time_ms:6.2f} ms   "
              f"f0_est={r.f0_est}   error={r.error_hz}")
