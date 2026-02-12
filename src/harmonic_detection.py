import numpy as np
from src import config

def calculate_quality(harmonic):
    """
    Calculates the quality factor for a single harmonic.
    Q = w1*SNR + w2*Power + w3*(1-Drift)
    """
    # Weights for quality calculation (configurable if needed)
    w_snr = 0.5
    w_pwr = 0.3
    w_drift = 0.2

    # Normalize SNR (approx 0-50 dB range)
    snr_norm = min(max(harmonic['snr'], 0), 50) / 50.0

    # Normalize Power (approx -100 to 0 dB range)
    pwr_norm = min(max(harmonic['power'] + 100, 0), 100) / 100.0

    # Drift is already 0-1 (ideally close to 0)
    drift_score = max(0, 1.0 - harmonic['drift'])

    q = (w_snr * snr_norm) + (w_pwr * pwr_norm) + (w_drift * drift_score)
    return q

def detect_harmonics_iterative(peaks, max_candidates=5):
    """
    Refined harmonic detection.
    Args:
        peaks: List of peak dicts.
        max_candidates: Number of best candidates to return.
    Returns:
        List of candidate dicts: {'base_freq', 'harmonics': list, 'score': float}
    """
    candidates = []

    if not peaks:
        return candidates

    # 1. Sort peaks by frequency to ensure we check fundamental possibilities sequentially
    peaks_sorted_freq = sorted(peaks, key=lambda x: x['freq'])

    # 2. Iterate through each peak as a potential fundamental (f0)
    for i, base_peak in enumerate(peaks_sorted_freq):
        f0 = base_peak['freq']
        harmonics = [] # List of peak dicts

        # Base peak is Harmonic #1
        # Store as dict with extra info
        base_harmonic = base_peak.copy()
        base_harmonic['harmonic_index'] = 1
        base_harmonic['drift'] = 0.0
        base_harmonic['quality'] = calculate_quality(base_harmonic)
        harmonics.append(base_harmonic)

        # 3. Search for harmonics 2, 3, ... N
        current_harmonic_idx = 2
        consecutive_misses = 0
        max_misses = 2

        while True:
            target_freq = f0 * current_harmonic_idx
            if target_freq > config.MAX_FREQ * 1.1: # Allow slight overshot
                break

            # Find closest peak in the list
            best_match = None
            min_dist = float('inf')

            # Optimization: could use binary search or just iterate since N is small (<20)
            for p in peaks_sorted_freq:
                # Optimized range check
                lower_bound = target_freq * (1 - config.TOLERANCE)
                upper_bound = target_freq * (1 + config.TOLERANCE)

                if p['freq'] < lower_bound: continue
                if p['freq'] > upper_bound: break # Sorted, so can stop early

                dist = abs(p['freq'] - target_freq)
                # Check absolute distance relative to tolerance
                if dist < (target_freq * config.TOLERANCE):
                    if dist < min_dist:
                        min_dist = dist
                        best_match = p

            if best_match:
                # Add found harmonic
                # Avoid duplicates: check if this peak is already used *for this candidate*
                if not any(h['freq'] == best_match['freq'] for h in harmonics):
                    h_info = best_match.copy()
                    h_info['harmonic_index'] = current_harmonic_idx
                    h_info['drift'] = min_dist / target_freq
                    h_info['quality'] = calculate_quality(h_info)
                    harmonics.append(h_info)
                consecutive_misses = 0
            else:
                consecutive_misses += 1

            current_harmonic_idx += 1
            if consecutive_misses > max_misses: # Stop searching if we miss too many
                break

        # 4. Score this candidate
        if len(harmonics) >= config.MIN_HARMONICS:
            # Score formula:
            # - Sum of harmonic qualities
            # - Bonus for consecutive harmonics?
            # - Penalty for missing harmonics (gaps)?

            total_quality = sum(h['quality'] for h in harmonics)
            avg_drift = sum(h['drift'] for h in harmonics) / len(harmonics)

            # Weighted score: Quality * (1 - Drift) * (Num Harmonics / Expected)
            score = total_quality * (1.0 - avg_drift)

            candidates.append({
                'base_freq': f0,
                'harmonics': harmonics,
                'score': score
            })

    # 5. Filter Candidates (Sub-harmonic Check)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates[:max_candidates]

def extract_linear_features(candidates, num_harmonics=10):
    """
    Extracts a feature vector for the linear model from the best candidate.
    Returns: np.array of shape (num_harmonics * 2,)
    """
    vec = np.zeros(num_harmonics * 2, dtype=np.float32)

    if not candidates:
        return vec

    best = candidates[0]

    for h in best['harmonics']:
        idx = h['harmonic_index']
        if idx <= num_harmonics:
            vec_idx = (idx - 1) * 2
            # Features: SNR (norm), Power (norm)
            # Use same normalization as calculate_quality or independent?
            # Using independent normalization for model stability
            vec[vec_idx] = min(max(h['snr'], 0), 50) / 50.0
            vec[vec_idx+1] = min(max(h['power'] + 100, 0), 100) / 100.0

    return vec
