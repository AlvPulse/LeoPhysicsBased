import numpy as np
from src import config

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
        harmonics.append(base_harmonic)

        # 3. Search for harmonics 2, 3, ... N
        # We don't need to check peaks *before* the base (since they are lower freq)
        # But wait, a lower peak could be the TRUE fundamental if the current peak is the 2nd harmonic.
        # However, the outer loop will eventually check that lower peak as a candidate base.
        # So we only look forward for harmonics of THIS base.

        current_harmonic_idx = 2
        consecutive_misses = 0
        max_misses = 2 # Stop if we miss 2 harmonics in a row? Maybe keep searching if we expect higher ones.

        while True:
            target_freq = f0 * current_harmonic_idx
            if target_freq > config.MAX_FREQ * 1.1: # Allow slight overshot
                break

            # Find closest peak in the list
            best_match = None
            min_dist = float('inf')

            # Optimization: could use binary search or just iterate since N is small (<20)
            for p in peaks_sorted_freq:
                if p['freq'] < target_freq * (1 - config.TOLERANCE): continue
                if p['freq'] > target_freq * (1 + config.TOLERANCE): break # Sorted, so can stop early

                dist = abs(p['freq'] - target_freq)
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
                    harmonics.append(h_info)
                consecutive_misses = 0
            else:
                consecutive_misses += 1

            current_harmonic_idx += 1
            if consecutive_misses > 3: # Stop searching if we miss too many
                break

        # 4. Score this candidate
        if len(harmonics) >= config.MIN_HARMONICS:
            # Score formula:
            # - Reward total SNR
            # - Reward number of harmonics
            # - Penalize missing harmonics (gaps)?
            # - Penalize drift?

            total_snr = sum(h['snr'] for h in harmonics)
            avg_drift = sum(h['drift'] for h in harmonics) / len(harmonics)

            # Simple heuristic score:
            # Score = (Total SNR) * (Num Harmonics)^1.5 * (1 - Avg Drift)
            score = total_snr * (len(harmonics) ** 1.5) * (1.0 - avg_drift)

            candidates.append({
                'base_freq': f0,
                'harmonics': harmonics,
                'score': score
            })

    # 5. Filter Candidates (Sub-harmonic Check)
    # If we find f0=220 (with 440, 660, 880) and f0=440 (with 880),
    # the 220 candidate is likely the "true" fundamental if it explains more peaks.
    # But sometimes noise at 220 might trigger a false fundamental.
    # Logic: Prefer the candidate with the highest score.

    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates[:max_candidates]
