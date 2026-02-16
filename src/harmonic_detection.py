import numpy as np
from src import config

def calculate_quality(harmonic, weights=None):
    """
    Calculates the quality factor for a single harmonic.
    Q = w1*SNR + w2*Power + w3*(1-Drift)
    """
    if weights is None:
        weights = config.QUALITY_WEIGHTS

    w_snr, w_pwr, w_drift = weights

    # Normalize SNR (0-50 dB map to 0-1)
    snr_norm = min(max(harmonic['snr'], 0), 50) / 50.0

    # Normalize Power (-100 to 0 dB map to 0-1)
    # Assuming power is roughly -100 to 0 dB
    pwr_norm = min(max(harmonic['power'] + 100, 0), 100) / 100.0

    # Drift score (smaller drift is better)
    drift_score = max(0, 1.0 - (harmonic['drift'] * 10)) # Penalize drift heavily

    q = (w_snr * snr_norm) + (w_pwr * pwr_norm) + (w_drift * drift_score)
    return q

def detect_harmonics_iterative(peaks, max_candidates=5, config_params=None):
    """
    Detects harmonic series from a list of spectral peaks.
    config_params: dict with keys 'tolerance', 'snr_threshold', 'power_threshold', 'weights'
    """
    candidates = []

    # Default parameters
    tolerance = config.TOLERANCE
    snr_threshold = config.HARMONIC_MIN_SNR
    power_threshold = config.HARMONIC_MIN_POWER
    weights = config.QUALITY_WEIGHTS

    # Apply overrides
    if config_params:
        tolerance = config_params.get('tolerance', tolerance)
        snr_threshold = config_params.get('snr_threshold', snr_threshold)
        power_threshold = config_params.get('power_threshold', power_threshold)
        weights = config_params.get('weights', weights)

    if not peaks:
        return candidates

    # Sort peaks by frequency for efficient searching
    peaks_sorted_freq = sorted(peaks, key=lambda x: x['freq'])

    # Iterate through each peak, treating it as a potential Fundamental Frequency (f0)
    for i, base_peak in enumerate(peaks_sorted_freq):
        f0 = base_peak['freq']

        # Skip very low frequencies if needed
        if f0 < config.MIN_FREQ: continue

        harmonics = []

        # Add the fundamental
        base_harmonic = base_peak.copy()
        base_harmonic['harmonic_index'] = 1
        base_harmonic['drift'] = 0.0
        base_harmonic['quality'] = calculate_quality(base_harmonic, weights)
        harmonics.append(base_harmonic)

        current_harmonic_idx = 2
        consecutive_misses = 0
        max_misses = 2 # Allow missing a few harmonics

        while True:
            target_freq = f0 * current_harmonic_idx

            # Stop if out of range
            if target_freq > config.MAX_FREQ * 1.1:
                break

            best_match = None
            min_dist = float('inf')

            # Search for a peak near target_freq
            # Optimization: could use binary search, but linear scan is fine for small N
            lower_bound = target_freq * (1 - tolerance)
            upper_bound = target_freq * (1 + tolerance)

            for p in peaks_sorted_freq:
                # Basic filtering
                if p['snr'] < snr_threshold: continue
                if p['power'] < power_threshold: continue

                if p['freq'] < lower_bound: continue
                if p['freq'] > upper_bound: break # Sorted by freq, can stop early

                dist = abs(p['freq'] - target_freq)

                # Check tolerance
                if dist < (target_freq * tolerance):
                    if dist < min_dist:
                        min_dist = dist
                        best_match = p

            if best_match:
                # Avoid duplicates in the same series (though unlikely with freq sort)
                if not any(h['freq'] == best_match['freq'] for h in harmonics):
                    h_info = best_match.copy()
                    h_info['harmonic_index'] = current_harmonic_idx
                    h_info['drift'] = min_dist / target_freq
                    h_info['quality'] = calculate_quality(h_info, weights)
                    harmonics.append(h_info)
                consecutive_misses = 0
            else:
                consecutive_misses += 1

            current_harmonic_idx += 1
            if consecutive_misses > max_misses:
                break

        # Evaluate the Series
        if len(harmonics) >= config.MIN_HARMONICS:
            total_quality = sum(h['quality'] for h in harmonics)
            avg_drift = sum(h['drift'] for h in harmonics) / len(harmonics)

            # Base Score: Sum of qualities penalised by average drift
            score = total_quality * (1.0 - avg_drift)

            # Penalty for missing lower-order harmonics
            # (e.g. found 3rd and 4th but missed 2nd)
            max_found_idx = harmonics[-1]['harmonic_index']
            check_range = range(2, min(6, max_found_idx))
            found_indices = set(h['harmonic_index'] for h in harmonics)

            missing_low_order_count = 0
            for idx in check_range:
                if idx not in found_indices:
                    missing_low_order_count += 1

            if missing_low_order_count > 0:
                penalty = config.MISSING_HARMONIC_PENALTY ** missing_low_order_count
                score *= penalty

            candidates.append({
                'base_freq': f0,
                'harmonics': harmonics,
                'score': score
            })

    # Return top candidates
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:max_candidates]

def track_harmonics(peaks_per_frame, times):
    """
    Tracks harmonic series over time to build persistent objects.
    """
    active_tracks = []
    completed_tracks = []

    for frame_idx, peaks in enumerate(peaks_per_frame):
        candidates = detect_harmonics_iterative(peaks, max_candidates=5)
        matched_track_indices = set()

        for cand in candidates:
            best_track_idx = -1
            best_dist = float('inf')

            # Try to match candidate to an active track
            for t_idx, track in enumerate(active_tracks):
                if t_idx in matched_track_indices: continue

                # Simple frequency tracking: is the base frequency close?
                dist = abs(track['freq'] - cand['base_freq'])
                if dist < (track['freq'] * config.TOLERANCE):
                    if dist < best_dist:
                        best_dist = dist
                        best_track_idx = t_idx

            if best_track_idx != -1:
                # Update existing track
                track = active_tracks[best_track_idx]
                track['persistence'] += 1
                track['last_seen'] = frame_idx
                # Smooth update of frequency
                track['freq'] = 0.9 * track['freq'] + 0.1 * cand['base_freq']

                if cand['score'] > track['max_score']:
                    track['max_score'] = cand['score']
                    track['best_candidate'] = cand
                    track['best_frame_idx'] = frame_idx

                matched_track_indices.add(best_track_idx)
            else:
                # Create new track
                new_track = {
                    'freq': cand['base_freq'],
                    'persistence': 1,
                    'start_frame': frame_idx,
                    'last_seen': frame_idx,
                    'max_score': cand['score'],
                    'best_candidate': cand,
                    'best_frame_idx': frame_idx
                }
                active_tracks.append(new_track)

        # Maintenance: Remove old tracks, move completed ones
        active_tracks_next = []
        for t in active_tracks:
            if frame_idx - t['last_seen'] > config.PERSISTENCE_BUFFER:
                if t['persistence'] >= config.PERSISTENCE_THRESHOLD:
                    completed_tracks.append(t)
            else:
                active_tracks_next.append(t)
        active_tracks = active_tracks_next

    # Flush remaining active tracks
    for t in active_tracks:
        if t['persistence'] >= config.PERSISTENCE_THRESHOLD:
            completed_tracks.append(t)

    completed_tracks.sort(key=lambda x: x['max_score'], reverse=True)
    return completed_tracks

def extract_linear_features(candidates, num_harmonics=10):
    """
    Extracts a fixed-size feature vector for the Linear Model.
    Input: Candidate object (or list of candidates)
    Output: np.array of shape (num_harmonics * 2,) -> [SNR1, Pwr1, SNR2, Pwr2...]
    """
    vec = np.zeros(num_harmonics * 2, dtype=np.float32)

    best = None
    if isinstance(candidates, list):
        if len(candidates) > 0:
            if 'best_candidate' in candidates[0]:
                 # It's a track object
                best = candidates[0]['best_candidate']
            else:
                 # It's a raw candidate
                best = candidates[0]
    elif isinstance(candidates, dict):
         best = candidates

    if best is None:
        return vec

    for h in best['harmonics']:
        idx = h['harmonic_index']
        if idx <= num_harmonics:
            vec_idx = (idx - 1) * 2

            # Normalize for NN stability
            vec[vec_idx] = min(max(h['snr'], 0), 50) / 50.0
            vec[vec_idx+1] = min(max(h['power'] + 100, 0), 100) / 100.0

    return vec
