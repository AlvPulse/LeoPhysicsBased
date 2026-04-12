import bisect
import numpy as np
from src import config

def calculate_quality(harmonic, weights=None):
    """
    Calculates the quality factor for a single harmonic.
    Q = w1*SNR + w2*Power + w3*(1-Drift)
    """
    w_snr = 0.5
    w_pwr = 0.3
    w_drift = 0.2

    snr_norm = harmonic['snr']
    if snr_norm < 0: snr_norm = 0
    elif snr_norm > 50: snr_norm = 50
    snr_norm *= 0.02

    pwr_norm = harmonic['power'] + 100
    if pwr_norm < 0: pwr_norm = 0
    elif pwr_norm > 100: pwr_norm = 100
    pwr_norm *= 0.01

    drift = harmonic['drift']
    drift_score = 1.0 - drift
    if drift_score < 0: drift_score = 0

    q = (w_snr * snr_norm) + (w_pwr * pwr_norm) + (w_drift * drift_score)
    return q


def calculate_quality(harmonic, weights=None):
    """
    Calculates the quality factor for a single harmonic.
    Q = w1*SNR + w2*Power + w3*(1-Drift)
    """
    w_snr = 0.5
    w_pwr = 0.3
    w_drift = 0.2

    snr_norm = harmonic['snr']
    if snr_norm < 0: snr_norm = 0
    elif snr_norm > 50: snr_norm = 50
    snr_norm *= 0.02

    pwr_norm = harmonic['power'] + 100
    if pwr_norm < 0: pwr_norm = 0
    elif pwr_norm > 100: pwr_norm = 100
    pwr_norm *= 0.01

    drift = harmonic['drift']
    drift_score = 1.0 - drift
    if drift_score < 0: drift_score = 0

    q = (w_snr * snr_norm) + (w_pwr * pwr_norm) + (w_drift * drift_score)
    return q


def detect_harmonics_iterative(peaks, max_candidates=5, snr_threshold=None, power_threshold=None, tolerance=None, max_freq_limit=None, min_harmonics=None, missing_penalty=None):
    if not peaks:
        return []

    snr_threshold = snr_threshold if snr_threshold is not None else config.HARMONIC_MIN_SNR
    power_threshold = power_threshold if power_threshold is not None else config.HARMONIC_MIN_POWER
    tolerance = tolerance if tolerance is not None else config.TOLERANCE

    max_freq_limit = max_freq_limit if max_freq_limit is not None else config.MAX_FREQ * 1.1
    min_harmonics = min_harmonics if min_harmonics is not None else config.MIN_HARMONICS
    missing_penalty = missing_penalty if missing_penalty is not None else config.MISSING_HARMONIC_PENALTY

    peaks_sorted_freq = sorted(peaks, key=lambda x: x['freq'])
    num_peaks = len(peaks_sorted_freq)
    freqs = [p['freq'] for p in peaks_sorted_freq]

    candidates = []

    for i in range(num_peaks):
        base_peak = peaks_sorted_freq[i]
        f0 = freqs[i]

        harmonics = []

        # Add the fundamental
        base_harmonic = base_peak.copy()
        base_harmonic['harmonic_index'] = 1

        harmonics.append(base_harmonic)

        current_harmonic_idx = 2
        consecutive_misses = 0

        while consecutive_misses <= 2:
            target_freq = f0 * current_harmonic_idx
            if target_freq > max_freq_limit:
                break

            best_match = None
            min_dist = float('inf')

            # Search for a peak near target_freq
            # Optimization: could use binary search, but linear scan is fine for small N
            lower_bound = target_freq * (1 - tolerance)
            upper_bound = target_freq * (1 + tolerance)

            target_tol = target_freq * tolerance

            lower_bound = target_freq - target_tol
            upper_bound = target_freq + target_tol

            start_idx = bisect.bisect_left(freqs, lower_bound)
            for j in range(start_idx, num_peaks):
                p_freq = freqs[j]
                if p_freq > upper_bound: break

                p = peaks_sorted_freq[j]
                if p['snr'] < snr_threshold or p['power'] < power_threshold:
                    continue

                dist = abs(p_freq - target_freq)
                if dist < min_dist:
                    min_dist = dist
                    best_match = p

            if best_match:
                best_freq = best_match['freq']
                is_dup = False
                for h in harmonics:
                    if h['freq'] == best_freq:
                        is_dup = True
                        break

                if not is_dup:
                    h_info = best_match.copy()
                    h_info['harmonic_index'] = current_harmonic_idx
                    harmonics.append(h_info)
                consecutive_misses = 0
            else:
                consecutive_misses += 1

            current_harmonic_idx += 1

        if len(harmonics) >= min_harmonics:
            total_power = 0.0
            found_indices = set()
            for h in harmonics:
                total_power += h['power']
                found_indices.add(h['harmonic_index'])

            score = total_power

            # Penalty for missing lower-order harmonics
            # (e.g. found 3rd and 4th but missed 2nd)
            max_found_idx = harmonics[-1]['harmonic_index']
            check_upper = max_found_idx if max_found_idx < 6 else 6
            if check_upper > 2:
                missing_count = sum(1 for idx in range(2, check_upper) if idx not in found_indices)
                if missing_count > 0:
                    score *= (missing_penalty ** missing_count)

            candidates.append({
                'base_freq': f0,
                'harmonics': harmonics,
                'score': score,
                'signal_power': total_power
            })

    # Return top candidates
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:max_candidates]


def track_harmonics(peaks_per_frame, times=None, snr_threshold=None, power_threshold=None, tolerance=None, max_freq_limit=None, min_harmonics=None, missing_penalty=None):
    active_tracks = []
    completed_tracks = []

    tol = tolerance if tolerance is not None else config.TOLERANCE
    p_buf = config.PERSISTENCE_BUFFER
    p_thresh = config.PERSISTENCE_THRESHOLD

    for frame_idx, peaks in enumerate(peaks_per_frame):
        candidates = detect_harmonics_iterative(peaks, max_candidates=5, snr_threshold=snr_threshold, power_threshold=power_threshold, tolerance=tolerance, max_freq_limit=max_freq_limit, min_harmonics=min_harmonics, missing_penalty=missing_penalty)

        if not candidates:
            active_tracks_next = []
            for t in active_tracks:
                if frame_idx - t['last_seen'] > p_buf:
                    if t['persistence'] >= p_thresh:
                        completed_tracks.append(t)
                else:
                    active_tracks_next.append(t)
            active_tracks = active_tracks_next
            continue

        matched_track_indices = set()

        for cand in candidates:
            best_track_idx = -1
            best_dist = float('inf')
            cand_freq = cand['base_freq']

            # Try to match candidate to an active track
            for t_idx, track in enumerate(active_tracks):
                if t_idx in matched_track_indices: continue

                track_freq = track['freq']
                dist = abs(track_freq - cand_freq)
                if dist < (track_freq * tol) and dist < best_dist:
                    best_dist = dist
                    best_track_idx = t_idx

            if best_track_idx != -1:
                # Update existing track
                track = active_tracks[best_track_idx]
                track['persistence'] += 1
                track['last_seen'] = frame_idx
                track['freq'] = 0.9 * track['freq'] + 0.1 * cand_freq

                cand_score = cand['score']
                if cand_score > track['max_score']:
                    track['max_score'] = cand_score
                    track['best_candidate'] = cand
                    track['best_frame_idx'] = frame_idx
                    track['base_freq'] = cand['base_freq']
                    track['signal_power'] = cand['signal_power']

                matched_track_indices.add(best_track_idx)
            else:
                active_tracks.append({
                    'freq': cand_freq,
                    'persistence': 1,
                    'start_frame': frame_idx,
                    'last_seen': frame_idx,
                    'max_score': cand['score'],
                    'best_candidate': cand,
                    'best_frame_idx': frame_idx,
                    'base_freq': cand['base_freq'],
                    'signal_power': cand['signal_power']
                })

        # Maintenance: Remove old tracks, move completed ones
        active_tracks_next = []
        for t in active_tracks:
            if frame_idx - t['last_seen'] > p_buf:
                if t['persistence'] >= p_thresh:
                    completed_tracks.append(t)
            else:
                active_tracks_next.append(t)
        active_tracks = active_tracks_next

    # Flush remaining active tracks
    for t in active_tracks:
        if t['persistence'] >= p_thresh:
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