import numpy as np
from src import config

def calculate_quality(harmonic):
    """
    Calculates the quality factor for a single harmonic.
    Q = w1*SNR + w2*Power + w3*(1-Drift)
    """
    w_snr = 0.5
    w_pwr = 0.3
    w_drift = 0.2
    snr_norm = min(max(harmonic['snr'], 0), 50) / 50.0
    pwr_norm = min(max(harmonic['power'] + 100, 0), 100) / 100.0
    drift_score = max(0, 1.0 - harmonic['drift'])
    q = (w_snr * snr_norm) + (w_pwr * pwr_norm) + (w_drift * drift_score)
    return q

def detect_harmonics_iterative(peaks, max_candidates=5, snr_threshold=None, power_threshold=None, tolerance=None):
    candidates = []
    if snr_threshold is None: snr_threshold = config.HARMONIC_MIN_SNR
    if power_threshold is None: power_threshold = config.HARMONIC_MIN_POWER
    if tolerance is None: tolerance = config.TOLERANCE

    if not peaks:
        return candidates

    peaks_sorted_freq = sorted(peaks, key=lambda x: x['freq'])

    for i, base_peak in enumerate(peaks_sorted_freq):
        f0 = base_peak['freq']
        harmonics = []

        base_harmonic = base_peak.copy()
        base_harmonic['harmonic_index'] = 1
        base_harmonic['drift'] = 0.0
        base_harmonic['quality'] = calculate_quality(base_harmonic)
        harmonics.append(base_harmonic)

        current_harmonic_idx = 2
        consecutive_misses = 0
        max_misses = 2

        while True:
            target_freq = f0 * current_harmonic_idx
            if target_freq > config.MAX_FREQ * 1.1:
                break

            best_match = None
            min_dist = float('inf')

            for p in peaks_sorted_freq:
                if p['snr'] < snr_threshold: continue
                if p['power'] < power_threshold: continue

                lower_bound = target_freq * (1 - tolerance)
                upper_bound = target_freq * (1 + tolerance)

                if p['freq'] < lower_bound: continue
                if p['freq'] > upper_bound: break

                dist = abs(p['freq'] - target_freq)
                if dist < (target_freq * tolerance):
                    if dist < min_dist:
                        min_dist = dist
                        best_match = p

            if best_match:
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
            if consecutive_misses > max_misses:
                break

        if len(harmonics) >= config.MIN_HARMONICS:
            total_quality = sum(h['quality'] for h in harmonics)
            avg_drift = sum(h['drift'] for h in harmonics) / len(harmonics)
            score = total_quality * (1.0 - avg_drift)

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

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:max_candidates]

def track_harmonics(peaks_per_frame, times, spectral_features_per_frame=None):
    """
    Tracks harmonic candidates across frames.
    Optionally accepts spectral_features_per_frame to store best snapshot context.
    """
    active_tracks = []
    completed_tracks = []

    for frame_idx, peaks in enumerate(peaks_per_frame):
        candidates = detect_harmonics_iterative(peaks, max_candidates=5)
        matched_track_indices = set()

        # Get features for this frame if available
        current_frame_features = {}
        if spectral_features_per_frame and frame_idx < len(spectral_features_per_frame):
            current_frame_features = spectral_features_per_frame[frame_idx]

        for cand in candidates:
            best_track_idx = -1
            best_dist = float('inf')

            for t_idx, track in enumerate(active_tracks):
                if t_idx in matched_track_indices: continue

                dist = abs(track['freq'] - cand['base_freq'])
                if dist < (track['freq'] * config.TOLERANCE):
                    if dist < best_dist:
                        best_dist = dist
                        best_track_idx = t_idx

            if best_track_idx != -1:
                track = active_tracks[best_track_idx]
                track['persistence'] += 1
                track['last_seen'] = frame_idx
                track['freq'] = 0.9 * track['freq'] + 0.1 * cand['base_freq']

                if cand['score'] > track['max_score']:
                    track['max_score'] = cand['score']
                    track['best_candidate'] = cand
                    track['best_frame_idx'] = frame_idx
                    track['best_frame_features'] = current_frame_features

                matched_track_indices.add(best_track_idx)
            else:
                new_track = {
                    'freq': cand['base_freq'],
                    'persistence': 1,
                    'start_frame': frame_idx,
                    'last_seen': frame_idx,
                    'max_score': cand['score'],
                    'best_candidate': cand,
                    'best_frame_idx': frame_idx,
                    'best_frame_features': current_frame_features
                }
                active_tracks.append(new_track)

        active_tracks_next = []
        for t in active_tracks:
            if frame_idx - t['last_seen'] > config.PERSISTENCE_BUFFER:
                if t['persistence'] >= config.PERSISTENCE_THRESHOLD:
                    completed_tracks.append(t)
            else:
                active_tracks_next.append(t)
        active_tracks = active_tracks_next

    for t in active_tracks:
        if t['persistence'] >= config.PERSISTENCE_THRESHOLD:
            completed_tracks.append(t)

    completed_tracks.sort(key=lambda x: x['max_score'], reverse=True)
    return completed_tracks

def extract_linear_features(candidates, num_harmonics=10):
    vec = np.zeros(num_harmonics * 2, dtype=np.float32)
    if not candidates:
        return vec

    if isinstance(candidates, list) and len(candidates) > 0:
        if 'best_candidate' in candidates[0]:
            best = candidates[0]['best_candidate']
        else:
            best = candidates[0]
    elif isinstance(candidates, dict):
         best = candidates
    else:
        return vec

    for h in best['harmonics']:
        idx = h['harmonic_index']
        if idx <= num_harmonics:
            vec_idx = (idx - 1) * 2
            vec[vec_idx] = min(max(h['snr'], 0), 50) / 50.0
            vec[vec_idx+1] = min(max(h['power'] + 100, 0), 100) / 100.0

    return vec
