import time
import random
import sys
import bisect
from unittest.mock import MagicMock

class Config:
    HARMONIC_MIN_SNR = 5
    HARMONIC_MIN_POWER = 10
    TOLERANCE = 0.05
    MAX_FREQ = 2000
    MIN_HARMONICS = 3
    MISSING_HARMONIC_PENALTY = 0.8
    PERSISTENCE_BUFFER = 5
    PERSISTENCE_THRESHOLD = 3

sys.modules['src.config'] = Config
sys.modules['numpy'] = MagicMock()

def calculate_quality(harmonic):
    snr_norm = min(max(harmonic['snr'], 0), 50) * 0.01
    pwr_norm = min(max(harmonic['power'] + 100, 0), 100) * 0.003
    drift_score = max(0, 1.0 - harmonic['drift'])
    return snr_norm + pwr_norm + (0.2 * drift_score)

def detect_harmonics_iterative(peaks, max_candidates=5, snr_threshold=None, power_threshold=None, tolerance=None):
    if not peaks:
        return []

    snr_threshold = snr_threshold if snr_threshold is not None else Config.HARMONIC_MIN_SNR
    power_threshold = power_threshold if power_threshold is not None else Config.HARMONIC_MIN_POWER
    tolerance = tolerance if tolerance is not None else Config.TOLERANCE

    max_freq_limit = Config.MAX_FREQ * 1.1
    min_harmonics = Config.MIN_HARMONICS
    missing_penalty = Config.MISSING_HARMONIC_PENALTY

    peaks_sorted_freq = sorted(peaks, key=lambda x: x['freq'])
    num_peaks = len(peaks_sorted_freq)
    freqs = [p['freq'] for p in peaks_sorted_freq]

    candidates = []

    for i in range(num_peaks):
        base_peak = peaks_sorted_freq[i]
        f0 = freqs[i]

        harmonics = []

        base_harmonic = base_peak.copy()
        base_harmonic['harmonic_index'] = 1
        base_harmonic['drift'] = 0.0

        snr_norm = base_harmonic['snr']
        if snr_norm < 0: snr_norm = 0
        elif snr_norm > 50: snr_norm = 50
        snr_norm *= 0.01

        pwr_norm = base_harmonic['power'] + 100
        if pwr_norm < 0: pwr_norm = 0
        elif pwr_norm > 100: pwr_norm = 100
        pwr_norm *= 0.003

        base_harmonic['quality'] = snr_norm + pwr_norm + 0.2

        harmonics.append(base_harmonic)

        current_harmonic_idx = 2
        consecutive_misses = 0

        while consecutive_misses <= 2:
            target_freq = f0 * current_harmonic_idx
            if target_freq > max_freq_limit:
                break

            best_match = None
            min_dist = float('inf')

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
                    h_drift = min_dist / target_freq
                    h_info['drift'] = h_drift

                    snr_norm = h_info['snr']
                    if snr_norm < 0: snr_norm = 0
                    elif snr_norm > 50: snr_norm = 50
                    snr_norm *= 0.01

                    pwr_norm = h_info['power'] + 100
                    if pwr_norm < 0: pwr_norm = 0
                    elif pwr_norm > 100: pwr_norm = 100
                    pwr_norm *= 0.003

                    drift_score = 1.0 - h_drift
                    if drift_score < 0: drift_score = 0

                    h_info['quality'] = snr_norm + pwr_norm + (0.2 * drift_score)

                    harmonics.append(h_info)
                consecutive_misses = 0
            else:
                consecutive_misses += 1

            current_harmonic_idx += 1

        if len(harmonics) >= min_harmonics:
            total_quality = 0.0
            total_drift = 0.0
            total_power = 0.0
            found_indices = set()
            for h in harmonics:
                total_quality += h['quality']
                total_drift += h['drift']
                # Weighted power sum for observability
                # Wait, what weights? Let's just sum power * quality for now, or maybe just simple power
                # "considering summation of its weighted harmonics" - we can use quality as the weight
                total_power += h['power'] * h['quality']
                found_indices.add(h['harmonic_index'])

            avg_drift = total_drift / len(harmonics)
            score = total_quality * (1.0 - avg_drift)

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
                'signal_power': total_power # added for observability
            })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:max_candidates]


def track_harmonics(peaks_per_frame, times):
    active_tracks = []
    completed_tracks = []

    tol = Config.TOLERANCE
    p_buf = Config.PERSISTENCE_BUFFER
    p_thresh = Config.PERSISTENCE_THRESHOLD

    for frame_idx, peaks in enumerate(peaks_per_frame):
        candidates = detect_harmonics_iterative(peaks, max_candidates=5)

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

            for t_idx, track in enumerate(active_tracks):
                if t_idx in matched_track_indices: continue

                track_freq = track['freq']
                dist = abs(track_freq - cand_freq)
                if dist < (track_freq * tol) and dist < best_dist:
                    best_dist = dist
                    best_track_idx = t_idx

            if best_track_idx != -1:
                track = active_tracks[best_track_idx]
                track['persistence'] += 1
                track['last_seen'] = frame_idx
                track['freq'] = 0.9 * track['freq'] + 0.1 * cand_freq

                cand_score = cand['score']
                if cand_score > track['max_score']:
                    track['max_score'] = cand_score
                    track['best_candidate'] = cand
                    track['best_frame_idx'] = frame_idx
                    # observability properties
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
                    # observability properties
                    'base_freq': cand['base_freq'],
                    'signal_power': cand['signal_power']
                })

        active_tracks_next = []
        for t in active_tracks:
            if frame_idx - t['last_seen'] > p_buf:
                if t['persistence'] >= p_thresh:
                    completed_tracks.append(t)
            else:
                active_tracks_next.append(t)
        active_tracks = active_tracks_next

    for t in active_tracks:
        if t['persistence'] >= p_thresh:
            completed_tracks.append(t)

    completed_tracks.sort(key=lambda x: x['max_score'], reverse=True)
    return completed_tracks

def run_pipeline():
    peaks_per_frame = []
    # Less peaks per frame might be more realistic? Let's use 100 peaks per frame to match previous test
    for f in range(100):
        peaks = []
        for i in range(100):
            peaks.append({'freq': random.uniform(50, 2500), 'snr': random.uniform(0, 20), 'power': random.uniform(0, 50), 'drift': random.uniform(0, 0.1)})
        peaks.sort(key=lambda x: x['freq'])
        peaks_per_frame.append(peaks)

    start = time.perf_counter()
    active_series = track_harmonics(peaks_per_frame, None)
    end = time.perf_counter()

    return end - start

random.seed(42)
t = run_pipeline()
print(f"Time per frame (separate detect_harmonics_iterative) V9: {(t*1000)/100:.4f} ms")
