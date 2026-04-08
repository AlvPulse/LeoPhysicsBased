from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src import config


def calculate_quality(harmonic):
    """Score a single harmonic using SNR, power, and drift."""
    #w_snr = 0.5
    w_pwr = 1
    #w_drift = 0.2
    
    #snr_norm = min(max(harmonic['snr'], 0), 50) / 50.0
    pwr_norm = min(max(harmonic['power'] + 100, 0), 100) / 100.0
    #drift_score = max(0.0, 1.0 - harmonic['drift'])
    #print("snr_norm",snr_norm)
    #print("pwr_norm",pwr_norm)
    #print("drif_score",drift_score)
    return (w_pwr * pwr_norm) 


# def _prepare_peaks(peaks, snr_threshold, power_threshold):
#     peaks_sorted_freq = sorted(peaks, key=lambda peak: peak['freq'])

    
#     print("length of lists",len(valid_freqs), len(peaks_sorted_freq))
#     return peaks_sorted_freq, valid_peaks, valid_freqs


def _build_harmonic_entry(peak, harmonic_index, drift):
    harmonic = peak.copy()
    harmonic['harmonic_index'] = harmonic_index
    harmonic['drift'] = drift
    harmonic['quality'] = calculate_quality(harmonic)
    return harmonic


def _score_candidate(harmonics):
    total_quality = sum(harmonic['quality'] for harmonic in harmonics)
    avg_drift = sum(harmonic['drift'] for harmonic in harmonics) / len(harmonics)
    max_found_idx = harmonics[-1]['harmonic_index']
    found_indices = {harmonic['harmonic_index'] for harmonic in harmonics}

    missing_low_order_count = 0
    for harmonic_index in range(2, min(6, max_found_idx)):
        if harmonic_index not in found_indices:
            missing_low_order_count += 1

    penalty = config.MISSING_HARMONIC_PENALTY ** missing_low_order_count
    coverage = len(harmonics) / max(max_found_idx, 1)
    mean_quality = total_quality / len(harmonics)
    drift_factor = max(0.0, 1.0 - avg_drift)

    return {
        'score': total_quality * drift_factor * penalty,
        'quality_score': mean_quality * coverage * drift_factor * penalty,
        'quality_mean': mean_quality,
        'avg_drift': avg_drift,
        'coverage': coverage,
        'missing_low_order_count': missing_low_order_count,
    }

def _score_candidate_simple(peaks):
    """Sum of power for all peaks in the harmonic set."""
    total_power = np.sum([min(max(peak['power'] + 100, 0), 100) / 100.0 for peak in peaks])
    return {
        'score': total_power,
        'quality_score': total_power / len(peaks) if peaks else 0,
    }


    


def detect_harmonics_iterative(
    peaks,
    max_candidates=2,
    snr_threshold=None,
    power_threshold=None,
    tolerance=None,
):
    candidates = []
    if snr_threshold is None:
        snr_threshold = config.HARMONIC_MIN_SNR
    if power_threshold is None:
        power_threshold = config.HARMONIC_MIN_POWER
    if tolerance is None:
        tolerance = config.TOLERANCE

    if not peaks:
        return candidates

    # peaks_sorted_freq, valid_peaks, valid_freqs = _prepare_peaks(
    #     peaks,
    #     snr_threshold,
    #     power_threshold,
    # )

    candidate_roots = peaks

    freq_arr = np.array([p['freq'] for p in candidate_roots], dtype=np.float32)

    for i, base_peak in enumerate(candidate_roots):
        f0 = base_peak['freq']

        # Only look at freqs >= f0
        higher = freq_arr[i:]
        ratios = higher / f0

        # Check how close each ratio is to the nearest integer
        nearest_int = np.round(ratios)
        drift = np.abs(ratios - nearest_int) / nearest_int  # relative drift

        matched = np.where((nearest_int >= 1) & (drift <= tolerance))[0]
        harmonics = [
            _build_harmonic_entry(candidate_roots[i + j], int(nearest_int[j]), float(drift[j]))
            for j in matched
        ]

        if len(harmonics) >= config.MIN_HARMONICS:
            candidates.append({
                'base_freq': f0,
                'harmonics': harmonics,
                **_score_candidate(harmonics),
            })

    candidates.sort(
        key=lambda c: (c['score'], c['quality_score'], len(c['harmonics'])),
        reverse=True,
    )
    return candidates[:max_candidates]

#         if len(matched) > config.MIN_HARMONICS:
#             candidates.append({
#                 'base_freq': f0,
#                 'harmonics': [freq_arr[i] for i in matched_indices],
# #                **_score_candidate([peaks[i] for i in matched_indices]),
#                 **_score_candidate_simple([peaks[i] for i in matched_indices])
#             })

#     candidates.sort(key=lambda c: ( len(c['harmonics'])),
#         reverse=True,
#     )    
#     return candidates[:max_candidates]


class _OrderedCandidatePipeline:
    """Keep candidate detection queued in the background while tracking stays ordered."""

    def __init__(self, peaks_per_frame, max_candidates, parallel, max_workers, prefetch):
        self.peaks_per_frame = peaks_per_frame
        self.max_candidates = max_candidates
        self.parallel = parallel if parallel is not None else len(peaks_per_frame) >= 8
        self.max_workers = max_workers or min(4, max(1, len(peaks_per_frame)))
        self.prefetch = prefetch or min(len(peaks_per_frame), self.max_workers * 2)

    def __iter__(self):
        if not self.parallel or len(self.peaks_per_frame) < 2:
            for peaks in self.peaks_per_frame:
                yield detect_harmonics_iterative(peaks, max_candidates=self.max_candidates)
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            next_submit = 0
            next_yield = 0

            while next_submit < self.prefetch:
                futures[next_submit] = executor.submit(
                    detect_harmonics_iterative,
                    self.peaks_per_frame[next_submit],
                    self.max_candidates,
                )
                next_submit += 1

            while next_yield < len(self.peaks_per_frame):
                future = futures.pop(next_yield)
                yield future.result()

                if next_submit < len(self.peaks_per_frame):
                    futures[next_submit] = executor.submit(
                        detect_harmonics_iterative,
                        self.peaks_per_frame[next_submit],
                        self.max_candidates,
                    )
                    next_submit += 1

                next_yield += 1


def iter_frame_candidates(
    peaks_per_frame,
    max_candidates=5,
    parallel=None,
    max_workers=None,
    prefetch=None,
):
    return iter(
        _OrderedCandidatePipeline(
            peaks_per_frame,
            max_candidates=max_candidates,
            parallel=parallel,
            max_workers=max_workers,
            prefetch=prefetch,
        )
    )


def _finalize_track(track):
    persistence = max(track['persistence'], 1)
    quality_mean = track['quality_total'] / persistence
    track['quality_mean'] = quality_mean
    track['quality_score'] = (0.6 * quality_mean) + (0.4 * track['quality_peak'])
    return track


def track_harmonics(
    peaks_per_frame,
    times=None,
    parallel=None,
    max_workers=None,
    prefetch=None,
    max_candidates=5,
):
    active_tracks = []
    completed_tracks = []
    candidate_iter = iter_frame_candidates(
        peaks_per_frame,
        max_candidates=max_candidates,
        parallel=parallel,
        max_workers=max_workers,
        prefetch=prefetch,
    )

    for frame_idx, candidates in enumerate(candidate_iter):
        matched_track_indices = set()

        for candidate in candidates:
            best_track_idx = -1
            best_dist = float('inf')

            for track_idx, track in enumerate(active_tracks):
                if track_idx in matched_track_indices:
                    continue

                dist = abs(track['freq'] - candidate['base_freq'])
                if dist < (track['freq'] * config.TOLERANCE) and dist < best_dist:
                    best_dist = dist
                    best_track_idx = track_idx

            if best_track_idx == -1:
                active_tracks.append({
                    'freq': candidate['base_freq'],
                    'persistence': 1,
                    'start_frame': frame_idx,
                    'last_seen': frame_idx,
                    'max_score': candidate['score'],
                    'best_candidate': candidate,
                    'best_frame_idx': frame_idx,
                    'quality_total': candidate.get('quality_score', 0.0),
                    'quality_peak': candidate.get('quality_score', 0.0),
                    'avg_drift':candidate.get('avg_drift')
                })
                continue

            track = active_tracks[best_track_idx]
            track['persistence'] += 1
            track['last_seen'] = frame_idx
            track['freq'] = 0.9 * track['freq'] + 0.1 * candidate['base_freq']
            track['quality_total'] += candidate.get('quality_score', 0.0)
            track['quality_peak'] = max(track['quality_peak'], candidate.get('quality_score', 0.0))

            if candidate['score'] > track['max_score']:
                track['max_score'] = candidate['score']
                track['best_candidate'] = candidate
                track['best_frame_idx'] = frame_idx

            matched_track_indices.add(best_track_idx)

        active_tracks_next = []
        for track in active_tracks:
            if frame_idx - track['last_seen'] > config.PERSISTENCE_BUFFER:
                if track['persistence'] >= config.PERSISTENCE_THRESHOLD:
                    completed_tracks.append(_finalize_track(track))
            else:
                active_tracks_next.append(track)
        active_tracks = active_tracks_next

    for track in active_tracks:
        if track['persistence'] >= config.PERSISTENCE_THRESHOLD:
            completed_tracks.append(_finalize_track(track))

    completed_tracks.sort(
        key=lambda track: (track['quality_score'], track['max_score'], track['persistence']),
        reverse=True,
    )
    return completed_tracks


def summarize_tracks(tracks, total_frames):
    if not tracks or total_frames <= 0:
        return {
            'harmonic_score': 0.0,
            'harmonic_quality': 0.0,
            'harmonic_persistence': 0.0,
            'harmonic_track_count': 0,
        }

    persistences = [track['persistence'] for track in tracks]
    total_persistence = sum(persistences)
    weighted_quality = sum(
        track['persistence'] * track.get('quality_score', 0.0)
        for track in tracks
    ) / total_frames

    persistence_ratio = min(1.0, max(persistences) / total_frames)
    coverage_ratio = min(1.0, total_persistence / total_frames)
    top_track = tracks[0] if tracks else {}
    return {
        'harmonic_score_top': round(top_track.get('quality_score', 0.0), 3) if top_track else 0.0,
        'harmonic_quality': round(weighted_quality,3),
        'harmonic_persistence': round(persistence_ratio,3),
        'harmonic_track_count': len(tracks),
        'best_base_freq': round(top_track.get('freq', 0.0), 3) if top_track else 0.0,
        'avg_drif': round(top_track.get('avg_drift', 0.0), 3) if top_track else 0.0,
    }


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

    for harmonic in best['harmonics']:
        harmonic_index = harmonic['harmonic_index']
        if harmonic_index <= num_harmonics:
            vec_idx = (harmonic_index - 1) * 2
            vec[vec_idx] = min(max(harmonic['snr'], 0), 50) / 50.0
            vec[vec_idx + 1] = min(max(harmonic['power'] + 100, 0), 100) / 100.0

    return vec

def describe_tracks(module, frames, times, parallel=None):
    if parallel is None:
        tracks = module.track_harmonics(frames, times)
    else:
        tracks = module.track_harmonics(frames, times, parallel=parallel)

    top_track = tracks[0] if tracks else {}
    return {
        'track_count': len(tracks),
        'best_base_freq': round(top_track.get('freq', 0.0), 3) if top_track else 0.0,
        'best_persistence': top_track.get('persistence', 0),
        'best_quality': round(top_track.get('quality_score', 0.0), 4) if top_track else 0.0,
    }

