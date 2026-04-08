from harmonic_detection_v2 import summarize_tracks, track_harmonics


def add_quality_aware_harmonic_features(features, peaks_per_frame, frame_times=None, parallel=True):
    """
    Update an existing feature dictionary with quality-aware harmonic metrics.

    This mirrors the old `harmonic_score` flow, but the score is no longer based only
    on persistence. It also adds a separate `harmonic_quality` feature.
    """
    if frame_times is None:
        total_frames = len(peaks_per_frame)
    else:
        total_frames = len(frame_times)

    tracks = track_harmonics(peaks_per_frame, frame_times, parallel=parallel)
    features.update(summarize_tracks(tracks, total_frames))
    return tracks, features
