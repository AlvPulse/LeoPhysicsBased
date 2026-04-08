import argparse
import importlib.util
import pathlib
import random
import sys
import time
import types

ROOT = pathlib.Path(__file__).resolve().parent
OLD_PATH = ROOT / 'src' / 'harmonic_detection.py'
NEW_PATH = ROOT / 'harmonic_detection_v2.py'


class _FakeNumpy(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            zeros=lambda shape, dtype=None: [0.0] * shape[0],
            float32=float,
        )


def load_module(module_name, path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ModuleNotFoundError as exc:
        if exc.name != 'numpy':
            raise
        sys.modules.setdefault('numpy', _FakeNumpy())
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def make_frame(rng, num_peaks=20):
    peaks = []
    base_freq = rng.uniform(120.0, 600.0)

    for harmonic_index in range(1, 7):
        freq = base_freq * harmonic_index * rng.uniform(0.985, 1.015)
        if freq > 2000.0:
            break
        peaks.append({
            'freq': freq,
            'power': rng.uniform(-45.0, -10.0),
            'snr': rng.uniform(8.0, 25.0),
        })

    while len(peaks) < num_peaks:
        peaks.append({
            'freq': rng.uniform(100.0, 2000.0),
            'power': rng.uniform(-80.0, -15.0),
            'snr': rng.uniform(0.0, 20.0),
        })

    peaks.sort(key=lambda peak: peak['freq'])
    return peaks


def build_dataset(seed, frame_count, num_peaks):
    rng = random.Random(seed)
    frames = [make_frame(rng, num_peaks=num_peaks) for _ in range(frame_count)]
    times = [frame_idx * 0.125 for frame_idx in range(frame_count)]
    return frames, times


def benchmark(module, frames, times, loops, parallel=None):
    for peaks in frames[:5]:
        module.detect_harmonics_iterative(peaks)
    if parallel is None:
        module.track_harmonics(frames, times)
    else:
        module.track_harmonics(frames, times, parallel=parallel)

    start = time.perf_counter()
    for _ in range(loops):
        for peaks in frames:
            module.detect_harmonics_iterative(peaks)
    detect_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(loops):
        if parallel is None:
            module.track_harmonics(frames, times)
        else:
            module.track_harmonics(frames, times, parallel=parallel)
    track_seconds = time.perf_counter() - start

    return detect_seconds, track_seconds


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


def main():
    parser = argparse.ArgumentParser(description='Compare old and new harmonic detection runtime.')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--peaks', type=int, default=20)
    parser.add_argument('--loops', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    old_module = load_module('harmonic_detection_old', OLD_PATH)
    new_module = load_module('harmonic_detection_v2', NEW_PATH)
    frames, times = build_dataset(args.seed, args.frames, args.peaks)

    old_detect, old_track = benchmark(old_module, frames, times, args.loops)
    new_detect, new_track = benchmark(new_module, frames, times, args.loops, parallel=False)
    new_detect_parallel, new_track_parallel = benchmark(new_module, frames, times, args.loops, parallel=True)

    old_desc = describe_tracks(old_module, frames, times)
    new_desc = describe_tracks(new_module, frames, times, parallel=False)

    print('Dataset:', {'frames': args.frames, 'peaks_per_frame': args.peaks, 'loops': args.loops, 'seed': args.seed})
    print('Old module:', {'detect_s': round(old_detect, 4), 'track_s': round(old_track, 4), **old_desc})
    print('New module (serial):', {'detect_s': round(new_detect, 4), 'track_s': round(new_track, 4), **new_desc})
    print('New module (pipelined):', {'detect_s': round(new_detect_parallel, 4), 'track_s': round(new_track_parallel, 4)})

    if new_detect > 0:
        print('Detect speedup (serial):', round(old_detect / new_detect, 2), 'x')
    if new_track > 0:
        print('Track speedup (serial):', round(old_track / new_track, 2), 'x')
    if new_track_parallel > 0:
        print('Track speedup (pipelined):', round(old_track / new_track_parallel, 2), 'x')


if __name__ == '__main__':
    main()
