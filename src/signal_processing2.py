import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from multiprocessing import Pool
from src import config
import os

def load_audio(filepath, target_fs=config.SAMPLE_RATE):
    try:
        fs, audio = wavfile.read(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    if fs != target_fs:
        num_samples = int(len(audio) * target_fs / fs)
        audio = signal.resample(audio, num_samples)
        fs = target_fs

    return audio, fs

def compute_psd(audio, fs, nperseg=config.N_FFT):
    f, Pxx = signal.welch(audio, fs, nperseg=nperseg)
    Pxx_db = 10 * np.log10(Pxx + 1e-10)
    return f, Pxx_db

# Helper function for multiprocessing
def _process_frame(args):
    psd_frame, f_valid, start_f_idx, min_prominence, min_dist, max_peaks = args
    p_idxs, properties = signal.find_peaks(psd_frame, prominence=min_prominence, distance=min_dist)

    detected_peaks = []
    if len(p_idxs) > 0:
        prominences = properties['prominences']
        powers = psd_frame[p_idxs]
        freqs_found = f_valid[p_idxs]

        sort_order = np.argsort(powers)[::-1][:max_peaks]

        for idx in sort_order:
            detected_peaks.append({
                'freq': freqs_found[idx],
                'power': powers[idx],
                'snr': prominences[idx],
                'idx': p_idxs[idx] + start_f_idx
            })

        detected_peaks.sort(key=lambda x: x['freq'])

    return detected_peaks

def compute_spectrogram_and_peaks(audio, fs, nperseg=config.N_FFT, noverlap=None, parallel=False):
    if noverlap is None:
        noverlap = nperseg - config.HOP_LENGTH

    f, t, Zxx = signal.stft(audio, fs, nperseg=nperseg, noverlap=noverlap)
    Pxx = np.abs(Zxx)**2
    Pxx_db = 10 * np.log10(Pxx + 1e-10)

    valid_mask = (f >= config.MIN_FREQ) & (f <= config.MAX_FREQ)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return f, t, Pxx_db, [[] for _ in t]

    start_f_idx = valid_indices[0]
    end_f_idx = valid_indices[-1] + 1

    f_valid = f[start_f_idx:end_f_idx]
    Pxx_db_valid = Pxx_db[start_f_idx:end_f_idx, :]

    min_prominence = config.PEAK_PROMINENCE
    min_dist = config.PEAK_DISTANCE
    max_peaks = config.NUM_PEAKS_TO_KEEP

    num_frames = Pxx_db_valid.shape[1]

    if parallel and num_frames > 10:
        # Avoid process pool overhead for very short files
        pool_args = [
            (Pxx_db_valid[:, i], f_valid, start_f_idx, min_prominence, min_dist, max_peaks)
            for i in range(num_frames)
        ]

        # Use available CPUs
        num_cores = min(os.cpu_count() or 1, 8)
        with Pool(processes=num_cores) as pool:
            peaks_per_frame = pool.map(_process_frame, pool_args)
    else:
        peaks_per_frame = []
        for i in range(num_frames):
            peaks_per_frame.append(
                _process_frame((Pxx_db_valid[:, i], f_valid, start_f_idx, min_prominence, min_dist, max_peaks))
            )

    return f, t, Pxx_db, peaks_per_frame
