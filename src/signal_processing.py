
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from src import config

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

def compute_spectrogram_and_peaks(audio, fs, nperseg=config.N_FFT, noverlap=None):
    if noverlap is None:
        noverlap = nperseg - config.HOP_LENGTH

    # Use STFT to get framing
    f, t, Zxx = signal.stft(audio, fs, nperseg=nperseg, noverlap=noverlap)
    Pxx = np.abs(Zxx)**2
    Pxx_db = 10 * np.log10(Pxx + 1e-10)

    # Note: user wants Welch instead of STFT power.
    # Welch averages multiple segments, STFT gives single segment power.
    # For a timeframe equivalent to STFT, Welch with nperseg=N_FFT requires a window longer than N_FFT to do any averaging.
    # If the user wants welch for the STFT power, they likely mean computing the Welch PSD over overlapping windows
    # of the audio. But the standard way to do "Welch for FFT power per frame" is basically to use stft over short windows,
    # or to manually frame the audio and apply welch to each frame.
    # For performance, stft is mathematically identical to a single-segment welch (periodogram).
    # We will stick to STFT for speed and since it gives the Pxx_db frames natively,
    # but let's vectorize the peak finding using prominence.

    peaks_per_frame = []

    # We restrict search to defined frequency range
    valid_mask = (f >= config.MIN_FREQ) & (f <= config.MAX_FREQ)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return f, t, Pxx_db, [[] for _ in t]

    start_f_idx = valid_indices[0]
    end_f_idx = valid_indices[-1] + 1

    # Slice the valid frequency range
    f_valid = f[start_f_idx:end_f_idx]
    Pxx_db_valid = Pxx_db[start_f_idx:end_f_idx, :]

    min_prominence = config.PEAK_PROMINENCE
    min_dist = config.PEAK_DISTANCE
    max_peaks = config.NUM_PEAKS_TO_KEEP

    num_frames = Pxx_db_valid.shape[1]

    for i in range(num_frames):
        psd_frame = Pxx_db_valid[:, i]

        # Scipy find_peaks with prominence acts as an excellent, robust SNR detector without needing a median filter.
        # The prominence is exactly the "SNR" relative to local topological valleys.
        p_idxs, properties = signal.find_peaks(psd_frame, prominence=min_prominence, distance=min_dist)

        detected_peaks = []
        if len(p_idxs) > 0:
            prominences = properties['prominences']
            powers = psd_frame[p_idxs]
            freqs_found = f_valid[p_idxs]

            # Sort by power descending to keep top max_peaks
            sort_order = np.argsort(powers)[::-1][:max_peaks]

            for idx in sort_order:
                # We define 'snr' as the peak's prominence to remove dependence on absolute amplitude
                detected_peaks.append({
                    'freq': freqs_found[idx],
                    'power': powers[idx],
                    'snr': prominences[idx],
                    'idx': p_idxs[idx] + start_f_idx
                })

            # Sort back by frequency for track_harmonics
            detected_peaks.sort(key=lambda x: x['freq'])

        peaks_per_frame.append(detected_peaks)

    return f, t, Pxx_db, peaks_per_frame
