import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.io.wavfile as wavfile
from src import config

def load_audio(filepath, target_fs=config.SAMPLE_RATE):
    """Loads a wav file and resamples if necessary."""
    try:
        fs, audio = wavfile.read(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

    # Convert to mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Normalize
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Resample if needed
    if fs != target_fs:
        num_samples = int(len(audio) * target_fs / fs)
        audio = signal.resample(audio, num_samples)
        fs = target_fs

    return audio, fs

def compute_psd(audio, fs, nperseg=config.N_FFT):
    """Computes Power Spectral Density using Welch's method."""
    f, Pxx = signal.welch(audio, fs, nperseg=nperseg)
    Pxx_db = 10 * np.log10(Pxx + 1e-10)
    return f, Pxx_db

def compute_spectrogram_and_peaks(audio, fs, nperseg=config.N_FFT, noverlap=None):
    """Computes spectrogram and finds peaks per time frame."""
    if noverlap is None:
        noverlap = nperseg - config.HOP_LENGTH

    f, t, Zxx = signal.stft(audio, fs, nperseg=nperseg, noverlap=noverlap)
    Pxx = np.abs(Zxx)**2
    Pxx_db = 10 * np.log10(Pxx + 1e-10)

    peaks_per_frame = []

    for i in range(Pxx_db.shape[1]):
        psd_frame = Pxx_db[:, i]
        nf = estimate_noise_floor(psd_frame)
        peaks = find_significant_peaks(f, psd_frame, nf)
        peaks_per_frame.append(peaks)

    return f, t, Pxx_db, peaks_per_frame

def estimate_noise_floor(psd_db, window_size=config.NOISE_FLOOR_WINDOW):
    """Estimates the noise floor using a median filter."""
    return ndimage.median_filter(psd_db, size=window_size)

def parabolic_interpolation(y, i):
    """
    Refines peak location using parabolic interpolation.
    Returns: (refined_index_offset, refined_magnitude)
    """
    if i <= 0 or i >= len(y) - 1:
        return 0.0, y[i]

    alpha = y[i-1]
    beta = y[i]
    gamma = y[i+1]

    denominator = alpha - 2*beta + gamma
    if denominator == 0:
        return 0.0, beta

    delta = 0.5 * (alpha - gamma) / denominator
    refined_mag = beta - 0.25 * (alpha - gamma) * delta
    return delta, refined_mag

def integrated_power(y_db, i, radius=2):
    """
    Calculates integrated power around a peak index in linear scale.
    Returns power in dB.
    """
    start = max(0, i - radius)
    end = min(len(y_db), i + radius + 1)

    # Convert dB to linear power
    y_linear = 10 ** (y_db[start:end] / 10.0)
    total_power = np.sum(y_linear)

    return 10 * np.log10(total_power + 1e-10)

def find_significant_peaks(freqs, psd_db, noise_floor_db,
                         min_prominence=config.PEAK_PROMINENCE,
                         min_dist=config.PEAK_DISTANCE,
                         max_peaks=config.NUM_PEAKS_TO_KEEP):
    """
    Finds peaks that are significantly above the noise floor.
    Returns: list of dicts {'freq': f, 'power': p, 'snr': snr, 'idx': i}
    """
    # Calculate SNR curve
    snr_curve = psd_db - noise_floor_db

    # We restrict search to defined frequency range
    valid_mask = (freqs >= config.MIN_FREQ) & (freqs <= config.MAX_FREQ)

    max_h = np.max(psd_db)
    height_threshold = max_h - 120 # Permissive threshold

    peaks, properties = signal.find_peaks(psd_db,
                                          prominence=min_prominence,
                                          distance=min_dist,
                                          height=height_threshold)

    detected_peaks = []
    bin_width = freqs[1] - freqs[0]

    for p_idx in peaks:
        if not valid_mask[p_idx]:
            continue

        # 1. Refine Frequency & Power (Parabolic Interpolation)
        delta, refined_mag = parabolic_interpolation(psd_db, p_idx)
        refined_freq = freqs[p_idx] + delta * bin_width

        # 2. Integrated Power (to account for spectral leakage)
        int_pwr = integrated_power(psd_db, p_idx, radius=1)

        # 3. Noise Floor at peak
        nf = noise_floor_db[p_idx]

        # 4. Refined SNR
        snr = refined_mag - nf # Use interpolated peak magnitude for SNR

        # Enforce Minimum SNR from Config
        if snr < config.SNR_THRESHOLD:
            continue

        detected_peaks.append({
            'freq': refined_freq,
            'power': int_pwr, # Use integrated power for "Power" metric
            'peak_power': refined_mag, # Store peak magnitude separately if needed
            'noise_floor': nf,
            'snr': snr,
            'idx': p_idx
        })

    # Sort by Integrated Power (descending) and keep top N
    detected_peaks.sort(key=lambda x: x['power'], reverse=True)
    detected_peaks = detected_peaks[:max_peaks]

    # Sort back by Frequency for harmonic analysis
    detected_peaks.sort(key=lambda x: x['freq'])

    return detected_peaks
