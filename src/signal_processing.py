import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
from src import config

def apply_bandpass_filter(audio, fs, lowcut=config.MIN_FREQ, highcut=config.MAX_FREQ, order=5):
    """Applies a Butterworth bandpass filter to the audio."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure frequencies are valid
    if low <= 0: low = 0.001
    if high >= 1: high = 0.999
    
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, audio)
    return y

def load_audio(filepath, target_fs=config.SAMPLE_RATE):
    """Loads a wav file, resamples, and applies bandpass filter."""
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

    # Bandpass Filter
    audio = apply_bandpass_filter(audio, fs)

    return audio, fs

def compute_psd(audio, fs, nperseg=config.N_FFT):
    """Computes Power Spectral Density using Welch's method."""
    # Ensure nperseg is not larger than signal
    if len(audio) < nperseg:
        # Pad or repeat
        repeats = int(np.ceil(nperseg / len(audio)))
        audio = np.tile(audio, repeats)[:nperseg]
    
    f, Pxx = signal.welch(audio, fs, nperseg=nperseg)
    Pxx_db = 10 * np.log10(Pxx + 1e-10)
    return f, Pxx_db

def compute_mfcc_from_spectrum(magnitude_spectrum, fs, n_fft, num_ceps=13):
    """
    Computes MFCCs from a magnitude spectrum frame.
    magnitude_spectrum: |X(f)| (N_FFT/2 + 1 bins)
    """
    # Define Mel Filterbank (simplified, constructed on fly or cached)
    # Ideally should cache this.
    num_filters = 26
    low_freq = 0
    high_freq = fs / 2
    
    # Convert to Mel scale
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    
    # Filterbank
    fbank = np.zeros((num_filters, int(n_fft / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    # Apply Filterbank
    filter_banks = np.dot(fbank, magnitude_spectrum**2) # Power
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 10 * np.log10(filter_banks)  # dB
    
    # DCT
    mfcc = dct(filter_banks, type=2, axis=0, norm='ortho')[:num_ceps]
    return mfcc


def compute_spectral_features(magnitude_spectrum, power_spectrum, freqs):
    """
    Computes spectral features for a single frame.
    magnitude_spectrum: |X(f)|
    power_spectrum: |X(f)|^2
    freqs: frequency bins
    """
    # Avoid division by zero
    sum_power = np.sum(power_spectrum) + 1e-10
    sum_mag = np.sum(magnitude_spectrum) + 1e-10

    # 1. Spectral Centroid
    centroid = np.sum(freqs * power_spectrum) / sum_power

    # 2. Spectral Flatness (Geometric Mean / Arithmetic Mean)
    # limit to > 0
    ps_safe = power_spectrum + 1e-10
    gmean = np.exp(np.mean(np.log(ps_safe)))
    amean = np.mean(ps_safe)
    flatness = gmean / (amean + 1e-10)

    # 3. RMS (approx from frequency domain via Parseval's)
    # Energy is proportional to sum of power spectrum
    # We can just use sqrt(mean(power)) as a feature proportional to RMS
    rms = np.sqrt(np.mean(power_spectrum))

    # PAPR proxy: Max Power / Mean Power
    papr = np.max(power_spectrum) / (np.mean(power_spectrum) + 1e-10)

    return {
        'centroid': centroid,
        'flatness': flatness,
        'rms': rms,
        'papr': papr
    }

def compute_spectrogram_and_peaks(audio, fs, nperseg=config.N_FFT, noverlap=None):
    """Computes spectrogram and finds peaks per time frame."""
    
    # Handle short audio by Wrapping (Tiling)
    # This maintains frequency resolution for XGBoost/Spectral features
    if len(audio) < nperseg:
        # print(f"Warning: Audio length {len(audio)} < nperseg {nperseg}. Wrapping signal.")
        repeats = int(np.ceil(nperseg / len(audio))) + 1
        audio = np.tile(audio, repeats)
        # We don't truncate strictly to nperseg to allow at least one hop if possible, 
        # but let's ensure it's at least nperseg.
        # STFT will handle the rest based on overlap.

    if noverlap is None:
        noverlap = nperseg - config.HOP_LENGTH
    
    # Safety check if noverlap >= nperseg (shouldn't happen with fixed nperseg)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    f, t, Zxx = signal.stft(audio, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Zxx is complex: (freqs, times)
    magnitude = np.abs(Zxx)
    Pxx = magnitude**2
    Pxx_db = 10 * np.log10(Pxx + 1e-10)

    peaks_per_frame = []
    spectral_features = []
    
    prev_magnitude = None
    prev_rms = None

    for i in range(Pxx_db.shape[1]):
        # Per-frame spectral features
        mag_frame = magnitude[:, i]
        pow_frame = Pxx[:, i]
        
        feats = compute_spectral_features(mag_frame, pow_frame, f)
        
        # MFCCs
        # Note: nperseg here acts as N_FFT
        mfccs = compute_mfcc_from_spectrum(mag_frame, fs, nperseg, num_ceps=13)
        feats['mfcc'] = mfccs

        # Calculate Flux (Delta Magnitude)
        if prev_magnitude is None:
            flux = 0.0
        else:
            flux = np.linalg.norm(mag_frame - prev_magnitude)
        
        feats['flux'] = flux
        
        # Calculate Delta RMS (Energy Velocity)
        curr_rms = feats['rms']
        if prev_rms is None:
            delta_rms = 0.0
        else:
            delta_rms = curr_rms - prev_rms
        
        feats['delta_rms'] = delta_rms

        spectral_features.append(feats)
        prev_magnitude = mag_frame
        prev_rms = curr_rms

        # Peak Detection
        psd_frame = Pxx_db[:, i]
        nf = estimate_noise_floor(psd_frame)
        peaks = find_significant_peaks(f, psd_frame, nf)
        peaks_per_frame.append(peaks)

    return f, t, Pxx_db, peaks_per_frame, spectral_features

def estimate_noise_floor(psd_db, window_size=config.NOISE_FLOOR_WINDOW):
    """Estimates the noise floor using a median filter."""
    return ndimage.median_filter(psd_db, size=window_size)

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
    height_threshold = max_h - 120 # Even more permissive

    peaks, properties = signal.find_peaks(psd_db,
                                          prominence=min_prominence,
                                          distance=min_dist,
                                          height=height_threshold)

    detected_peaks = []

    for p_idx in peaks:
        if not valid_mask[p_idx]:
            continue

        f = freqs[p_idx]
        p = psd_db[p_idx]
        nf = noise_floor_db[p_idx]
        snr = p - nf

        # Enforce Minimum SNR from Config
        if snr < config.SNR_THRESHOLD:
            continue

        detected_peaks.append({
            'freq': f,
            'power': p,
            'noise_floor': nf,
            'snr': snr,
            'idx': p_idx
        })

    # Sort by Power (descending) and keep top N
    detected_peaks.sort(key=lambda x: x['power'], reverse=True)
    detected_peaks = detected_peaks[:max_peaks]

    # Sort back by Frequency for easier harmonic analysis
    detected_peaks.sort(key=lambda x: x['freq'])

    return detected_peaks
