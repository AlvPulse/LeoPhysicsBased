import numpy as np
import scipy.signal as signal
from src import signal_processing
from src import config
import os

def debug_file(filename):
    print(f"DEBUG: Processing {filename}")
    audio, fs = signal_processing.load_audio(filename)
    if audio is None:
        print("Failed to load audio.")
        return

    print(f"Audio: FS={fs}, Duration={len(audio)/fs:.2f}s, Min={audio.min():.2f}, Max={audio.max():.2f}")

    f, psd = signal_processing.compute_psd(audio, fs)
    print(f"PSD: Min={psd.min():.2f} dB, Max={psd.max():.2f} dB")

    nf = signal_processing.estimate_noise_floor(psd)
    print(f"Noise Floor: Min={nf.min():.2f} dB, Max={nf.max():.2f} dB")

    peaks = signal_processing.find_significant_peaks(f, psd, nf)
    print(f"Found {len(peaks)} peaks.")
    for p in peaks:
        print(f"  - {p['freq']:.1f}Hz: Power={p['power']:.1f}dB, SNR={p['snr']:.1f}dB")

if __name__ == "__main__":
    debug_file("debug_data/yes/Autel_Evo_II_20.wav")
