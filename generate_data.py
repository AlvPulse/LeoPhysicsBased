import numpy as np
import scipy.io.wavfile as wavfile
import os
import random

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_harmonic_signal(fs, duration, f0, num_harmonics, snr_db):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Add fundamental and harmonics
    for i in range(1, num_harmonics + 1):
        freq = f0 * i
        # Amplitude decays with harmonic index (1/i) or random variation
        amplitude = (1.0 / i) * random.uniform(0.8, 1.2)
        phase = random.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Normalize signal power to 1
    signal_power = np.mean(signal ** 2)
    signal = signal / np.sqrt(signal_power)

    # Add noise
    noise = np.random.normal(0, 1, len(t))
    noise_power = np.mean(noise ** 2)

    # Calculate required noise amplitude for target SNR
    # SNR_db = 10 * log10(P_signal / P_noise)
    # P_noise_target = P_signal / (10 ** (SNR_db / 10))
    # But signal power is 1 now.
    target_noise_power = 1.0 / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    final_signal = signal + noise

    # Normalize to -1 to 1 for wav file
    max_val = np.max(np.abs(final_signal))
    if max_val > 0:
        final_signal = final_signal / max_val

    return final_signal

def generate_random_peaks_signal(fs, duration, num_peaks, snr_db):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Add random peaks (non-harmonic)
    # Ensure they are not multiples of each other to avoid accidental harmonics
    freqs = []
    for _ in range(num_peaks):
        while True:
            f = random.uniform(200, 4000)
            # Check collision with existing freqs (simple check)
            if all(abs(f - ef) > 50 for ef in freqs):
                freqs.append(f)
                break

    for f in freqs:
        amplitude = random.uniform(0.1, 1.0)
        phase = random.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * f * t + phase)

    # Normalize signal power to 1
    if len(freqs) > 0:
        signal_power = np.mean(signal ** 2)
        signal = signal / np.sqrt(signal_power)

    # Add noise
    noise = np.random.normal(0, 1, len(t))
    noise_power = np.mean(noise ** 2)
    target_noise_power = 1.0 / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    final_signal = signal + noise

    # Normalize
    max_val = np.max(np.abs(final_signal))
    if max_val > 0:
        final_signal = final_signal / max_val

    return final_signal

def main():
    fs = 44100
    ensure_dir("data/yes")
    ensure_dir("data/no")

    # Generate YES samples (Harmonic)
    print("Generating 'YES' samples...")
    for i in range(50):
        duration = random.uniform(2.0, 4.0)
        f0 = random.uniform(150, 800)
        num_harmonics = random.randint(3, 8)
        snr = random.uniform(5, 20) # dB

        sig = generate_harmonic_signal(fs, duration, f0, num_harmonics, snr)
        wavfile.write(f"data/yes/sample_{i:03d}.wav", fs, np.float32(sig))

    # Generate NO samples (Random Peaks / Noise)
    print("Generating 'NO' samples...")
    for i in range(50):
        duration = random.uniform(2.0, 4.0)
        num_peaks = random.randint(3, 10)
        snr = random.uniform(0, 15) # dB

        # 50% random peaks, 50% pure noise
        if random.random() > 0.5:
            sig = generate_random_peaks_signal(fs, duration, num_peaks, snr)
        else:
            # Pure noise
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            noise = np.random.normal(0, 1, len(t))
            sig = noise / np.max(np.abs(noise))

        wavfile.write(f"data/no/sample_{i:03d}.wav", fs, np.float32(sig))

    print("Done.")

if __name__ == "__main__":
    main()
