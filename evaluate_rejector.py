import os
import glob
import numpy as np
import scipy.signal as signal
import argparse
from src.false_alarm_rejector import FalseAlarmRejector
from src.signal_processing import load_audio
from src import config

def evaluate_directory(directory, rejector):
    """
    Evaluates all .wav files in a directory using the rejector.
    """
    files = glob.glob(os.path.join(directory, "*.wav"))

    if not files:
        print(f"No .wav files found in {directory}")
        return

    print(f"\nEvaluating False Alarm Rejector on: {directory}")
    print("-" * 80)
    print(f"{'Filename':<20} | {'Decision':<15} | {'Conf':<4} | {'Source':<12} | {'Reason'}")
    print("-" * 80)

    decisions = []

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        audio, fs = load_audio(filepath, config.SAMPLE_RATE)

        if audio is None:
            continue

        result = rejector.process_signal(audio, fs)
        decisions.append(result['decision'])

        # Format the output for readability
        conf_str = f"{result['confidence']:.2f}"

        # Truncate reason if too long
        reason = result['reason']
        if len(reason) > 40:
            reason = reason[:37] + "..."

        print(f"{filename:<20} | {result['decision']:<15} | {conf_str:<4} | {result['source']:<12} | {reason}")

    # Print summary statistics
    print("-" * 80)
    print("Summary:")
    unique, counts = np.unique(decisions, return_counts=True)
    total = len(decisions)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} ({(c/total)*100:.1f}%)")

def train_dummy_ml(rejector):
    """
    Trains the Stage 2 ML Decision Tree with some synthetic/dummy noise data
    to demonstrate how the '20% unknown' fallback works when rules fail.
    In a real scenario, this would use a real labeled dataset of false alarms.
    """
    print("\nTraining Stage 2 Decision Tree with synthetic fallback data...")
    fs = config.SAMPLE_RATE
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    X_train = []
    y_train = []

    # 1. "Complex Noise" (Mix of mid-freq noise and impulses) -> Needs ML to distinguish
    for _ in range(50):
        noise = np.random.randn(len(t)) * 0.1
        # Add some mid-frequency components that might confuse the rules
        mid_noise = signal.butter(4, [500, 1500], btype='bandpass', fs=fs, output='sos')
        complex_noise = signal.sosfilt(mid_noise, noise)

        # Add impulsive clicks to break the flatness rule
        clicks = np.zeros_like(t)
        click_idx = np.random.randint(0, len(t), size=10)
        clicks[click_idx] = 1.0

        sig = complex_noise + clicks
        X_train.append(sig)
        y_train.append("Complex Noise")

    # 2. "Bird Call" (Frequency Modulated chirp) -> Tonal but not constant harmonic
    for _ in range(50):
        # 1000Hz to 2000Hz chirp
        sig = signal.chirp(t, f0=1000, f1=2000, t1=duration, method='linear') * 0.5
        # Add some background noise
        sig += np.random.randn(len(t)) * 0.05
        X_train.append(sig)
        y_train.append("Bird Call")

    # 3. "Vehicle Engine" (Low frequency pulsing)
    for _ in range(50):
        base_hz = 60
        pulse_hz = 5
        sig = np.sin(2 * np.pi * base_hz * t) * (0.5 + 0.5 * np.sin(2 * np.pi * pulse_hz * t))
        sig += np.random.randn(len(t)) * 0.1
        X_train.append(sig)
        y_train.append("Vehicle Engine")

    # Train the rejector
    fs_list = [fs] * len(X_train)
    rejector.train_ml_stage(X_train, fs_list, y_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the False Alarm Rejector module.")
    parser.add_argument("--data_dir", type=str, default="data/yes", help="Directory containing audio to test.")
    parser.add_argument("--train_ml", action="store_true", help="Train the Stage 2 Decision Tree with dummy data first.")
    args = parser.parse_args()

    # Initialize the rejector, injecting the tunable configuration
    rejector = FalseAlarmRejector(rules_config=config.REJECTOR_RULES)

    print("False Alarm Rejector Initialized with Config:")
    print(f"  Wind (Low Freq > {config.REJECTOR_RULES['WIND_LOW_BAND_RATIO_MIN']*100}%)")
    print(f"  Airplane (High Freq > {config.REJECTOR_RULES['AIRPLANE_HIGH_BAND_RATIO_MIN']*100}%)")
    print(f"  Tonal Flatness < {config.REJECTOR_RULES['FLATNESS_MAX_TONAL']}")

    if args.train_ml:
        train_dummy_ml(rejector)

    evaluate_directory(args.data_dir, rejector)
