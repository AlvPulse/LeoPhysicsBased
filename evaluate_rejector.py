import os
import glob
import argparse
from evaluate_helper import train_dummy_ml
from src.signal_processing import load_audio
from src.false_alarm_rejector import FalseAlarmRejector
from src import config
import numpy as np

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

        ## the function you need
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the False Alarm Rejector module.")
    parser.add_argument("--data_dir", type=str, default="data/no", help="Directory containing audio to test.")
    parser.add_argument("--train_ml", action="store_true", help="Train the Stage 2 Decision Tree with dummy data first.")
    args = parser.parse_args()

    # Initialize the rejector, injecting the tunable configuration
    rejector = FalseAlarmRejector(classifier_version=1)

    print("False Alarm Rejector Initialized with Config:")
    print(f"  Wind (Low Freq > {config.REJECTOR_RULES['WIND_LOW_BAND_RATIO_MIN']*100}%)")
    print(f"  Airplane (High Freq > {config.REJECTOR_RULES['AIRPLANE_HIGH_BAND_RATIO_MIN']*100}%)")
    print(f"  Tonal Flatness < {config.REJECTOR_RULES['FLATNESS_MAX_TONAL']}")

    #if args.train_ml:
    #train_dummy_ml(rejector)

    evaluate_directory(args.data_dir, rejector)
