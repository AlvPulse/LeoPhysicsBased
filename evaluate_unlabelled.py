import os
import json
import glob
import shutil
import librosa
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from src.false_alarm_rejector import FalseAlarmRejector
from src import config

def evaluate_unlabelled_data(unlabelled_dir, config_path, output_dir):
    """
    Evaluates unlabelled audio files, places them into predicted class folders,
    and logs the predictions and config for later comparison.
    """
    print(f"Loading feature selection config from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    with open(config_path, 'r') as f:
        evaluation_config = json.load(f)

    selected_features = evaluation_config.get("selected_features", [])
    if not selected_features:
        print("Warning: No selected features found in config. Defaulting to all features.")

    rejector = FalseAlarmRejector()

    # If the user selected specific features, we update the ML stage feature set
    # (assuming they trained the rejector based ONLY on these features, or we just
    # log them. For pure expert-system evaluation, we can still extract all features
    # but base decisions on the ones we want).
    # For now, let's keep all features for the Expert system, but print which are "important".
    print(f"Using selected features for evaluation context: {selected_features}")

    # Override the rejector to only use the selected features for ML predictions
    if selected_features:
        rejector.feature_names = selected_features

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_session_dir = os.path.join(output_dir, f"eval_session_{timestamp}")
    os.makedirs(eval_session_dir, exist_ok=True)

    # Save a copy of the config used in this session
    shutil.copy2(config_path, os.path.join(eval_session_dir, "used_config.json"))

    eval_results = []

    # Process unlabelled audio
    # Assuming short unlabelled .wav files based on user clarification
    files = glob.glob(os.path.join(unlabelled_dir, "*.wav"))
    if not files:
        print(f"No .wav files found in {unlabelled_dir}")
        return

    print(f"Found {len(files)} files to evaluate.")

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
            result = rejector.process_signal(audio, sr)

            pred_class = result['decision']

            # Record result
            eval_results.append({
                "Filename": file_name,
                "Predicted_Class": pred_class,
                "Confidence": f"{result['confidence']:.2f}",
                "Source": result['source'],
                "Reason": result['reason']
            })

            # Move/Copy to organized folder
            class_dir = os.path.join(eval_session_dir, pred_class.replace("/", "_"))
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(class_dir, file_name))

        except Exception as e:
            print(f"Failed processing {file_name}: {e}")

    # Save CSV
    df = pd.DataFrame(eval_results)
    csv_path = os.path.join(eval_session_dir, "evaluation_report.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Evaluation complete. Session saved to {eval_session_dir}")
    print(f"Detailed report: {csv_path}")
    print(f"Listen to the predictions inside the subfolders of {eval_session_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate unlabelled datasets and organize by predicted class.")
    parser.add_argument("--unlabelled_dir", type=str, required=True, help="Directory containing unlabelled .wav files.")
    parser.add_argument("--config", type=str, default="selected_features_config.json", help="Path to selected features config.")
    parser.add_argument("--output_dir", type=str, default="evaluations", help="Directory to save evaluation sessions.")

    args = parser.parse_args()

    evaluate_unlabelled_data(args.unlabelled_dir, args.config, args.output_dir)
