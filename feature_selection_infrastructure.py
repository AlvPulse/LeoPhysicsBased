import os
import glob
import json
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from src.false_alarm_rejector import FalseAlarmRejector
from src import config

def load_data_and_extract_features(data_dir, rejector):
    """
    Scans the structured dataset directory, loads audio, and extracts features.
    """
    features_list = []
    labels = []
    file_paths = []

    print(f"Scanning directory: {data_dir} for feature extraction...")

    # Iterate through all class subdirectories
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return [], [], []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load audio using librosa directly for feature extraction script
                    audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)

                    # Extract features using our Rejector's feature extraction logic
                    feats = rejector.extract_features(audio, sr)

                    # Store data
                    features_list.append(rejector._features_to_array(feats))
                    labels.append(class_name)
                    file_paths.append(file_path)
                except Exception as e:
                    print(f"Failed processing {file_path}: {e}")

    return np.array(features_list), np.array(labels), file_paths

def select_top_features(features_array, labels, feature_names, top_n=5):
    """
    Trains a Random Forest classifier to determine feature importance.
    """
    print(f"\nTraining Random Forest to determine feature importance over {len(set(labels))} classes...")
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the RF model
    y_pred = rf.predict(X_test)
    print("\nInitial RF Model Performance (All Features):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature Ranking:")
    top_feature_names = []
    for f in range(min(top_n, len(feature_names))):
        print(f"{f + 1}. feature {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        top_feature_names.append(feature_names[indices[f]])

    return top_feature_names

def save_config(selected_features, output_path="selected_features_config.json"):
    config_data = {
        "selected_features": selected_features,
        "description": "Config with features selected for highest importance in distinguishing Targets vs False Alarms."
    }
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"\nSelected features configuration saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features, determine importance, and select top features.")
    parser.add_argument("--data_dir", type=str, default="data/merged_dataset", help="Directory containing the structured dataset folders.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top features to select.")
    args = parser.parse_args()

    rejector = FalseAlarmRejector()

    features_array, labels, file_paths = load_data_and_extract_features(args.data_dir, rejector)

    if len(features_array) == 0:
        print("No valid data found to process.")
    else:
        top_features = select_top_features(features_array, labels, rejector.feature_names, top_n=args.top_n)
        save_config(top_features)
