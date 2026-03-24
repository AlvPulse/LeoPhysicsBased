import os
import glob
import shutil
import numpy as np
import scipy.signal as signal
import argparse
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.false_alarm_rejector import FalseAlarmRejector
from src.signal_processing import load_audio
from src import config

def load_dataset(data_dir):
    """Loads all yes/no audio files into memory."""
    data = []

    for label, folder in [("Yes", "yes"), ("No", "no")]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} not found.")
            continue

        files = glob.glob(os.path.join(folder_path, "*.wav"))
        for filepath in sorted(files):
            audio, fs = load_audio(filepath, config.SAMPLE_RATE)
            if audio is not None:
                data.append({
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "audio": audio,
                    "fs": fs,
                    "true_label": label
                })
    return data

def build_train_eval_pipeline(rejector, data, test_size=0.3):
    """
    Splits the data, trains the ML stage on the 'No' (false alarm) train subset,
    evaluates on the test set, and organizes files for manual cross-check.
    """
    if not data:
        print("No data available to process.")
        return

    print(f"\n--- Data Split (Test Size: {test_size * 100}%) ---")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, stratify=[d['true_label'] for d in data])
    print(f"Train Set: {len(train_data)} samples")
    print(f"Test Set:  {len(test_data)} samples")

    # 1. Train the ML Model
    print("\n--- Training Stage 2 ML Model ---")

    # We need to train the Decision Tree to distinguish between actual True Positives (Yes)
    # and the ambiguous False Alarms (No) that slip past the expert rules.
    # Since the user's "No" files are unlabeled but they want them split into groups,
    # we will use an unsupervised clustering step (K-Means) to discover distinct types of noise
    # among the un-discarded "No" files, and then train the interpretable Decision Tree on
    # [Yes] vs [Noise Group 0] vs [Noise Group 1].

    from sklearn.cluster import KMeans

    unknown_no_features = []
    unknown_no_indices = []

    # First Pass: Extract features for all training data
    all_train_features = []
    for i, item in enumerate(train_data):
        feats = rejector.extract_features(item['audio'], item['fs'])
        all_train_features.append(feats)

        # Identify "No" files that pass the expert rules (un-discarded)
        if item['true_label'] == "No" and rejector.evaluate_expert_rules(feats) is None:
            unknown_no_features.append(rejector._features_to_array(feats))
            unknown_no_indices.append(i)

    # Second Pass: Cluster the unknown "No" files into 2 distinct groups
    noise_clusters = None
    if len(unknown_no_features) > 1:
        print(f"Clustering {len(unknown_no_features)} ambiguous false alarms into 2 groups for debug cross-checking...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        noise_clusters = kmeans.fit_predict(unknown_no_features)

    X_train_audio = []
    fs_list = []
    y_train_pseudo = []

    for i, item in enumerate(train_data):
        if item['true_label'] == "Yes":
            # Include True Positives so the tree learns the binary distinction
            X_train_audio.append(item['audio'])
            fs_list.append(item['fs'])
            y_train_pseudo.append("Harmonic/Tonal")
        elif item['true_label'] == "No":
            # Check if this "No" file was ambiguous (and thus clustered)
            if i in unknown_no_indices:
                cluster_id = noise_clusters[unknown_no_indices.index(i)] if noise_clusters is not None else 0
                pseudo_class = f"Unknown Noise Type {cluster_id}"
                X_train_audio.append(item['audio'])
                fs_list.append(item['fs'])
                y_train_pseudo.append(pseudo_class)
            # (Note: "No" files that were already discarded by expert rules are intentionally
            # excluded from Decision Tree training, per user request to train on un-discarded ones)

    if len(set(y_train_pseudo)) > 1:
        rejector.train_ml_stage(X_train_audio, fs_list, y_train_pseudo)
    else:
        print("Warning: Not enough class diversity in the un-discarded training set to train the ML rejector.")

    # 2. Evaluate and Generate Metrics
    print("\n--- Evaluating Test Set ---")
    y_true_binary = []
    y_pred_binary = []

    # Store predictions for the detailed CSV
    eval_results = []

    # Setup directory for manual cross-check
    eval_dir = "eval_groups"
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir)

    for item in test_data:
        result = rejector.process_signal(item['audio'], item['fs'])
        pred_class = result['decision']

        # Binary translation: If the model says it's anything but "Harmonic/Tonal", it's a "No" (False Alarm)
        # Note: Depending on rules, you might want to adjust what passes the filter.
        is_harmonic = (pred_class == "Harmonic/Tonal")
        binary_pred = "Yes" if is_harmonic else "No"

        y_true_binary.append(item['true_label'])
        y_pred_binary.append(binary_pred)

        eval_results.append({
            "Filename": item['filename'],
            "True_Label": item['true_label'],
            "Predicted_Class": pred_class,
            "Binary_Prediction": binary_pred,
            "Confidence": f"{result['confidence']:.2f}",
            "Source": result['source'],
            "Reason": result['reason']
        })

        # 3. Manual Cross-Check Organization
        # Only organize the 'No' files (the false alarms) into their predicted folders
        if item['true_label'] == "No":
            group_dir = os.path.join(eval_dir, pred_class.replace("/", "_"))
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)
            shutil.copy2(item['filepath'], os.path.join(group_dir, item['filename']))

    # Print Binary Metrics
    print("\n=== Binary Classification Metrics (Yes vs No) ===")
    print(classification_report(y_true_binary, y_pred_binary, target_names=["No (False Alarm)", "Yes (True Harmonic)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_binary, y_pred_binary))

    # Print Multi-class Distribution
    print("\n=== Multi-Class Distribution on Test Set ('No' files only) ===")
    no_predictions = [res['Predicted_Class'] for res in eval_results if res['True_Label'] == 'No']
    unique_classes, class_counts = np.unique(no_predictions, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} files")

    # Write CSV for Operator
    csv_path = "eval_results.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["Filename", "True_Label", "Predicted_Class", "Binary_Prediction", "Confidence", "Source", "Reason"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_results)

    print(f"\n✅ Evaluation complete. Organized files placed in '{eval_dir}/'. Detailed report saved to '{csv_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate the False Alarm Rejector module.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory containing 'yes' and 'no' subfolders.")
    args = parser.parse_args()

    # Initialize the rejector
    rejector = FalseAlarmRejector(rules_config=config.REJECTOR_RULES)

    print("Initializing False Alarm Rejector Pipeline...")
    data = load_dataset(args.data_dir)
    build_train_eval_pipeline(rejector, data)
