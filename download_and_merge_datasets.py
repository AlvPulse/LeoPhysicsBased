import os
import urllib.request
import tarfile
import zipfile
import shutil
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import argparse

from src import config

# Note: AerosonicdB and UrbanSound8K might need manual downloads if direct links fail.
# We will provide Kaggle/Zenodo fallback paths if specified.

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
# UrbanSound8K direct link is often unavailable without filling a form, using a placeholder or local fallback.
URBANSOUND8K_FALLBACK = "UrbanSound8K.tar.gz"
AEROSONIC_FALLBACK = "aerosonicdb.zip"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download successful.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def extract_archive(archive_path, extract_dir):
    print(f"Extracting {archive_path} to {extract_dir}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")

# Class Mappings to target 10-15 classes
CLASS_MAPPING = {
    # Targets
    "UAV": ["drone", "uav", "quadcopter"], # Assuming potential future Aerosonic additions
    "Helicopter": ["helicopter"],
    "Piston_Engine": ["engine", "piston"],
    "Jet": ["jet", "airplane"],
    "Human_Activity": ["crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping", "children_playing", "street_music"],

    # Background / False Alarms
    "Wind_Weather": ["wind", "rain", "thunderstorm", "sea_waves", "crackling_fire", "water_drops"],
    "Highway_Car": ["car_horn", "engine_idling", "highway", "traffic", "car"],
    "Motorcycle": ["motorcycle", "scooter"],
    "Siren_Alarm": ["siren", "car_alarm", "fire_alarm", "bell", "clock_alarm", "smoke_detector"],
    "Animal_Bird": ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow", "chirping_birds", "dog_bark"],
    "Construction_Tool": ["drilling", "jackhammer", "chainsaw", "hand_saw", "glass_breaking"],
    "Music_Urban": ["street_music", "church_bells", "keyboard_typing", "door_wood_creaks", "mouse_click", "keyboard_typing", "door_wood_knock", "can_opening", "washing_machine", "vacuum_cleaner", "clock_tick"],
    "Unknown_Noise": [] # Catch-all
}

def map_class(original_class):
    original_class = original_class.lower().replace(" ", "_")
    for target_class, keywords in CLASS_MAPPING.items():
        if original_class in keywords:
            return target_class
    return "Unknown_Noise"

def process_and_save_audio(src_path, dest_dir, target_class, filename):
    try:
        audio, sr = librosa.load(src_path, sr=config.SAMPLE_RATE, mono=True)
        class_dir = os.path.join(dest_dir, target_class)
        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(class_dir, filename)
        sf.write(dest_path, audio, sr)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def process_esc50(base_dir, output_dir):
    print("Processing ESC-50...")
    esc50_dir = os.path.join(base_dir, "ESC-50-master")
    meta_path = os.path.join(esc50_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "audio")

    if not os.path.exists(meta_path):
        print("ESC-50 meta file not found. Ensure it's extracted correctly.")
        return

    df = pd.read_csv(meta_path)
    for index, row in df.iterrows():
        src_path = os.path.join(audio_dir, row['filename'])
        target_class = map_class(row['category'])
        process_and_save_audio(src_path, output_dir, target_class, f"esc50_{row['filename']}")

def process_urbansound8k(base_dir, output_dir):
    print("Processing UrbanSound8K...")
    us8k_dir = os.path.join(base_dir, "UrbanSound8K")
    meta_path = os.path.join(us8k_dir, "metadata", "UrbanSound8K.csv")

    if not os.path.exists(meta_path):
        print("UrbanSound8K meta file not found.")
        return

    df = pd.read_csv(meta_path)
    for index, row in df.iterrows():
        # Folders are like 'fold1', 'fold2', etc.
        src_path = os.path.join(us8k_dir, "audio", f"fold{row['fold']}", row['slice_file_name'])
        target_class = map_class(row['class'])
        process_and_save_audio(src_path, output_dir, target_class, f"us8k_{row['slice_file_name']}")

def process_aerosonic(base_dir, output_dir):
    print("Processing AerosonicdB...")
    # Assume flat directory or single metadata file for AerosonicdB
    aero_dir = os.path.join(base_dir, "AerosonicdB")
    meta_path = os.path.join(aero_dir, "metadata.csv") # Adjust based on actual AerosonicdB structure

    if not os.path.exists(meta_path):
         print("Aerosonic metadata not found. Skipping or requiring manual structure check.")
         # If no meta, maybe just iterate folders if they represent classes
         for root, dirs, files in os.walk(aero_dir):
             for file in files:
                 if file.endswith('.wav'):
                     class_name = os.path.basename(root)
                     target_class = map_class(class_name)
                     src_path = os.path.join(root, file)
                     process_and_save_audio(src_path, output_dir, target_class, f"aero_{file}")
         return

    try:
        df = pd.read_csv(meta_path)
        for index, row in df.iterrows():
             # Assuming columns 'filename' and 'class' exist, adjust as needed.
             filename = row.get('filename') or row.get('file_name')
             cls_name = row.get('class') or row.get('label')
             if filename and cls_name:
                 src_path = os.path.join(aero_dir, "audio", filename)
                 target_class = map_class(cls_name)
                 process_and_save_audio(src_path, output_dir, target_class, f"aero_{filename}")
    except Exception as e:
        print(f"Failed parsing Aerosonic metadata: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and merge datasets.")
    parser.add_argument("--output_dir", default="data/merged_dataset", help="Output directory for merged data.")
    parser.add_argument("--download_dir", default="data/raw_datasets", help="Directory to store downloaded raw archives.")
    parser.add_argument("--us8k_archive", default=None, help="Local path to UrbanSound8K archive (if already downloaded).")
    parser.add_argument("--aero_archive", default=None, help="Local path to AerosonicdB archive (if already downloaded).")

    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. ESC-50
    esc50_zip = os.path.join(args.download_dir, "esc50.zip")
    if not os.path.exists(esc50_zip):
        download_file(ESC50_URL, esc50_zip)
    extract_archive(esc50_zip, args.download_dir)
    process_esc50(args.download_dir, args.output_dir)

    # 2. UrbanSound8K
    if args.us8k_archive and os.path.exists(args.us8k_archive):
        extract_archive(args.us8k_archive, args.download_dir)
        process_urbansound8k(args.download_dir, args.output_dir)
    else:
        print("Skipping UrbanSound8K (no local archive provided). Please download manually and pass --us8k_archive.")

    # 3. AerosonicdB
    if args.aero_archive and os.path.exists(args.aero_archive):
        extract_archive(args.aero_archive, args.download_dir)
        process_aerosonic(args.download_dir, args.output_dir)
    else:
        print("Skipping AerosonicdB (no local archive provided). Please download manually and pass --aero_archive.")

    print(f"Data processing complete. Structured dataset is at {args.output_dir}")
