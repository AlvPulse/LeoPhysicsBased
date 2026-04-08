import os
import glob
import numpy as np
import scipy.signal as signal



from src.signal_processing import load_audio, chunk_audio
from src import config
from sklearn.tree import export_text

def find_label(file_name):
    if ("const" in file_name or "gust" in file_name ):
        return "wind"
    elif("noise" in file_name):
        return "noise"
    elif("silence" in file_name):
        return "silence"
    elif("running_tap" in file_name):
        return "rain"
    elif("bike" in file_name):
        return "bike"
    elif("dishes" in file_name or "kitchen" in file_name):
        return "kitchen"
    elif("Airplane" in file_name):
        return "Airplane"
    elif("walk" in file_name):
        return "walking"
    elif("vehicle" in file_name):
        return "vehicle"
    elif("Train" in file_name):
        return "Train"
    elif("wind" in file_name):
        return "wind"
    elif("Siren" in file_name):
        return "Siren"
    elif("silence" in file_name):
        return "silence"
    elif("Music" in file_name):
        return "Music"
    elif("Motor" in file_name):
        return "Motor"
    elif("Human" in file_name):
        return "speech"
    elif("animal" in file_name):
        return "animal"
    else:
        return "unknown"
    


def add_ml_files(X_train,y_train, fs_list,directory="data/train"):
    files = glob.glob(os.path.join(directory, "*.wav"))

    if not files:
        print(f"No .wav files found in {directory}")
        return

    print(f"Training False Alarm Rejector on: {directory}")
    

    # X_train = []
    # y_train = []
    # fs_list = []

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        audio, fs = load_audio(filepath, config.SAMPLE_RATE)

        if audio is None:
            continue
        chunks = chunk_audio(audio, fs, chunk_duration_sec=2)

        for ch in chunks:
            X_train.append(ch)
            y_train.append(find_label(filename))
            fs_list.append(fs)
    
    return X_train,y_train, fs_list
    #rejector.train_ml_stage(X_train, fs_list, y_train)
    



def train_dummy_ml(rejector):
    """
    Trains the Stage 2 ML Decision Tree with some synthetic/dummy noise data
    to demonstrate how the '20% unknown' fallback works when rules fail.
    In a real scenario, this would use a real labeled dataset of false alarms.
    """
    print("\nTraining Stage 2 Decision Tree with internet fallback data...")
    fs = config.SAMPLE_RATE
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    X_train = []
    y_train = []

    # # # 1. "Complex Noise" (Mix of mid-freq noise and impulses) -> Needs ML to distinguish
    # # for _ in range(50):
    # #     noise = np.random.randn(len(t)) * 0.1
    # #     # Add some mid-frequency components that might confuse the rules
    # #     mid_noise = signal.butter(4, [500, 1500], btype='bandpass', fs=fs, output='sos')
    # #     complex_noise = signal.sosfilt(mid_noise, noise)

    # #     # Add impulsive clicks to break the flatness rule
    # #     clicks = np.zeros_like(t)
    # #     click_idx = np.random.randint(0, len(t), size=10)
    # #     clicks[click_idx] = 1.0

    # #     sig = complex_noise + clicks
    # #     X_train.append(sig)
    # #     y_train.append("Complex Noise")

    # # # 2. "Bird Call" (Frequency Modulated chirp) -> Tonal but not constant harmonic
    # # for _ in range(50):
    # #     # 1000Hz to 2000Hz chirp
    # #     sig = signal.chirp(t, f0=1000, f1=2000, t1=duration, method='linear') * 0.5
    # #     # Add some background noise
    # #     sig += np.random.randn(len(t)) * 0.05
    # #     X_train.append(sig)
    # #     y_train.append("Bird Call")

    # # # 3. "Vehicle Engine" (Low frequency pulsing)
    # # for _ in range(50):
    # #     base_hz = 60
    # #     pulse_hz = 5
    # #     sig = np.sin(2 * np.pi * base_hz * t) * (0.5 + 0.5 * np.sin(2 * np.pi * pulse_hz * t))
    # #     sig += np.random.randn(len(t)) * 0.1
    # #     X_train.append(sig)
    # #     y_train.append("Vehicle Engine")

    # # Train the rejector
    # fs_list = [fs] * len(X_train)
    fs_list=[]
    X_train,y_train, fs_list= add_ml_files(X_train,y_train, fs_list)
    rejector.train_ml_stage(X_train, fs_list, y_train)
    
    # Extract rules (with class names for clarity)
    class_names = ['Airplane', 'wind', 'noise']  # REPLACE WITH YOUR CLASSES
    rules = export_text(
        rejector.ml_classifier,
        feature_names=rejector.feature_names,  # REPLACE WITH YOUR FEATURES
        show_weights=True,
        #class_names=class_names
    )

    print(rules)
