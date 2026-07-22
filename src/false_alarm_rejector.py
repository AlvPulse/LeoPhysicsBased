import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis
from sklearn.tree import DecisionTreeClassifier, export_text
import json
import os

from src import config
from src.harmonic_detection import detect_harmonics_iterative, track_harmonics
from src.signal_processing import compute_spectrogram_and_peaks

class FalseAlarmRejector:
    """
    A Hybrid Expert-System + Machine Learning false alarm rejection module.

    Stage 1: Rule-Based Expert System filters out obvious noise (e.g., Wind, Airplane)
             using physics-based thresholds (e.g., Band Energy Ratios, Spectral Flatness).
    Stage 2: Interpretable Machine Learning (Decision Tree) classifies remaining
             ambiguous signals and provides pseudo-probabilities.
    """
    def __init__(self, rules_config=None):
        # Load or set default rule thresholds
        if rules_config is None:
            self.rules = {
                "BAND_LOW": [0, 300],        # Hz
                "BAND_MID": [300, 2000],     # Hz
                "BAND_HIGH": [2000, 4000],   # Hz

                # Rule Thresholds
                "WIND_LOW_BAND_RATIO_MIN": 0.70, # If >70% energy is low freq -> Wind
                "AIRPLANE_HIGH_BAND_RATIO_MIN": 0.40, # If >40% energy is high freq -> Airplane
                "FLATNESS_MAX_TONAL": 0.2,   # Below this is considered tonal (e.g., harmonic, siren)
                "FLATNESS_MIN_NOISE": 0.6,   # Above this is broadband noise (wind, road noise)
                "ZCR_MIN_NOISE": 0.1,        # High zero crossing rate indicates noisy/unvoiced
            }
        else:
            self.rules = rules_config

        # The Decision Tree for Stage 2
        # max_depth=4 keeps it interpretable
        self.ml_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.is_trained = False

        # Define feature names for interpretation
        self.feature_names = [
            'low_band_ratio', 'mid_band_ratio', 'high_band_ratio',
            'spectral_flatness', 'spectral_entropy', 'spectral_kurtosis',
            'spectral_centroid', 'dominant_freq',
            'temporal_entropy', 'rms_energy', 'zcr_mean', 'zcr_std',
            'harmonic_score'
        ]

        self.classes = ['Unknown', 'Wind', 'Airplane', 'Harmonic/Tonal', 'Broadband Noise']

    def extract_features(self, audio, fs):
        """
        Extracts physics-based and statistical features from the audio signal.
        """
        # If signal is extremely short (e.g., < N_FFT), zero-pad it to
        # avoid degenerate errors in Welch PSD or other spectral functions
        if len(audio) < config.N_FFT:
            pad_width = config.N_FFT - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')

        features = {}

        # 1. Temporal Features
        # RMS Energy
        features['rms_energy'] = np.sqrt(np.mean(audio**2) + 1e-10)

        # Zero-Crossing Rate (ZCR)
        crossings = np.where(np.diff(np.signbit(audio).astype(int)))[0]
        # Approximate ZCR per frame (using ~20ms frames)
        frame_size = int(fs * 0.02)
        if len(audio) > frame_size:
            num_frames = len(audio) // frame_size
            audio_frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
            zcr_frames = np.sum(np.abs(np.diff(np.signbit(audio_frames).astype(int), axis=1)), axis=1) / frame_size
            features['zcr_mean'] = np.mean(zcr_frames)
            features['zcr_std'] = np.std(zcr_frames)
        else:
            zcr = len(crossings) / len(audio)
            features['zcr_mean'] = zcr
            features['zcr_std'] = 0.0

        # Temporal Entropy (Energy distribution over time)
        if len(audio) > frame_size:
            energy_frames = np.sum(audio_frames**2, axis=1)
            p_energy = energy_frames / (np.sum(energy_frames) + 1e-10)
            p_energy = p_energy[p_energy > 0]
            features['temporal_entropy'] = -np.sum(p_energy * np.log2(p_energy))
        else:
            features['temporal_entropy'] = 0.0

        # Onset Strength
        import librosa
        onset_env = librosa.onset.onset_strength(y=audio, sr=fs)
        features['onset_strength'] = np.mean(onset_env) if len(onset_env) > 0 else 0.0

        # 2. Spectral Features
        # Compute Power Spectral Density
        f, Pxx = signal.welch(audio, fs, nperseg=config.N_FFT)
        Pxx_norm = Pxx / (np.sum(Pxx) + 1e-10) # Normalize for probability distribution

        # Band Energy Ratios
        low_mask = (f >= self.rules["BAND_LOW"][0]) & (f < self.rules["BAND_LOW"][1])
        mid_mask = (f >= self.rules["BAND_MID"][0]) & (f < self.rules["BAND_MID"][1])
        high_mask = (f >= self.rules["BAND_HIGH"][0]) & (f < self.rules["BAND_HIGH"][1])

        total_energy = np.sum(Pxx) + 1e-10
        features['low_band_ratio'] = np.sum(Pxx[low_mask]) / total_energy
        features['mid_band_ratio'] = np.sum(Pxx[mid_mask]) / total_energy
        features['high_band_ratio'] = np.sum(Pxx[high_mask]) / total_energy

        # Spectral Flatness (Geometric Mean / Arithmetic Mean of Power Spectrum)
        # Add small epsilon to avoid log(0)
        Pxx_safe = Pxx + 1e-10
        geo_mean = np.exp(np.mean(np.log(Pxx_safe)))
        arith_mean = np.mean(Pxx_safe)
        features['spectral_flatness'] = geo_mean / arith_mean

        # Spectral Entropy
        p_spec = Pxx_norm[Pxx_norm > 0]
        features['spectral_entropy'] = -np.sum(p_spec * np.log2(p_spec))

        # Spectral Kurtosis (Peakiness of the spectrum)
        features['spectral_kurtosis'] = float(kurtosis(Pxx))

        # Spectral Centroid (Center of mass of spectrum)
        features['spectral_centroid'] = np.sum(f * Pxx_norm)

        # Band Index (which band has max energy)
        band_energies = [features['low_band_ratio'], features['mid_band_ratio'], features['high_band_ratio']]
        features['band_index'] = np.argmax(band_energies)

        # Dominant Frequency
        features['dominant_freq'] = f[np.argmax(Pxx)]

        # 3. Harmonic Features
        # Extract harmonic information using existing project logic
        f_stft, t_stft, Pxx_db, peaks_per_frame = compute_spectrogram_and_peaks(audio, fs)

        # track_harmonics handles the iterative detection and tracking internally
        # based on peaks_per_frame
        try:
            # Depending on the local codebase version, track_harmonics might take 1 or 2 arguments.
            # The traceback indicates it takes 1 argument locally.
            active_series = track_harmonics(peaks_per_frame)
        except TypeError:
            # Fallback if the environment has a version requiring both arguments
            active_series = track_harmonics(peaks_per_frame, t_stft)

        # Feature: Harmonic Score
        # We define harmonic score as the ratio of frames that contain
        # at least one active harmonic series over total frames.
        if len(t_stft) > 0 and len(active_series) > 0:
            # Calculate sum of persistences (durations) of all tracked series relative to total frames
            total_harmonic_frames = sum(series['persistence'] for series in active_series)
            # Normalize by total frames (could exceed 1 if overlapping series, so we clip)
            features['harmonic_score'] = min(1.0, total_harmonic_frames / len(t_stft))
        else:
            features['harmonic_score'] = 0.0

        return features

    def _features_to_array(self, features):
        """Converts feature dictionary to an ordered numpy array."""
        return np.array([features[name] for name in self.feature_names])

    def evaluate_expert_rules(self, features):
        """
        Stage 1: Apply physics-based rules.
        Returns: (Decision_String, Confidence, Triggered_Rule_Name)
                 or None if no rule matches.
        """
        r = self.rules

        # Rule 1: Wind Noise (Very high low-frequency energy, high flatness/noise)
        if features['low_band_ratio'] > r['WIND_LOW_BAND_RATIO_MIN'] and features['spectral_flatness'] > r['FLATNESS_MAX_TONAL']:
            return "Wind", 0.90, "High Low-Band Energy + High Flatness"

        # Rule 2: Airplane Noise (Significant high-frequency energy, broadband)
        if features['high_band_ratio'] > r['AIRPLANE_HIGH_BAND_RATIO_MIN'] and features['spectral_flatness'] > r['FLATNESS_MAX_TONAL']:
            return "Airplane", 0.85, "High High-Band Energy + High Flatness"

        # Rule 3: Clear Tonal/Harmonic Signal (Low flatness, clear dominant frequency)
        if features['spectral_flatness'] < r['FLATNESS_MAX_TONAL']:
            # Could be siren, bird, harmonic call
            return "Harmonic/Tonal", 0.80, "Low Spectral Flatness (Highly Tonal)"

        # Rule 4: Generic Broadband Noise
        if features['spectral_flatness'] > r['FLATNESS_MIN_NOISE'] and features['zcr_mean'] > r['ZCR_MIN_NOISE']:
            return "Broadband Noise", 0.75, "High Flatness + High ZCR"

        return None # Falls through to ML Stage

    def process_signal(self, audio, fs):
        """
        End-to-end processing of a signal.
        Returns: {
            'decision': str,
            'confidence': float,
            'source': 'Expert Rule' or 'ML Model',
            'reason': str,
            'features': dict
        }
        """
        features = self.extract_features(audio, fs)

        # Stage 1: Expert System
        rule_result = self.evaluate_expert_rules(features)

        if rule_result is not None:
            decision, confidence, reason = rule_result
            return {
                'decision': decision,
                'confidence': confidence,
                'source': 'Expert Rule',
                'reason': f"Rule Triggered: {reason}",
                'features': features
            }

        # Stage 2: Machine Learning
        if self.is_trained:
            X = self._features_to_array(features).reshape(1, -1)
            # Predict probabilities
            probs = self.ml_classifier.predict_proba(X)[0]
            pred_idx = np.argmax(probs)

            # Use the max probability as our pseudo-probability/confidence
            confidence = probs[pred_idx]
            decision = self.ml_classifier.classes_[pred_idx]

            # Find the decision path for interpretability
            node_indicator = self.ml_classifier.decision_path(X)
            leaf_id = self.ml_classifier.apply(X)[0]

            # Reconstruct a simple reason from the path
            path = []
            feature = self.ml_classifier.tree_.feature
            threshold = self.ml_classifier.tree_.threshold
            for node_id in node_indicator.indices:
                if leaf_id == node_id:
                    continue
                if X[0, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
                path.append(f"{self.feature_names[feature[node_id]]} {threshold_sign} {threshold[node_id]:.2f}")

            reason = " -> ".join(path)

            return {
                'decision': decision,
                'confidence': float(confidence),
                'source': 'ML Model (Decision Tree)',
                'reason': f"Path: {reason}",
                'features': features
            }
        else:
            return {
                'decision': "Unknown",
                'confidence': 0.0,
                'source': 'Fallback',
                'reason': "No rules triggered and ML model untrained",
                'features': features
            }

    def train_ml_stage(self, X_train_audio, fs_list, y_train):
        """
        Trains the Decision Tree on signals that slip past the expert rules.
        """
        X_features = []
        y_filtered = []

        print("Training Rejector ML Stage...")
        for i, (audio, fs, label) in enumerate(zip(X_train_audio, fs_list, y_train)):
            feats = self.extract_features(audio, fs)
            X_features.append(self._features_to_array(feats))
            y_filtered.append(label)

        if len(X_features) == 0:
            print("Warning: No data provided to train ML model.")
            return

        X = np.array(X_features)
        y = np.array(y_filtered)

        self.ml_classifier.fit(X, y)
        self.is_trained = True

        print(f"ML Model trained on {len(X_features)} samples.")
        print("Decision Tree Rules:")
        print(export_text(self.ml_classifier, feature_names=self.feature_names))

    def save_rules(self, filepath):
        """Save expert rules to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.rules, f, indent=4)

    def load_rules(self, filepath):
        """Load expert rules from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.rules = json.load(f)
