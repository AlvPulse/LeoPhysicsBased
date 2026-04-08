# Configuration parameters for Harmonic Detector

# Audio
SAMPLE_RATE = 8096
WINDOW_DURATION = 2.0  # seconds
STEP_SIZE = 0.5        # seconds

# Signal Processing
N_FFT = 2048           # Increased for better frequency resolution (was 2048)
HOP_LENGTH = 1024      # Overlap (adjusted for N_FFT)
MIN_FREQ = 100.0        # Minimum frequency to consider
MAX_FREQ = 2000.0      # Maximum frequency to consider

# Peak Detection
PEAK_PROMINENCE = 3    # dB above local minima
PEAK_DISTANCE = 10      # bins
NUM_PEAKS_TO_KEEP = 10 # Keep more peaks to avoid missing harmonics
SNR_THRESHOLD = 1.5    # Minimum SNR in dB to consider a peak significant
PWR_THRESHOLD = -40    # Minimum PWR in dB to consider a peak significant

# Harmonic Detection
TOLERANCE = 0.1        # Tolerance for harmonic matching (e.g. 0.1 means +/- 10% drift allowed)
MIN_HARMONICS = 2      # Minimum number of harmonics to be considered a valid event
HARMONIC_MIN_SNR = 5.0 # SNR threshold specifically for validating harmonic candidates
HARMONIC_MIN_POWER = -60.0 # Absolute power threshold (dB)
MISSING_HARMONIC_PENALTY = 0.5 # Penalty multiplier for each missing low-order harmonic

# Temporal Persistence
PERSISTENCE_THRESHOLD = 3 # Minimum consecutive frames to consider a harmonic series valid
PERSISTENCE_BUFFER = 5    # Max frames to look back for matching harmonic series (if intermittent)

# Noise Estimation
NOISE_FLOOR_WINDOW = 50 # Window size for median filter (in frequency bins)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20

# False Alarm Rejector Rules (Expert System Thresholds)
# -----------------------------------------------------
# Operator can tune these to adjust the 80% rule-based rejection logic.
REJECTOR_RULES = {
    "BAND_LOW": [0, 300],          # Frequency band (Hz) considered "Low" (e.g., Wind)
    "BAND_MID": [300, 2000],       # Frequency band (Hz) considered "Mid"
    "BAND_HIGH": [2000, 4000],     # Frequency band (Hz) considered "High" (e.g., Airplanes)

    # Ratios representing the proportion of total spectral energy in a specific band
    "WIND_LOW_BAND_RATIO_MIN": 0.8,     # If >70% of energy is in the LOW band, flag as Wind
    "AIRPLANE_HIGH_BAND_RATIO_MIN": 0.8,# If >40% of energy is in the HIGH band, flag as Airplane

    # Spectral Flatness measures how noise-like a signal is (0 = pure tone, 1 = white noise)
    "FLATNESS_MAX_TONAL": 0.35,     # Below 0.2 is considered highly tonal (e.g., harmonic, siren)
    "FLATNESS_MIN_NOISE": 0.7,     # Above 0.6 is considered broadband noise

    # Temporal Features
    "ZCR_MIN_NOISE": 0.1,          # High zero-crossing rate implies unvoiced/noisy signals
    "HARMONIC_SCORE_THRESHOLD":0.8 # Harmonic score threshold
}


RULES_v2 = {
    "BAND_LOW": [0, 300],          # Frequency band (Hz) considered "Low" (e.g., Wind)
    "BAND_MID": [300, 2000],       # Frequency band (Hz) considered "Mid"
    "BAND_HIGH": [2000, 4000],     # Frequency band (Hz) considered "High" (e.g., Airplanes)
    # Root split (from tree)
    "HIGH_BAND_RATIO_ROOT": 0.00,
    
    # Wind/Airplane branch
    "DOMINANT_FREQ_WIND": 221.38,
    
    # Mid branch (high_band_ratio <= 0.47)
    "HIGH_BAND_RATIO_MID": 0.47,
    "LOW_BAND_RATIO_KITCHEN": 0.24,
    "ZCR_MEAN_VEHICLE": 0.04,
    "ZCR_MEAN_VEHICLE_SUB": 0.02,
    "DOMINANT_FREQ_NOISE": 25.70,
    
    # Motor branch (high_band_ratio <= 1.60)
    "HIGH_BAND_RATIO_MOTOR": 1.60,
    "SPECTRAL_KURTOSIS_MOTOR": 95.65,
    "ZCR_MEAN_MOTOR": 0.27,
    
    # Rain branch (high_band_ratio > 1.60)
    "LOW_BAND_RATIO_RAIN": 5.99,
    
    # Class mappings for removed classes
    "KITCHEN_MAPPED_TO": "noise",    # kitchen → noise
    "BIKE_MAPPED_TO": "vehicle",     # bike → vehicle
    "SIREN_MAPPED_TO": "noise"       # siren → noise
}
