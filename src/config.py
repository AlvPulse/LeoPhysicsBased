# Configuration parameters for Harmonic Detector

# Audio
SAMPLE_RATE = 44100
WINDOW_DURATION = 2.0  # seconds
STEP_SIZE = 0.5        # seconds

# Signal Processing
N_FFT = 2048           # Restored to legacy 2048
HOP_LENGTH = 512       # 25% of 2048 (75% overlap)
MIN_FREQ = 150.0       # Restored to legacy 150
MAX_FREQ = 2000.0      # Restored to legacy 2000

# Peak Detection
PEAK_PROMINENCE = 4    # Restored to legacy 4
PEAK_DISTANCE = 5      # Adjusted for 44.1kHz (approx 100Hz spacing)
NUM_PEAKS_TO_KEEP = 30 # Keep more peaks to avoid missing harmonics
SNR_THRESHOLD = 1.5    # Minimum SNR in dB to consider a peak significant
PEAK_HEIGHT_REL = 100  # relative height threshold (max - 100)

# Harmonic Detection
TOLERANCE = 0.1        # Tolerance for harmonic matching (e.g. 0.1 means +/- 10% drift allowed)
MIN_HARMONICS = 2      # Minimum number of harmonics to be considered a valid event
HARMONIC_MIN_SNR = 5.0 # SNR threshold specifically for validating harmonic candidates
HARMONIC_MIN_POWER = -60.0 # Absolute power threshold (dB)
MISSING_HARMONIC_PENALTY = 0.5 # Penalty multiplier for each missing low-order harmonic
QUALITY_WEIGHTS = [0.4, 0.4, 0.2] # Default weights for [SNR, Power, Drift]

# Temporal Persistence
PERSISTENCE_THRESHOLD = 3 # Minimum consecutive frames to consider a harmonic series valid
PERSISTENCE_BUFFER = 5    # Max frames to look back for matching harmonic series (if intermittent)

# Noise Estimation
NOISE_FLOOR_WINDOW = 50 # Window size for median filter (in frequency bins)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20
