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
NUM_PEAKS_TO_KEEP = 20 # Keep more peaks to avoid missing harmonics
SNR_THRESHOLD = 1.5    # Minimum SNR in dB to consider a peak significant

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
