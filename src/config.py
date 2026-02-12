# Configuration parameters for Harmonic Detector

# Audio
SAMPLE_RATE = 44100
WINDOW_DURATION = 2.0  # seconds
STEP_SIZE = 0.5        # seconds

# Signal Processing
N_FFT = 2048
HOP_LENGTH = 512       # Overlap
MIN_FREQ = 100.0       # Minimum frequency to consider
MAX_FREQ = 2000.0      # Maximum frequency to consider

# Peak Detection
# The user's original code used:
# PEAK_HEIGHT = max_h - 100
# PROMINENCE = 4
# DISTANCE = 40
# We need to map these to our logic or use similar values.

# Since we normalize our synthetic data to -1..1, max power might be around 0 to -10 dB depending on windowing.
# If max_h is 0, height > -100 is very permissive.
# If max_h is -50 (quiet), height > -150 is permissive.
# So absolute threshold might be better, or relative to noise floor.

PEAK_PROMINENCE = 5    # dB above local minima
PEAK_DISTANCE = 20     # bins
NUM_PEAKS_TO_KEEP = 20 # Keep more peaks to avoid missing harmonics
SNR_THRESHOLD = 3.0    # Minimum SNR in dB to consider a peak significant (lowered from 5)

# Harmonic Detection
TOLERANCE = 0.05       # Tolerance for harmonic matching (5% deviation)
MIN_HARMONICS = 3      # Minimum number of harmonics to be considered a valid event

# Noise Estimation
NOISE_FLOOR_WINDOW = 50 # Window size for median filter (in frequency bins)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20
