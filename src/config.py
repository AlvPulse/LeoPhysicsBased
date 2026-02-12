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
PEAK_HEIGHT_REL = 0.1  # Relative height for peak detection (0-1 of max)
PEAK_PROMINENCE = 5    # Minimum prominence in dB
PEAK_DISTANCE = 20     # Minimum distance between peaks in bins
NUM_PEAKS_TO_KEEP = 15 # Max number of peaks to analyze

# Harmonic Detection
TOLERANCE = 0.05       # Tolerance for harmonic matching (5% deviation)
MIN_HARMONICS = 3      # Minimum number of harmonics to be considered a valid event

# Noise Estimation
NOISE_FLOOR_WINDOW = 50 # Window size for median filter (in frequency bins)

# Training
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20
