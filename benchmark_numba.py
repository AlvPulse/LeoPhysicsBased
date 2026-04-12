import time
import random
import sys
import bisect
from unittest.mock import MagicMock

class Config:
    HARMONIC_MIN_SNR = 5
    HARMONIC_MIN_POWER = 10
    TOLERANCE = 0.05
    MAX_FREQ = 2000
    MIN_HARMONICS = 3
    MISSING_HARMONIC_PENALTY = 0.8
    PERSISTENCE_BUFFER = 5
    PERSISTENCE_THRESHOLD = 3

sys.modules['src.config'] = Config
sys.modules['numpy'] = MagicMock()

def track_harmonics_optimized(peaks_per_frame, times):
    # Same as EXTREME from previous benchmark
    pass
