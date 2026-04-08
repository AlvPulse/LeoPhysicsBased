import numpy as np
from src import config

def detect_baseline_heuristic(peaks):
    """
    Implements the original heuristic logic:
    1. Filter peaks > 150Hz (MIN_FREQ) - done by find_significant_peaks
    2. Select Top N loudest peaks - done by find_significant_peaks
    3. Re-sort by frequency - done by find_significant_peaks
    4. Assume 1st peak is Base.
    5. Check ratio of others to Base.
    """
    if not peaks:
        return 0.0, 0.0

    # Original logic sorted by Loudness first, picked Top N, then sorted by Freq.
    # Our `find_significant_peaks` already does this!
    # It returns peaks sorted by Freq.

    # So, 1st peak is the lowest frequency among the loudest peaks.
    base_freq = peaks[0]['freq']

    matches = 0
    # In original code, it checked NUM_PEAKS_TO_CHECK (e.g. 7)
    # The first one is BASE (matches=0 initially or 1? Original code: matches=0, then loop)
    # Original loop:
    # if i==0: status="BASE"
    # else: ratio = f/base; check tolerance; if match: matches+=1

    # So max score is if all subsequent peaks are harmonics.
    # Score = (matches / (NUM_PEAKS_TO_CHECK - 1)) * 100?
    # Original: current_score = (matches / 4) * 100
    # Wait, original denominator was hardcoded 4?
    # "Max Score possible: 100% (if 7 harmonics found)" -> but denominator was 4.
    # Let's replicate the logic: count harmonic matches among the top peaks.

    # We will iterate through the provided peaks (which are Top N sorted by freq)
    for i in range(1, len(peaks)):
        f = peaks[i]['freq']
        ratio = f / base_freq
        coeff = round(ratio)
        drift = abs(ratio - coeff)

        if coeff > 1 and drift < config.TOLERANCE:
            matches += 1

    # Normalize score.
    # If we found at least 2 harmonics (matches >= 2), that's good.
    # Let's say 4 matches is 100%.
    score = min((matches / 4.0), 1.0)

    return score, base_freq
