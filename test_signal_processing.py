import unittest
import numpy as np
from src import signal_processing

class TestSignalProcessing(unittest.TestCase):
    def test_parabolic_interpolation(self):
        # Create a signal with peak at index 10.5 (amplitude 1.0)
        # Parabola y = -x^2 + 1
        # Sample at x = -1, 0, 1 relative to peak
        # y(-1) = 0, y(0) = 1, y(1) = 0
        y = np.array([0.0, 1.0, 0.0])
        delta, mag = signal_processing.parabolic_interpolation(y, 1)
        self.assertAlmostEqual(delta, 0.0)
        self.assertAlmostEqual(mag, 1.0)

        # Shifted peak
        # y = -(x - 0.5)^2 + 1
        # y(-1) = -(-1.5)^2 + 1 = -1.25
        # y(0) = -(-0.5)^2 + 1 = 0.75
        # y(1) = -(0.5)^2 + 1 = 0.75
        y = np.array([-1.25, 0.75, 0.75])
        delta, mag = signal_processing.parabolic_interpolation(y, 1)
        # Peak is at x=0.5 relative to center (index 1)
        self.assertAlmostEqual(delta, 0.5)
        self.assertAlmostEqual(mag, 1.0)

if __name__ == '__main__':
    unittest.main()
