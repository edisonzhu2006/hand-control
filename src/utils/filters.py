"""
Signal filters for smoothing noisy tracking data.

Currently provides the One Euro filter (Casiez et al., CHI 2012), the standard
choice for human pose/landmark smoothing: it adapts its cutoff to signal speed,
so it smooths aggressively when the signal is slow (killing jitter) while
staying responsive with minimal lag during fast motion.
"""

import time

import numpy as np


class _LowPass:
    """Simple exponential low-pass filter used internally by OneEuroFilter."""

    def __init__(self):
        self.y = None

    def apply(self, x, alpha):
        if self.y is None:
            self.y = np.asarray(x, dtype=float).copy()
        else:
            self.y = alpha * np.asarray(x, dtype=float) + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """One Euro filter for scalar or vector signals.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother when slow.
        beta: Speed coefficient. Higher = less lag during fast motion.
        d_cutoff: Cutoff frequency (Hz) for the derivative estimate.
    """

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_filter = _LowPass()
        self._dx_filter = _LowPass()
        self._last_time = None
        self._last_x = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, x, timestamp=None):
        """Filter a new sample.

        Args:
            x: Scalar or array sample.
            timestamp: Sample time in seconds (defaults to time.time()).

        Returns:
            Filtered sample with the same shape as x.
        """
        x = np.asarray(x, dtype=float)
        t = time.time() if timestamp is None else float(timestamp)

        if self._last_time is None:
            self._last_time = t
            self._last_x = x.copy()
            self._x_filter.apply(x, 1.0)
            self._dx_filter.apply(np.zeros_like(x), 1.0)
            return x.copy()

        dt = max(t - self._last_time, 1e-6)
        self._last_time = t

        # Estimate (filtered) signal speed.
        dx = (x - self._last_x) / dt
        self._last_x = x.copy()
        dx_hat = self._dx_filter.apply(dx, self._alpha(self.d_cutoff, dt))

        # Speed-adaptive cutoff: still -> min_cutoff (smooth), fast -> higher (responsive).
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        return self._x_filter.apply(x, self._alpha(cutoff, dt)).copy()

    def reset(self):
        """Clear filter state (e.g. after tracking is lost)."""
        self._x_filter = _LowPass()
        self._dx_filter = _LowPass()
        self._last_time = None
        self._last_x = None
