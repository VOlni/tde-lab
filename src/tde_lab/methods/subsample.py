"""Sub-sample delay estimation via parabolic peak interpolation.

Textbook technique: locate the integer cross-correlation peak, then fit a
parabola through the peak and its two neighbours; the vertex gives the
fractional offset.  See e.g. G. Jacovitti, G. Scarano, "Discrete Time
Techniques for Time Delay Estimation", IEEE Trans. Signal Processing, 1993.

Less precise than phase-regression approaches on clean signals (errors of a
few hundredths of a sample instead of ~1e-6), but simple, robust, and free of
third-party code.
"""
from __future__ import annotations

import numpy as np


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a cyclic signal by a (fractional) number of samples using the
    standard FFT phase ramp; real input stays real."""
    x = np.asarray(x)
    n = len(x)
    freqs = np.fft.fftfreq(n)                    # cycles/sample, in [-0.5, 0.5)
    rot = np.exp(-2j * np.pi * freqs * delay_samples)
    out = np.fft.ifft(np.fft.fft(x) * rot)
    return out.real if np.isrealobj(x) else out


def subsample_delay(signal: np.ndarray, ref: np.ndarray) -> float:
    """
    Delay of `signal` relative to `ref` in samples (positive = signal lags
    ref), with sub-sample resolution.  Both signals are treated as cyclic and
    must have the same length.  Sign-inverted signals are handled via the
    magnitude peak.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    n = len(signal)
    if len(ref) != n:
        raise ValueError("signal and ref must have the same length")

    xc = np.fft.ifft(np.fft.fft(signal) * np.conj(np.fft.fft(ref))).real
    k = int(np.argmax(np.abs(xc)))
    sign = 1.0 if xc[k] >= 0 else -1.0           # inverted-signal support

    y1 = sign * xc[(k - 1) % n]
    y2 = sign * xc[k]
    y3 = sign * xc[(k + 1) % n]

    denom = y1 - 2.0 * y2 + y3
    frac = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
    # a true peak keeps the vertex within the centre bin
    frac = float(np.clip(frac, -0.5, 0.5))

    half = n // 2
    return float(((k + frac) + half) % n - half)  # report beyond midpoint as negative
