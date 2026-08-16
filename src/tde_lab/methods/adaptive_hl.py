"""Adaptive Hodges-Lehmann Robust DFT."""
from __future__ import annotations

import numpy as np

from tde_lab.methods.base import BaseMethod
from tde_lab.methods.hodges_lehmann import hodges_lehmann_1d


def _kpe(x: np.ndarray) -> float:
    """
    Percentile-based criterion: KPE = IQR / (P90 - P10).
    Used to switch between median (KPE < threshold) and HL.
    """
    xs = np.sort(x)
    n = len(xs)

    def _idx(p):
        return max(0, min(round(p * (n + 1)) - 1, n - 1))

    q1 = xs[_idx(0.25)]
    q3 = xs[_idx(0.75)]
    p10 = xs[_idx(0.10)]
    p90 = xs[_idx(0.90)]

    denom = p90 - p10
    if denom == 0:
        return 0.0
    return float(0.5 * (q3 - q1) / denom)


class AdaptiveHLRDFT(BaseMethod):
    """
    Adaptive HL RDFT: KPE < threshold → median, else → HL.
    Threshold = 0.21 matches the original MATLAB implementation.
    """

    def __init__(self, threshold: float = 0.21):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Adaptive HL RDFT"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        N = len(signal)
        n = np.arange(N)
        k = np.arange(N)
        twiddle = np.exp(-1j * 2 * np.pi * np.outer(n, k) / N)
        dft_matrix = signal[:, np.newaxis] * twiddle  # (N, N)

        re_mat = np.real(dft_matrix)
        im_mat = np.imag(dft_matrix)
        spectrum = np.zeros(N, dtype=complex)

        for ki in range(N):
            re_col = re_mat[:, ki]
            im_col = im_mat[:, ki]
            if _kpe(re_col) < self.threshold:
                spectrum[ki] = np.median(re_col) + 1j * np.median(im_col)
            else:
                spectrum[ki] = hodges_lehmann_1d(re_col) + 1j * hodges_lehmann_1d(im_col)

        return spectrum
