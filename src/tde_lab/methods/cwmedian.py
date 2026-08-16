"""Complex Weighted Median Robust DFT."""
from __future__ import annotations

import numpy as np

from tde_lab.methods.base import BaseMethod


def cwmedian_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    1-D weighted median filter where the centre sample is duplicated twice
    to give it extra weight — matches the MATLAB cwmedian.m logic.
    Window is forced to odd size.
    """
    if window % 2 == 0:
        window += 1
    hw = (window - 1) // 2
    n = len(x)
    out = x.copy()

    for i in range(1, n - 1):
        k1 = max(0, i - hw)
        k2 = min(n - 1, i + hw)
        seg = list(x[k1: k2 + 1])
        # duplicate centre element twice (extra weight)
        centre_local = i - k1
        seg.append(seg[centre_local])
        seg.append(seg[centre_local])
        out[i] = float(np.median(seg))

    return out


class CWMedianRDFT(BaseMethod):
    """
    Robust DFT: applies a 1-D CW-Median filter to the real and imaginary
    parts of the DFT products across the sample dimension, then takes the
    median of the filtered values as the spectral estimate.
    """

    def __init__(self, window: int = 5):
        self.window = window

    @property
    def name(self) -> str:
        return f"CW-Median RDFT (w={self.window})"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        N = len(signal)
        n = np.arange(N)
        k = np.arange(N)
        twiddle = np.exp(-1j * 2 * np.pi * np.outer(n, k) / N)
        dft_matrix = signal[:, np.newaxis] * twiddle  # (N, N)

        spectrum = np.zeros(N, dtype=complex)
        for ki in range(N):
            re_filtered = cwmedian_1d(np.real(dft_matrix[:, ki]), self.window)
            im_filtered = cwmedian_1d(np.imag(dft_matrix[:, ki]), self.window)
            spectrum[ki] = np.median(re_filtered) + 1j * np.median(im_filtered)

        return spectrum
