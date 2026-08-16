"""Alpha-trimmed mean Robust DFT."""
from __future__ import annotations

import numpy as np

from tde_lab.methods.base import BaseMethod


class AlphaTrimmedRDFT(BaseMethod):
    """
    Robust DFT using alpha-trimmed mean across the sample dimension.

    trim_percent = 0   → ordinary mean (equivalent to standard DFT)
    trim_percent = 50  → median
    """

    def __init__(self, trim_percent: float = 25.0):
        if not (0 <= trim_percent <= 50):
            raise ValueError("trim_percent must be in [0, 50]")
        self.trim_percent = trim_percent

    @property
    def name(self) -> str:
        return f"Alpha-Trimmed RDFT ({self.trim_percent:.0f}%)"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        N = len(signal)
        n = np.arange(N)
        k = np.arange(N)
        twiddle = np.exp(-1j * 2 * np.pi * np.outer(n, k) / N)  # (N, N)
        dft_matrix = signal[:, np.newaxis] * twiddle              # (N, N)

        alpha = self.trim_percent / 100.0
        lower = round(N * alpha)
        upper = int(N - N * alpha)

        re_sorted = np.sort(np.real(dft_matrix), axis=0)   # sort across n
        im_sorted = np.sort(np.imag(dft_matrix), axis=0)

        if alpha == 0:
            re_est = np.mean(re_sorted, axis=0)
            im_est = np.mean(im_sorted, axis=0)
        elif alpha >= 0.5:
            re_est = np.median(re_sorted, axis=0)
            im_est = np.median(im_sorted, axis=0)
        else:
            re_est = np.mean(re_sorted[lower:upper, :], axis=0)
            im_est = np.mean(im_sorted[lower:upper, :], axis=0)

        return re_est + 1j * im_est
