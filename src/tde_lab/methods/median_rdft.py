"""Median Robust DFT."""
from __future__ import annotations

import numpy as np

from tde_lab.methods.base import BaseMethod


class MedianRDFT(BaseMethod):
    """
    Robust DFT: for each frequency bin k, compute
        X[k] = median_n(Re(x[n]·e^{-j2πnk/N})) + j·median_n(Im(x[n]·e^{-j2πnk/N}))
    across n = 0..N-1 (sample dimension).

    One impulsive sample x[n*] affects at most one term in the N products for
    each bin but cannot shift the median, making this robust to impulse noise.
    """

    @property
    def name(self) -> str:
        return "Median RDFT"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        N = len(signal)
        n = np.arange(N)
        k = np.arange(N)
        # twiddle[n, k] = exp(-j·2π·n·k/N)
        twiddle = np.exp(-1j * 2 * np.pi * np.outer(n, k) / N)  # (N, N)
        # dft_matrix[n, k] = signal[n] * twiddle[n, k]
        dft_matrix = signal[:, np.newaxis] * twiddle              # (N, N)
        # median across n (axis=0) for each k
        return (np.median(np.real(dft_matrix), axis=0)
                + 1j * np.median(np.imag(dft_matrix), axis=0))
