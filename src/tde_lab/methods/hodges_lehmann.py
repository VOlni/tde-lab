"""Hodges-Lehmann Robust DFT."""
from __future__ import annotations

import numpy as np

from tde_lab.methods.base import BaseMethod


def hodges_lehmann_1d(x: np.ndarray) -> float:
    """
    Hodges-Lehmann estimate of location for a 1-D sample.

    Computes the median of all Walsh averages (x[i]+x[j])/2, i<=j,
    approximated efficiently via the sorted-pairs formula used in the
    original MATLAB hl.m: augment sorted array with pairwise means
    of symmetric elements, then take the median.
    """
    xs = np.sort(x)
    n = len(xs)
    # Walsh averages of symmetric pairs: (xs[i] + xs[n-1-i]) / 2
    half = n // 2
    pairs = (xs[:half] + xs[n - 1: n - 1 - half: -1]) / 2.0
    combined = np.concatenate([xs, pairs])
    return float(np.median(combined))


def hodges_lehmann_cols(mat: np.ndarray) -> np.ndarray:
    """Apply hodges_lehmann_1d to each column of mat (N, M) → (M,)."""
    return np.array([hodges_lehmann_1d(mat[:, j]) for j in range(mat.shape[1])])


class HodgesLehmannRDFT(BaseMethod):
    """
    Robust DFT using the Hodges-Lehmann estimator across the sample dimension.
    More efficient than median under near-Gaussian noise while remaining robust.
    """

    @property
    def name(self) -> str:
        return "Hodges-Lehmann RDFT"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        N = len(signal)
        n = np.arange(N)
        k = np.arange(N)
        twiddle = np.exp(-1j * 2 * np.pi * np.outer(n, k) / N)
        dft_matrix = signal[:, np.newaxis] * twiddle  # (N, N)

        re_est = hodges_lehmann_cols(np.real(dft_matrix))
        im_est = hodges_lehmann_cols(np.imag(dft_matrix))
        return re_est + 1j * im_est
