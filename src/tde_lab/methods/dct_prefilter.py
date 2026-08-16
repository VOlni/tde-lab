"""DCT pre-filter decorator — wraps any BaseMethod."""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from tde_lab.methods.base import BaseMethod, MCFResult


def dct_threshold_filter(signal: np.ndarray, beta: float = 2.7) -> np.ndarray:
    """
    DCT-II hard-threshold filter (MAD-based noise estimate).

    1. Forward DCT-II (orthonormal)
    2. Noise floor = median(|coeffs|) / 0.6745  (robust MAD estimator)
    3. Hard threshold at beta * noise_floor
    4. Inverse DCT
    """
    coeffs = dct(signal, norm="ortho")
    noise_floor = np.median(np.abs(coeffs)) / 0.6745
    threshold = beta * noise_floor
    coeffs_thresh = coeffs * (np.abs(coeffs) >= threshold)
    return idct(coeffs_thresh, norm="ortho")


class DCTPreFilter(BaseMethod):
    """
    Decorator: applies DCT threshold filtering to each signal column before
    delegating to the wrapped method for RDFT and MCF computation.

    Can chain with any method:
        method = DCTPreFilter(HodgesLehmannRDFT(), beta=2.7)
    """

    def __init__(self, method: BaseMethod, beta: float = 2.7):
        self.method = method
        self.beta = beta

    @property
    def name(self) -> str:
        return f"DCT + {self.method.name}"

    def compute_boundaries(self, clean_1d, sdvig):
        return self.method.compute_boundaries(clean_1d, sdvig)

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        return self.method._compute_rdft_single(
            dct_threshold_filter(signal, self.beta)
        )

    def compute_mcf(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        lags: np.ndarray,
        boundaries: tuple[int, int],
        mic_distance: float = 1.0,
    ) -> MCFResult:
        sig1_f = self._filter(sig1)
        sig2_f = self._filter(sig2)
        result = self.method.compute_mcf(sig1_f, sig2_f, lags, boundaries, mic_distance)
        result.method_name = self.name
        return result

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            return dct_threshold_filter(signal, self.beta)
        out = np.empty_like(signal)
        for col in range(signal.shape[1]):
            out[:, col] = dct_threshold_filter(signal[:, col], self.beta)
        return out
