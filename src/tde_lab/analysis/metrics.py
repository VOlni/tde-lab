"""Metrics and MCF boundary detection."""
from __future__ import annotations

import numpy as np
from tde_lab.config.settings import SPEED_OF_SOUND


def mcf_boundaries(
    clean_signal: np.ndarray,
    sdvig: int,
    threshold: float = 0.05,
) -> tuple[int, int]:
    """
    Determine the left/right sample indices of the MCF main lobe in the
    fftshifted domain, matching the MCF_boundaries.m convention.

    The cross-correlation of the shifted signal with the original is used
    (SP2_shifted * conj(SP1_original)), which produces a peak at sdvig + N//2
    in the fftshifted result.  The 5% threshold finds the lobe edges.

    Parameters
    ----------
    clean_signal : 1-D noise-free fragment (first column of the clean array)
    sdvig        : delay in samples (positive integer)
    threshold    : fraction of peak height below which lobe ends (default 5%)

    Returns
    -------
    (left, right) integer indices in the fftshifted MCF array
    """
    N = len(clean_signal)

    # Create delayed version (same cyclic shift as in SpeechLikeGenerator)
    shifted = np.empty(N)
    if sdvig == 0:
        shifted = clean_signal.copy()
    else:
        shifted[:sdvig] = clean_signal[N - sdvig:]
        shifted[sdvig:] = clean_signal[:N - sdvig]

    # Cross-power: shifted * conj(original) — peak at +sdvig
    SP_shifted  = np.fft.fft(shifted)
    SP_original = np.fft.fft(clean_signal)
    cross = SP_shifted * np.conj(SP_original)

    mcf = np.abs(np.fft.fftshift(np.fft.ifft(cross))) / N

    peak_val = float(mcf.max())
    peak_idx = int(np.argmax(mcf))
    thresh = peak_val * threshold

    left = peak_idx
    while left > 0 and mcf[left] > thresh:
        left -= 1

    right = peak_idx
    while right < N - 1 and mcf[right] > thresh:
        right += 1

    return left, right


def delay_to_angle(delay_seconds: float, mic_distance: float) -> float:
    """θ = arcsin(τ·C / L) in degrees."""
    sin_val = np.clip(delay_seconds * SPEED_OF_SOUND / mic_distance, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_val)))


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio per fragment, averaged across fragments."""
    sig_var = np.var(signal, axis=0)
    noise_var = np.var(noise, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise_var > 0, sig_var / noise_var, np.inf)
    return float(np.mean(snr))
