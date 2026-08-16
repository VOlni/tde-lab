"""Standard FFT cross-correlation (reference method) + sub-sample variant."""
from __future__ import annotations

import numpy as np

from tde_lab.methods.base import BaseMethod, MCFResult
from tde_lab.config.settings import SPEED_OF_SOUND


class StandardFFT(BaseMethod):
    """Classical FFT cross-correlation — baseline reference."""

    @property
    def name(self) -> str:
        return "Standard FFT"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        return np.fft.fft(signal)


class SubSampleFFT(BaseMethod):
    """
    Sub-sample TDE via parabolic interpolation of the per-fragment
    cross-correlation peak (methods.subsample).  Fractional-sample delay
    resolution.
    """

    @property
    def name(self) -> str:
        return "Sub-sample FFT"

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        return np.fft.fft(signal)

    def compute_mcf(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        lags: np.ndarray,
        boundaries: tuple[int, int],
        mic_distance: float = 1.0,
    ) -> MCFResult:
        """Integer classification like the base pipeline; the delay estimate is
        the median of per-fragment sub-sample delays over the
        normally-classified fragments."""
        from tde_lab.methods.subsample import subsample_delay

        if sig1.ndim == 1:
            sig1 = sig1[:, None]
            sig2 = sig2[:, None]
        N, frags = sig1.shape
        dt = float(lags[1] - lags[0]) if len(lags) > 1 else 1.0

        SP1 = np.fft.fft(sig1, axis=0)
        SP2 = np.fft.fft(sig2, axis=0)

        # standard per-fragment classification (comparable with other methods)
        cross_all = SP2 * np.conj(SP1)
        mcf_abs = np.abs(np.fft.fftshift(np.fft.ifft(cross_all, axis=0), axes=0)) / N
        peaks = np.argmax(mcf_abs, axis=0)
        left, right = boundaries
        normal_mask = (peaks >= left) & (peaks <= right)
        normal_peaks = peaks[normal_mask].astype(float)
        normal_rate = float(normal_mask.mean())
        mse = float(np.mean((normal_peaks - normal_peaks.mean()) ** 2)) if len(normal_peaks) >= 2 else float("nan")

        # per-fragment sub-sample delays: sig2 = delayed → positive delay
        fractional_delays = np.array([
            subsample_delay(sig2[:, f], sig1[:, f]) for f in range(frags)
        ])

        pool = fractional_delays[normal_mask] if normal_mask.any() else fractional_delays
        delta_frac = float(np.median(pool))
        delay_seconds = delta_frac * dt

        sin_val = np.clip(delay_seconds * SPEED_OF_SOUND / mic_distance, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arcsin(sin_val)))

        mcf_display = np.mean(mcf_abs, axis=1) if frags > 1 else mcf_abs[:, 0]

        return MCFResult(
            mcf=mcf_display,
            lags=lags,
            delay_samples=int(round(delta_frac)),
            delay_seconds=delay_seconds,
            angle_degrees=angle_deg,
            normal_rate=normal_rate,
            mse=mse,
            method_name=self.name,
            extra={
                "fractional_delay": delta_frac,
                "fractional_delays": fractional_delays,
                "peaks": peaks,
            },
        )
