"""Abstract base class for all TDE methods and the shared MCFResult dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from tde_lab.config.settings import SPEED_OF_SOUND


@dataclass
class MCFResult:
    """
    Output of one method applied to one signal pair.

    mcf           – aggregate |MCF| (mean over fragments), fftshifted, length = frag_length
    lags          – time-lag axis in seconds (same length as mcf)
    delay_samples – delay estimate in samples (positive = sig2 lags sig1)
    delay_seconds – delay_samples * dt
    angle_degrees – DOA angle θ = arcsin(delay_seconds * C / mic_distance)
    normal_rate   – fraction of fragments whose peak landed inside the main lobe [0..1]
    mse           – MSE of normal-estimate peak positions (in samples²)
    method_name   – name tag of the method that produced this result
    """
    mcf: np.ndarray
    lags: np.ndarray
    delay_samples: int
    delay_seconds: float
    angle_degrees: float
    normal_rate: float
    mse: float
    method_name: str
    extra: dict = field(default_factory=dict)


class BaseMethod(ABC):
    """
    Common interface for all TDE / RDFT methods.

    Each subclass implements `_compute_rdft_single(signal_1d)` — the robust DFT
    estimator applied to ONE fragment of length N.

    For each frequency k the standard DFT sums:
        X[k] = Σ_n  x[n] * exp(-j·2π·n·k/N)
    The robust variant replaces the sum with a location estimator applied to the
    N products {x[n]·exp(-j·2π·n·k/N), n=0..N-1}, separately for Re and Im parts.

    The shared `compute_mcf` pipeline:
        1. Compute robust DFT for every fragment of sig1 and sig2 → (N, frags) each
        2. Cross-power per fragment: SP2 · conj(SP1)
           Convention: SP2 = FFT(delayed signal), SP1 = FFT(original)
           ⟹ peak of ifft(cross) at lag = +sdvig (positive = sig2 behind sig1)
        3. IFFT + fftshift per fragment → (N, frags) MCF
        4. Peak per fragment → (frags,) delay estimates
        5. Classify normal / abnormal using MCF main-lobe boundaries
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        """
        Robust DFT of a single 1-D fragment.

        Parameters
        ----------
        signal : (N,) float array — one fragment

        Returns
        -------
        spectrum : (N,) complex array — robust spectral estimate
        """

    def compute_boundaries(self, clean_1d: np.ndarray, sdvig: int) -> tuple[int, int]:
        """
        Main-lobe boundaries for normal/abnormal classification, in the same
        index space as this method's per-fragment peaks.

        Correlation methods use the MCF 5%-lobe rule; distance methods
        override this with the clean-curve argmin ± halfwidth convention.
        """
        # local import: analysis.runner imports this module
        from tde_lab.analysis.metrics import mcf_boundaries
        return mcf_boundaries(clean_1d, sdvig=sdvig)

    def compute_rdft(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute robust DFT for one or many fragments.

        Parameters
        ----------
        signal : (N,) or (N, frags) array

        Returns
        -------
        (N,) or (N, frags) complex array — one spectrum per fragment
        """
        if signal.ndim == 1:
            return self._compute_rdft_single(signal)

        N, frags = signal.shape
        output = np.zeros((N, frags), dtype=complex)
        for col in range(frags):
            output[:, col] = self._compute_rdft_single(signal[:, col])
        return output

    def compute_mcf(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        lags: np.ndarray,
        boundaries: tuple[int, int],
        mic_distance: float = 1.0,
    ) -> MCFResult:
        """
        Full per-fragment MCF pipeline shared by all methods.

        Parameters
        ----------
        sig1, sig2    : (N, frags) — sig2 is the delayed version of sig1
        lags          : time-lag axis (seconds), length = N
        boundaries    : (left, right) sample indices of the MCF main lobe
                        in the fftshifted domain (from analysis.metrics.mcf_boundaries)
        mic_distance  : distance between microphones in metres
        """
        N = sig1.shape[0]
        frags = sig1.shape[1] if sig1.ndim > 1 else 1
        dt = float(lags[1] - lags[0]) if len(lags) > 1 else 1.0

        # Robust DFT for every fragment
        SP1 = self.compute_rdft(sig1)   # (N, frags)
        SP2 = self.compute_rdft(sig2)   # (N, frags)

        # Cross-power: SP2 * conj(SP1) — peak at +sdvig after ifft+fftshift
        cross = SP2 * np.conj(SP1)      # (N, frags)

        # Per-fragment MCF
        mcf_per_frag = np.fft.fftshift(
            np.fft.ifft(cross, axis=0), axes=0
        ) / N                           # (N, frags)

        mcf_abs = np.abs(mcf_per_frag)  # (N, frags)

        # Peak position per fragment
        peaks = np.argmax(mcf_abs, axis=0)  # (frags,)

        # Normal / abnormal classification
        left, right = boundaries
        normal_mask = (peaks >= left) & (peaks <= right)
        normal_peaks = peaks[normal_mask].astype(float)

        normal_rate = float(normal_mask.mean())
        if len(normal_peaks) >= 2:
            mse = float(np.mean((normal_peaks - normal_peaks.mean()) ** 2))
        else:
            mse = float("nan")

        # Aggregate MCF for display (mean over fragments)
        mcf_display = np.mean(mcf_abs, axis=1) if frags > 1 else mcf_abs[:, 0]

        # Delay from the dominant peak of the aggregate MCF
        peak_idx = int(np.argmax(mcf_display))
        delay_samples = peak_idx - N // 2
        delay_seconds = delay_samples * dt

        sin_val = np.clip(delay_seconds * SPEED_OF_SOUND / mic_distance, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arcsin(sin_val)))

        return MCFResult(
            mcf=mcf_display,
            lags=lags,
            delay_samples=delay_samples,
            delay_seconds=delay_seconds,
            angle_degrees=angle_deg,
            normal_rate=normal_rate,
            mse=mse,
            method_name=self.name,
            extra={"peaks": peaks},
        )
