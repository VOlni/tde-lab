"""Distance-metric delay estimators — the 'new approach' family.

Instead of locating the argmax of a cross-correlation, these methods build a
distance curve S_e(j) between the reference channel and the delayed channel
circularly un-shifted by each trial lag j, and estimate the delay as the
ARGMIN of that curve (DSP_new_approach/circshift_sas_* scripts).

They plug into the same MCFResult / ExperimentRunner / plotting pipeline as
the correlation methods; results carry extra["curve_kind"] = "distance" so
plots label the curve S_e(j) and mark the minimum.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from tde_lab.config.settings import SPEED_OF_SOUND
from tde_lab.methods.base import BaseMethod, MCFResult
from tde_lab.methods import distance_metrics as dm


class DistanceMethod(BaseMethod):
    """
    Base class for argmin-of-distance-curve delay estimators.

    Parameters
    ----------
    lag_limit        : half-width of the search window in samples, or None to
                       search every circular lag.  MATLAB used ±100 with the
                       true delay at 0; the window must contain the true delay.
    normal_halfwidth : classification window around the clean-curve argmin
                       (MATLAB: ±5 samples).
    """

    def __init__(self, lag_limit: int | None = None, normal_halfwidth: int = 5):
        self.lag_limit = lag_limit
        self.normal_halfwidth = normal_halfwidth

    @abstractmethod
    def _distance_curve(self, ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """Distance of each candidate row to the reference — see distance_metrics."""

    def _compute_rdft_single(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} is a distance method; it has no spectral estimator."
        )

    # ── lag geometry ─────────────────────────────────────────────────────────

    def _candidate_shifts(self, n: int) -> np.ndarray:
        """Trial lag values j.  Index k in the curve ↔ lag shifts[k]."""
        if self.lag_limit is None:
            return np.arange(n) - n // 2          # same convention as fftshifted MCF
        return np.arange(-self.lag_limit, self.lag_limit + 1)

    @staticmethod
    def _candidates(frag: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        """(L, N) matrix whose row for lag j equals np.roll(frag, -j)."""
        n = len(frag)
        idx = (np.arange(n)[None, :] + shifts[:, None]) % n
        return frag[idx]

    # ── boundaries: clean-curve argmin ± halfwidth (MATLAB convention) ───────

    def compute_boundaries(self, clean_1d: np.ndarray, sdvig: int) -> tuple[int, int]:
        shifts = self._candidate_shifts(len(clean_1d))
        delayed = np.roll(clean_1d, sdvig)
        curve = self._distance_curve(clean_1d, self._candidates(delayed, shifts))
        center = int(np.argmin(np.where(np.isnan(curve), np.inf, curve)))
        return center - self.normal_halfwidth, center + self.normal_halfwidth

    # ── main pipeline ────────────────────────────────────────────────────────

    def compute_mcf(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        lags: np.ndarray,
        boundaries: tuple[int, int],
        mic_distance: float = 1.0,
    ) -> MCFResult:
        if sig1.ndim == 1:
            sig1 = sig1[:, None]
            sig2 = sig2[:, None]
        n, frags = sig1.shape
        dt = float(lags[1] - lags[0]) if len(lags) > 1 else 1.0

        shifts = self._candidate_shifts(n)
        curve = np.empty((len(shifts), frags))
        for f in range(frags):
            cands = self._candidates(sig2[:, f], shifts)
            curve[:, f] = self._distance_curve(sig1[:, f], cands)

        # NaN-safe argmin (MATLAB min() also skips NaN)
        curve_safe = np.where(np.isnan(curve), np.inf, curve)
        peaks = np.argmin(curve_safe, axis=0)          # (frags,) indices into shifts

        left, right = boundaries
        normal_mask = (peaks >= left) & (peaks <= right)
        normal_peaks = peaks[normal_mask].astype(float)

        normal_rate = float(normal_mask.mean())
        if len(normal_peaks) >= 2:
            mse = float(np.mean((normal_peaks - normal_peaks.mean()) ** 2))
        else:
            mse = float("nan")

        mean_curve = curve.mean(axis=1)
        min_idx = int(np.argmin(np.where(np.isnan(mean_curve), np.inf, mean_curve)))
        delay_samples = int(shifts[min_idx])
        delay_seconds = delay_samples * dt

        sin_val = np.clip(delay_seconds * SPEED_OF_SOUND / mic_distance, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arcsin(sin_val)))

        return MCFResult(
            mcf=mean_curve,
            lags=shifts * dt,                          # curve's own lag axis (length L)
            delay_samples=delay_samples,
            delay_seconds=delay_seconds,
            angle_degrees=angle_deg,
            normal_rate=normal_rate,
            mse=mse,
            method_name=self.name,
            extra={"peaks": peaks, "curve_kind": "distance"},
        )


# ── concrete metrics ─────────────────────────────────────────────────────────

class EuclideanPowerDistance(DistanceMethod):
    def __init__(self, b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.b = b

    @property
    def name(self) -> str:
        return f"Euclidean (b={self.b:g})"

    def _distance_curve(self, ref, cands):
        return dm.euclidean_power(ref, cands, self.b)


class MinkowskiDistance(DistanceMethod):
    def __init__(self, p: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    @property
    def name(self) -> str:
        return f"Minkowski (p={self.p:g})"

    def _distance_curve(self, ref, cands):
        return dm.minkowski(ref, cands, self.p)


class CanberraDistance(DistanceMethod):
    @property
    def name(self) -> str:
        return "Canberra"

    def _distance_curve(self, ref, cands):
        return dm.canberra(ref, cands)


class BrayCurtisDistance(DistanceMethod):
    @property
    def name(self) -> str:
        return "Bray-Curtis"

    def _distance_curve(self, ref, cands):
        return dm.bray_curtis(ref, cands)


class HellingerDistance(DistanceMethod):
    @property
    def name(self) -> str:
        return "Hellinger"

    def _distance_curve(self, ref, cands):
        return dm.hellinger(ref, cands)


class MahalanobisIMMSEDistance(DistanceMethod):
    @property
    def name(self) -> str:
        return "Mahalanobis (immse)"

    def _distance_curve(self, ref, cands):
        return dm.mahalanobis_immse(ref, cands)


class CosineDistance(DistanceMethod):
    @property
    def name(self) -> str:
        return "Cosine"

    def _distance_curve(self, ref, cands):
        return dm.cosine_distance(ref, cands)


class PearsonDistance(DistanceMethod):
    @property
    def name(self) -> str:
        return "Pearson corr."

    def _distance_curve(self, ref, cands):
        return dm.pearson_distance(ref, cands)
