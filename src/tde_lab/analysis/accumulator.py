"""Streaming statistics for chunked sweep experiments.

A sweep cell (one method at one alpha, gamma point) may be computed in many
chunks of fragments, possibly across separate process runs.  StatsAccumulator
keeps just the sufficient statistics so chunks can be merged in any order and
cached to disk cheaply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StatsAccumulator:
    """Sufficient statistics of per-fragment peak positions.

    sigma matches the MATLAB convention: RMSE of the *normal* peak positions
    around their own mean (mse_norm = sqrt(mean((norm_oc - mean)^2))).
    """
    n_total: int = 0
    n_normal: int = 0
    peak_sum: float = 0.0
    peak_sumsq: float = 0.0

    def update(self, peaks: np.ndarray, normal_mask: np.ndarray) -> None:
        peaks = np.asarray(peaks, dtype=float)
        normal = peaks[np.asarray(normal_mask, dtype=bool)]
        self.n_total += int(peaks.size)
        self.n_normal += int(normal.size)
        self.peak_sum += float(normal.sum())
        self.peak_sumsq += float((normal ** 2).sum())

    def merge(self, other: "StatsAccumulator") -> None:
        self.n_total += other.n_total
        self.n_normal += other.n_normal
        self.peak_sum += other.peak_sum
        self.peak_sumsq += other.peak_sumsq

    @property
    def normal_rate(self) -> float:
        return self.n_normal / self.n_total if self.n_total else float("nan")

    @property
    def norm_percent(self) -> float:
        return 100.0 * self.normal_rate

    @property
    def pabn_percent(self) -> float:
        return 100.0 * (1.0 - self.normal_rate) if self.n_total else float("nan")

    @property
    def sigma(self) -> float:
        """RMSE of normal peak positions (samples)."""
        if self.n_normal < 2:
            return float("nan")
        mean = self.peak_sum / self.n_normal
        var = self.peak_sumsq / self.n_normal - mean ** 2
        return float(np.sqrt(max(var, 0.0)))

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_normal": self.n_normal,
            "peak_sum": self.peak_sum,
            "peak_sumsq": self.peak_sumsq,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StatsAccumulator":
        return cls(
            n_total=int(data["n_total"]),
            n_normal=int(data["n_normal"]),
            peak_sum=float(data["peak_sum"]),
            peak_sumsq=float(data["peak_sumsq"]),
        )
