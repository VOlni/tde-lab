"""Distance/dissimilarity functions for the 'new approach' delay estimators.

Ported from DSP_new_approach/estimators.m and the circshift_sas_* scripts.
Every function has the same shape contract:

    ref   : (N,)   reference fragment
    cands : (L, N) candidate fragments — the delayed channel circularly
            un-shifted by each trial lag (one row per lag)
    →       (L,)   distance of each candidate to the reference

The delay estimate is the argmin of the returned curve.

MATLAB porting notes
--------------------
- euclidean_power applies the exponent INSIDE the sum (Σ|x−y|^b).  The MATLAB
  sketch writes sum(...)^b, but a power outside the sum is monotone and cannot
  change the argmin, while the cached pow05/pow15 result matrices differ from
  b=1 — so the runs that produced them must have used the element-wise form.
- hellinger takes sqrt of signal samples, which are zero-mean and go negative;
  MATLAB silently switches to complex arithmetic there, numpy would produce
  NaN.  We replicate MATLAB by computing in complex and taking |·|.
- mahalanobis_immse is a literal port of the MATLAB formula built on immse();
  it is NOT a real Mahalanobis distance and degenerates when the channels are
  nearly equal (immse → 0).  Kept for completeness, excluded from defaults.
"""
from __future__ import annotations

import numpy as np


def euclidean_power(ref: np.ndarray, cands: np.ndarray, b: float = 1.0) -> np.ndarray:
    """Σ|x−y|^b with the power inside the sum (b = 0.5 / 1.0 / 1.5)."""
    return np.sum(np.abs(cands - ref) ** b, axis=1)


def minkowski(ref: np.ndarray, cands: np.ndarray, p: float = 2.0) -> np.ndarray:
    """(Σ|x−y|^p)^(1/p)."""
    return np.sum(np.abs(cands - ref) ** p, axis=1) ** (1.0 / p)


def canberra(ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Σ |x−y| / (|x|+|y|), with 0/0 terms counted as 0."""
    num = np.abs(cands - ref)
    den = np.abs(cands) + np.abs(ref)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).sum(axis=1)


def bray_curtis(ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Σ|x−y| / Σ|x+y|."""
    num = np.abs(cands - ref).sum(axis=1)
    den = np.abs(cands + ref).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def hellinger(ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """(1/√2)·sqrt(Σ|√x−√y|²), complex sqrt to replicate MATLAB on negatives."""
    diff = np.sqrt(cands.astype(complex)) - np.sqrt(ref.astype(complex))
    return (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum(np.abs(diff) ** 2, axis=1))


def mahalanobis_immse(ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """sqrt((Σ|x−y|)² / immse(x,y)²) — literal MATLAB port, immse = mean((x−y)²)."""
    diff = cands - ref
    num = np.abs(diff).sum(axis=1) ** 2
    immse = np.mean(diff ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num / immse ** 2)


def cosine_distance(ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """1 − Σxy / (‖x‖·‖y‖)."""
    num = cands @ ref
    den = np.linalg.norm(cands, axis=1) * np.linalg.norm(ref)
    return 1.0 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def pearson_distance(ref: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """1 − sample correlation coefficient."""
    ref_c = ref - ref.mean()
    cands_c = cands - cands.mean(axis=1, keepdims=True)
    num = cands_c @ ref_c
    den = np.linalg.norm(cands_c, axis=1) * np.linalg.norm(ref_c)
    return 1.0 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)
