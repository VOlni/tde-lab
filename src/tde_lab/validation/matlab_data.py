"""Registry and loader for the cached MATLAB result matrices.

The DSP_new_approach experiments stored Pabn (percent of abnormal estimates)
and sigma (RMSE of normal estimates, samples) matrices per method over the
grid alpha = [2, 1.8, 1.6, 1.4, 1.2] × gamma = [0..6] (gamma × 0.1 for the
alpha = 1.2 column).  Files store them as (7 gammas, 5 alphas) or transposed
depending on the script — the loader normalizes to (alpha, gamma).

Experiment conventions per circshift_sas_sig_set_generation.m: WS = 3,
frag_length = 1024, true delay 0, lags ±100, normal window = clean-curve
argmin ± 5, reference channel scaled by 0.5 (w3_05 sets) or 0.8 (w3_08).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ALPHA_VALUES = [2.0, 1.8, 1.6, 1.4, 1.2]
GAMMA_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@dataclass(frozen=True)
class MatlabRef:
    """One cached MATLAB matrix and the Python setup that reproduces it."""
    path: str                # .mat file, relative to the MATLAB research root
    var: str                 # variable inside the file
    method_key: str          # tde_lab method registry key
    metric: str = "pabn_percent"     # "pabn_percent" | "sigma"
    gain1: float = 0.5       # reference-channel gain
    ws: int = 3
    tau: float = 0.0         # true relative delay of the MATLAB run
    strict: bool = True      # False = overlay/report only, no pass/fail
    n_matlab: int = 1000     # realizations behind the cached numbers (tolerance)


_D = "DSP_new_approach"

REGISTRY: list[MatlabRef] = [
    # ── w3_05 family: gain 0.5, the main distance-metric comparison set ──────
    MatlabRef(f"{_D}/data_w3_05.mat", "panom_conventional_w3_05", "standard"),
    MatlabRef(f"{_D}/data_w3_05.mat", "panom_w3_05", "dist-l1"),
    # cached pow05/pow15 predate the power-inside-the-sum reading — qualitative
    MatlabRef(f"{_D}/data_w3_05.mat", "panom_w3_05_pow05", "dist-pow05", strict=False),
    MatlabRef(f"{_D}/data_w3_05.mat", "panom_w3_05_pow15", "dist-pow15", strict=False),
    MatlabRef(f"{_D}/estimatores/minkovskii_p1_w3_05.mat", "panom_minkovskii_p1", "dist-mink1"),
    MatlabRef(f"{_D}/estimatores/minkovskii_p2_w3_05.mat", "panom_minkovskii_p2", "dist-mink2"),
    MatlabRef(f"{_D}/estimatores/canberra_w3_05.mat", "panom_canberra_w3_05", "dist-canberra"),
    MatlabRef(f"{_D}/estimatores/brey_curtis_w3_05.mat", "panom_brey_curtis_w3_05", "dist-braycurtis"),
    MatlabRef(f"{_D}/estimatores/hellinger_w3_05.mat", "panom_hellinger_w3_05", "dist-hellinger"),
    MatlabRef(f"{_D}/estimatores/cosine_w3_05.mat", "panom_cosine_w3_05", "dist-cosine"),
    MatlabRef(f"{_D}/estimatores/pcc_w3_05.mat", "panom_pcc_w3_05", "dist-pearson"),
    # degenerate immse-Mahalanobis: literal port, curves only
    MatlabRef(f"{_D}/estimatores/mahalanobis_w3_05.mat", "panom_mahalanobis_w3_05",
              "dist-mahalanobis", strict=False),
    # sigma counterparts (loose: MATLAB norm_oc scalar-init bug taints cells
    # with few normal estimates)
    MatlabRef(f"{_D}/data_w3_05.mat", "sigma_conventional_w3_05", "standard",
              metric="sigma", strict=False),
    MatlabRef(f"{_D}/data_w3_05.mat", "sigma_w3_05", "dist-l1", metric="sigma", strict=False),

    # ── w3_08 family: gain 0.8 ────────────────────────────────────────────────
    MatlabRef(f"{_D}/data_w3_08.mat", "panom_conventional_w3_08", "standard", gain1=0.8),
    MatlabRef(f"{_D}/data_w3_08.mat", "panom_w3_08", "dist-l1", gain1=0.8),
    MatlabRef(f"{_D}/data_w3_08.mat", "panom_w3_08_pow05", "dist-pow05", gain1=0.8, strict=False),
    MatlabRef(f"{_D}/data_w3_08.mat", "panom_w3_08_pow15", "dist-pow15", gain1=0.8, strict=False),

    # ── all_conv: 10k-realization conventional benchmark (setup less
    #    documented — reference overlay only) ──────────────────────────────────
    MatlabRef(f"{_D}/all_conv.mat", "panom", "standard", gain1=1.0, tau=0.1,
              strict=False, n_matlab=10_000),
]


def load_ref(mat_root: str | Path, ref: MatlabRef) -> np.ndarray:
    """Load one matrix as (5 alphas, 7 gammas), percent / samples as stored."""
    data = loadmat(Path(mat_root) / ref.path)
    if ref.var not in data:
        raise KeyError(f"{ref.var!r} not found in {ref.path}")
    m = np.asarray(data[ref.var], dtype=float)
    if m.shape == (len(GAMMA_VALUES), len(ALPHA_VALUES)):
        m = m.T
    if m.shape != (len(ALPHA_VALUES), len(GAMMA_VALUES)):
        raise ValueError(f"{ref.path}:{ref.var} has unexpected shape {m.shape}")
    return m
