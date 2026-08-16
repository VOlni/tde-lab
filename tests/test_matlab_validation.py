from pathlib import Path

import numpy as np
import pytest

from tde_lab.validation.compare import _binomial_tolerance, run_validation
from tde_lab.validation.matlab_data import (
    ALPHA_VALUES, GAMMA_VALUES, REGISTRY, load_ref,
)

MAT_ROOT = Path("/Users/slava/Workspace/Research/Studies/coding")

needs_mat = pytest.mark.skipif(
    not (MAT_ROOT / "DSP_new_approach").exists(),
    reason="MATLAB research folder not available on this machine",
)


@needs_mat
@pytest.mark.parametrize("ref", REGISTRY, ids=lambda r: r.var)
def test_registry_files_load_and_normalize(ref):
    m = load_ref(MAT_ROOT, ref)
    assert m.shape == (len(ALPHA_VALUES), len(GAMMA_VALUES))
    finite = m[np.isfinite(m)]
    assert finite.size > 0
    if ref.metric == "pabn_percent":
        assert finite.min() >= 0.0 and finite.max() <= 100.0
        # gamma=0 column is noiseless → (near) zero abnormal estimates,
        # except the degenerate Mahalanobis variant
        if ref.method_key != "dist-mahalanobis":
            assert np.nanmax(m[:, 0]) <= 6.0


def test_binomial_tolerance_floor_and_growth():
    assert _binomial_tolerance(0.0, 0.0, 1000, 1000) == 10.0    # floor
    mid = _binomial_tolerance(50.0, 50.0, 100, 100)             # worst case p=0.5
    assert mid > 10.0
    assert _binomial_tolerance(50.0, 50.0, 10_000, 10_000) < mid


@needs_mat
@pytest.mark.slow
def test_single_cell_validation_standard_and_l1(tmp_path):
    # one alpha row, 500 realizations: Pabn must match the cached w3_05
    # matrices within the binomial tolerance for standard and dist-l1
    report = run_validation(
        mat_root=MAT_ROOT,
        method_keys=["standard", "dist-l1"],
        realizations=500,
        chunk_size=250,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        alpha_values=[1.6],
        show_progress=False,
    )
    strict = [r for r in report.reports if r.ref.strict]
    assert strict, "expected strict registry entries"
    for r in strict:
        assert r.pass_rate >= 0.85, (
            f"{r.ref.var}: pass rate {r.pass_rate:.2f}\n"
            + "\n".join(
                f"  a={c.alpha} g={c.gamma}: py={c.python:.1f} mat={c.matlab:.1f} tol={c.tolerance:.1f}"
                for c in r.cells if not c.passed
            )
        )
