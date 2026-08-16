import numpy as np
import pytest

from tde_lab.analysis.accumulator import StatsAccumulator
from tde_lab.analysis.sweep_store import SweepStore, config_fingerprint
from tde_lab.config.settings import SignalConfig
from tde_lab.experiments import sas_sweep
from tde_lab.methods.standard_fft import StandardFFT


# ── StatsAccumulator ─────────────────────────────────────────────────────────

def test_accumulator_single_pass_equals_chunked():
    peaks = np.random.randint(0, 200, size=1000)
    mask = np.random.rand(1000) < 0.8

    single = StatsAccumulator()
    single.update(peaks, mask)

    chunked = StatsAccumulator()
    for lo in range(0, 1000, 137):
        part = StatsAccumulator()
        part.update(peaks[lo:lo + 137], mask[lo:lo + 137])
        chunked.merge(part)

    assert chunked.n_total == single.n_total == 1000
    assert chunked.n_normal == single.n_normal
    assert chunked.pabn_percent == pytest.approx(single.pabn_percent)
    assert chunked.sigma == pytest.approx(single.sigma)


def test_accumulator_sigma_matches_matlab_convention():
    # sigma = sqrt(mean((norm_oc - mean(norm_oc))^2)) over normal peaks only
    peaks = np.array([100, 102, 104, 999])
    mask = np.array([True, True, True, False])
    acc = StatsAccumulator()
    acc.update(peaks, mask)

    normal = peaks[mask].astype(float)
    expected = np.sqrt(np.mean((normal - normal.mean()) ** 2))
    assert acc.sigma == pytest.approx(expected)
    assert acc.pabn_percent == pytest.approx(25.0)


def test_accumulator_degenerate_cases():
    acc = StatsAccumulator()
    assert np.isnan(acc.normal_rate)
    acc.update(np.array([5]), np.array([True]))
    assert np.isnan(acc.sigma)  # < 2 normal estimates
    assert acc.norm_percent == 100.0


# ── SweepStore ───────────────────────────────────────────────────────────────

def test_store_roundtrip_and_fingerprint(tmp_path):
    config = {"experiment": "sas_sweep", "frag_length": 256, "ws": 3}
    store = SweepStore(tmp_path, config)
    store.write_manifest()

    acc = StatsAccumulator(n_total=500, n_normal=480, peak_sum=1e5, peak_sumsq=2e7)
    assert not store.has("Standard FFT", 1.6, 3.0, 0)
    store.save("Standard FFT", 1.6, 3.0, 0, acc)
    assert store.has("Standard FFT", 1.6, 3.0, 0)

    loaded = store.load("Standard FFT", 1.6, 3.0, 0)
    assert loaded == acc

    # different config → different cache namespace
    other = SweepStore(tmp_path, {**config, "ws": 5})
    assert not other.has("Standard FFT", 1.6, 3.0, 0)
    assert config_fingerprint(config) != config_fingerprint({**config, "ws": 5})

    store.clear()
    assert not store.has("Standard FFT", 1.6, 3.0, 0)


# ── sas_sweep chunking / resume ──────────────────────────────────────────────

def test_small_gamma_rule():
    grid = sas_sweep.gammas_for_alpha(1.2, [0, 1, 2, 3, 4, 5, 6])
    assert grid == pytest.approx([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert sas_sweep.gammas_for_alpha(2.0, [0, 1, 2]) == [0, 1, 2]


def _run_small_sweep(tmp_path, **kwargs):
    return sas_sweep.run(
        methods=[StandardFFT()],
        signal_config=SignalConfig(frag_length=128, frags=8, tau=0.1, ws=3),
        alpha_values=[2.0, 1.2],
        gamma_values=[0.0, 1.0],
        realizations=20,
        chunk_size=8,
        cache_dir=str(tmp_path / "cache"),
        seed=42,
        save=False,
        **kwargs,
    )


def test_sweep_chunked_run_and_resume(tmp_path, monkeypatch):
    result = _run_small_sweep(tmp_path)

    # grid respected, incl. small-gamma rescale for alpha=1.2
    assert result.gamma_grid[2.0] == [0.0, 1.0]
    assert result.gamma_grid[1.2] == pytest.approx([0.0, 0.1])
    cells = result.stats["Standard FFT"]
    assert all(acc.n_total == 20 for acc in cells.values())  # 8 + 8 + 4 chunks

    # noiseless cells must be perfect
    assert cells[(2.0, 0.0)].pabn_percent == 0.0

    # second run: everything served from cache, no chunk recomputed
    calls = []
    original = sas_sweep._run_cell_chunk
    monkeypatch.setattr(
        sas_sweep, "_run_cell_chunk",
        lambda *a, **k: calls.append(1) or original(*a, **k),
    )
    result2 = _run_small_sweep(tmp_path)
    assert calls == []
    for key, acc in result2.stats["Standard FFT"].items():
        assert acc == cells[key]

    # fresh run discards the cache and recomputes
    result3 = _run_small_sweep(tmp_path, resume=False)
    assert len(calls) == 12  # 2 alphas x 2 gammas x 3 chunks
    # same seed → identical statistics
    for key, acc in result3.stats["Standard FFT"].items():
        assert acc == cells[key]
