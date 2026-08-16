import numpy as np
import pytest

from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.config.settings import MethodConfig, SignalConfig
from tde_lab.methods import ALL_METHODS, DEFAULT_KEYS, build_method
from tde_lab.methods import distance_metrics as dm
from tde_lab.methods.distance_base import (
    DistanceMethod, EuclideanPowerDistance, MinkowskiDistance,
)
from tde_lab.methods.standard_fft import StandardFFT
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import GaussianNoise

# every distance key except the degenerate immse-Mahalanobis
DIST_KEYS = [k for k in ALL_METHODS if k.startswith("dist-") and k != "dist-mahalanobis"]


def _make_pair(config, variance=0.0):
    gen = SpeechLikeGenerator(config)
    return gen.generate(GaussianNoise(variance), GaussianNoise(variance))


def _run(config, methods, variance=0.0):
    pair = _make_pair(config, variance)
    runner = ExperimentRunner(
        sig1=pair.sig1, sig2=pair.sig2, clean=pair.clean,
        lags=config.lag_axis, sdvig=pair.sdvig,
    )
    return runner.run(methods), pair


# ── pure metric functions ────────────────────────────────────────────────────

def test_metric_functions_zero_at_identity():
    ref = np.random.randn(64)
    cands = np.vstack([ref, np.roll(ref, 5)])
    for func in (lambda r, c: dm.euclidean_power(r, c, 1.0),
                 lambda r, c: dm.minkowski(r, c, 2.0),
                 dm.canberra, dm.bray_curtis, dm.hellinger,
                 dm.cosine_distance, dm.pearson_distance):
        curve = func(ref, cands)
        assert curve.shape == (2,)
        assert curve[0] == pytest.approx(0.0, abs=1e-9)
        assert curve[1] > curve[0]


def test_canberra_zero_over_zero_guard():
    ref = np.array([0.0, 1.0, 0.0, -2.0])
    cands = np.array([[0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0]])
    curve = dm.canberra(ref, cands)
    assert np.all(np.isfinite(curve))
    assert curve[0] == 0.0


def test_hellinger_finite_on_negative_samples():
    # zero-mean signals go negative; MATLAB switches to complex sqrt silently
    ref = np.random.randn(128)
    cands = np.vstack([np.roll(ref, k) for k in range(4)])
    curve = dm.hellinger(ref, cands)
    assert np.all(np.isfinite(curve))
    assert np.isrealobj(curve)


def test_monotone_power_preserves_argmin():
    # Minkowski p=1 and Euclidean b=1 are the same sum — identical argmin;
    # b=0.5/1.5 apply the power per element, so only check argmin agreement
    # on a well-separated noisy pair
    ref = np.random.randn(256)
    noisy = np.roll(ref, 17) + 0.1 * np.random.randn(256)
    shifts = np.arange(-30, 31)
    idx = (np.arange(256)[None, :] + shifts[:, None]) % 256
    cands = noisy[idx]

    l1 = dm.euclidean_power(ref, cands, 1.0)
    mink1 = dm.minkowski(ref, cands, 1.0)
    np.testing.assert_allclose(l1, mink1)
    assert np.argmin(l1) == np.argmin(dm.euclidean_power(ref, cands, 0.5))
    assert int(shifts[np.argmin(l1)]) == 17


# ── DistanceMethod estimators ────────────────────────────────────────────────

@pytest.mark.parametrize("key", DIST_KEYS)
def test_distance_method_recovers_delay_noiseless(key, small_config):
    results, pair = _run(small_config, [build_method(key)])
    (result,) = results.values()

    assert result.extra["curve_kind"] == "distance"
    assert result.delay_samples == pair.sdvig
    assert result.normal_rate == 1.0
    assert result.mse == 0.0
    assert result.delay_seconds == pytest.approx(pair.sdvig * small_config.dt)


def test_windowed_lag_limit(small_config):
    # sdvig = round(0.1 * 256) = 26 fits inside ±30
    cfg = MethodConfig(lag_limit=30)
    method = build_method("dist-l1", cfg)
    results, pair = _run(small_config, [method])
    (result,) = results.values()

    assert len(result.mcf) == 61                       # 2·30 + 1 curve points
    assert result.delay_samples == pair.sdvig
    assert result.normal_rate == 1.0


def test_distance_argmin_matches_fft_argmax(small_config):
    # mild noise: both families must agree on the integer delay
    results, pair = _run(
        small_config,
        [StandardFFT(), MinkowskiDistance(p=2.0)],
        variance=0.1,
    )
    delays = {r.delay_samples for r in results.values()}
    assert delays == {pair.sdvig}


def test_distance_and_rdft_coexist_in_runner(tiny_config):
    methods = [build_method(k) for k in ("standard", "median", "dist-l1", "dist-cosine")]
    results, _ = _run(tiny_config, methods, variance=0.01)
    assert len(results) == 4
    for r in results.values():
        assert "peaks" in r.extra
        assert 0.0 <= r.normal_rate <= 1.0


def test_mahalanobis_runs_without_crashing(tiny_config):
    # degenerate metric: only require it to produce finite classification stats
    results, _ = _run(tiny_config, [build_method("dist-mahalanobis")], variance=0.5)
    (result,) = results.values()
    assert 0.0 <= result.normal_rate <= 1.0


def test_default_keys_exclude_mahalanobis():
    assert "dist-mahalanobis" not in DEFAULT_KEYS
    assert "dist-l1" in DEFAULT_KEYS and "standard" in DEFAULT_KEYS


def test_dct_wrapped_distance_method(tiny_config):
    method = build_method("dist-l1", with_dct=True)
    results, pair = _run(tiny_config, [method])
    (result,) = results.values()
    assert result.method_name.startswith("DCT + ")
    assert result.delay_samples == pair.sdvig


def test_distance_method_in_sweep(tmp_path):
    from tde_lab.experiments import sas_sweep
    result = sas_sweep.run(
        methods=[build_method("dist-l1"), StandardFFT()],
        signal_config=SignalConfig(frag_length=128, frags=6, tau=0.1, ws=3),
        alpha_values=[2.0],
        gamma_values=[0.0, 0.5],
        realizations=6,
        chunk_size=6,
        cache_dir=str(tmp_path),
        seed=1,
        save=False,
    )
    acc = result.stats["Euclidean (b=1)"][(2.0, 0.0)]
    assert acc.n_total == 6
    assert acc.pabn_percent == 0.0  # noiseless cell


def test_windowed_distance_method_in_sweep(tmp_path):
    # regression: a ±lag_limit window puts peaks in a different index space
    # than the fftshifted MCF — the sweep must classify with the method's
    # own boundaries (validation setup: tau=0, gain1=0.5, lag_limit=100)
    from tde_lab.experiments import sas_sweep
    result = sas_sweep.run(
        methods=[build_method("dist-l1", MethodConfig(lag_limit=30))],
        signal_config=SignalConfig(frag_length=256, frags=6, tau=0.0, ws=3, gain1=0.5),
        alpha_values=[2.0],
        gamma_values=[0.0],
        realizations=6,
        chunk_size=6,
        cache_dir=str(tmp_path),
        seed=1,
        save=False,
    )
    acc = result.stats["Euclidean (b=1)"][(2.0, 0.0)]
    assert acc.pabn_percent == 0.0
