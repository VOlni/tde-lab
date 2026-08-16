import numpy as np
import pytest

from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.methods import ALL_METHODS, build_method
from tde_lab.methods.standard_fft import StandardFFT
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import GaussianNoise


def _make_runner(config, variance=0.0, **kwargs):
    gen = SpeechLikeGenerator(config)
    pair = gen.generate(GaussianNoise(variance), GaussianNoise(variance))
    runner = ExperimentRunner(
        sig1=pair.sig1, sig2=pair.sig2, clean=pair.clean,
        lags=config.lag_axis, sdvig=pair.sdvig, **kwargs,
    )
    return runner, pair


def test_standard_fft_recovers_delay_noiseless(small_config):
    runner, pair = _make_runner(small_config)
    result = runner.run([StandardFFT()])["Standard FFT"]

    assert result.delay_samples == pair.sdvig
    assert result.normal_rate == 1.0
    assert result.mse == 0.0
    assert result.delay_seconds == pytest.approx(pair.sdvig * small_config.dt)


def test_standard_fft_survives_mild_noise(small_config):
    runner, pair = _make_runner(small_config, variance=0.25)
    result = runner.run([StandardFFT()])["Standard FFT"]
    assert result.delay_samples == pair.sdvig
    assert result.normal_rate > 0.9


def test_all_registered_methods_run(tiny_config):
    runner, pair = _make_runner(tiny_config, variance=0.01)
    methods = [build_method(key) for key in ALL_METHODS]
    results = runner.run(methods)

    assert len(results) == len(ALL_METHODS)
    for name, r in results.items():
        assert r.mcf.shape == (tiny_config.frag_length,)
        assert 0.0 <= r.normal_rate <= 1.0
        assert np.isfinite(r.delay_seconds), name


def test_dct_wrapped_method_runs(tiny_config):
    runner, pair = _make_runner(tiny_config, variance=0.01)
    method = build_method("standard", with_dct=True)
    results = runner.run([method])
    (result,) = results.values()
    assert result.mcf.shape == (tiny_config.frag_length,)
