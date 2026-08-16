import numpy as np
import pytest

from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.methods.standard_fft import SubSampleFFT
from tde_lab.methods.subsample import fractional_delay, subsample_delay
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import GaussianNoise


def _bandlimited_signal(n=512, keep=0.3):
    """Random band-limited real signal — fractional delays are well defined."""
    spectrum = np.zeros(n, dtype=complex)
    n_keep = int(n * keep / 2)
    spectrum[1:n_keep] = np.random.randn(n_keep - 1) + 1j * np.random.randn(n_keep - 1)
    spectrum[-n_keep + 1:] = np.conj(spectrum[1:n_keep][::-1])
    return np.fft.ifft(spectrum).real


def test_fractional_delay_integer_matches_roll():
    x = np.random.randn(128)
    for d in (0, 1, 26, -5):
        np.testing.assert_allclose(fractional_delay(x, d), np.roll(x, d), atol=1e-12)


def test_integer_delay_recovered_exactly():
    ref = _bandlimited_signal()
    for d in (0, 7, -12):
        assert subsample_delay(np.roll(ref, d), ref) == pytest.approx(d, abs=1e-9)


@pytest.mark.parametrize("delay", [0.3, -2.7, 17.25])
def test_fractional_delay_recovered(delay):
    ref = _bandlimited_signal()
    signal = fractional_delay(ref, delay)
    # parabolic interpolation: a few hundredths of a sample on band-limited data
    assert subsample_delay(signal, ref) == pytest.approx(delay, abs=0.1)


def test_inverted_signal_handled():
    ref = _bandlimited_signal()
    signal = -fractional_delay(ref, 4.5)
    assert subsample_delay(signal, ref) == pytest.approx(4.5, abs=0.1)


def test_wraparound_delay_reported_negative():
    n = 512
    ref = _bandlimited_signal(n)
    signal = fractional_delay(ref, n - 10.5)     # equivalent to -10.5 on a cycle
    assert subsample_delay(signal, ref) == pytest.approx(-10.5, abs=0.1)


def test_noisy_delay_within_tolerance():
    ref = _bandlimited_signal()
    signal = fractional_delay(ref, 5.2) + 0.05 * np.random.randn(len(ref))
    assert subsample_delay(signal, ref) == pytest.approx(5.2, abs=0.2)


def test_subsample_method_integration(small_config):
    # the framework's integer cyclic delay is recovered with sub-sample output
    gen = SpeechLikeGenerator(small_config)
    pair = gen.generate(GaussianNoise(0.0), GaussianNoise(0.0))
    runner = ExperimentRunner(
        sig1=pair.sig1, sig2=pair.sig2, clean=pair.clean,
        lags=small_config.lag_axis, sdvig=pair.sdvig,
    )
    result = runner.run([SubSampleFFT()])["Sub-sample FFT"]

    assert result.extra["fractional_delay"] == pytest.approx(pair.sdvig, abs=1e-6)
    assert result.normal_rate == 1.0
    assert len(result.extra["fractional_delays"]) == small_config.frags
