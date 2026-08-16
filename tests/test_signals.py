import numpy as np
import pytest

from tde_lab.config.settings import NoiseConfig
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import AlphaStableNoise, GaussianNoise, make_noise


def test_generator_shapes_and_sdvig(small_config):
    gen = SpeechLikeGenerator(small_config)
    pair = gen.generate(GaussianNoise(1.0), GaussianNoise(1.0))

    expected = (small_config.frag_length, small_config.frags)
    assert pair.sig1.shape == expected
    assert pair.sig2.shape == expected
    assert pair.clean.shape == expected
    assert pair.sdvig == round(small_config.tau * small_config.frag_length)


def test_cyclic_shift_matches_np_roll():
    x = np.random.randn(128, 4)
    for sdvig in (0, 1, 26, 127):
        shifted = SpeechLikeGenerator._cyclic_shift(x, sdvig)
        np.testing.assert_allclose(shifted, np.roll(x, sdvig, axis=0))


def test_noiseless_sig2_is_rolled_clean(small_config):
    gen = SpeechLikeGenerator(small_config)
    pair = gen.generate(GaussianNoise(0.0), GaussianNoise(0.0))

    np.testing.assert_allclose(pair.sig1, pair.clean)
    np.testing.assert_allclose(pair.sig2, np.roll(pair.clean, pair.sdvig, axis=0))
    assert pair.snr == float("inf")


def test_clean_signal_unit_variance(small_config):
    gen = SpeechLikeGenerator(small_config)
    pair = gen.generate(GaussianNoise(0.0), GaussianNoise(0.0))
    assert np.var(pair.clean) == pytest.approx(1.0, rel=0.05)


def test_sas_alpha2_is_scaled_gaussian():
    # At alpha=2 the CMS algorithm reduces to sqrt(2)*randn scaled by gamma^(1/2):
    # variance = 2 * gamma
    for gamma in (1.0, 3.0):
        samples = AlphaStableNoise(alpha=2.0, gamma=gamma).sample((200_000,))
        assert np.var(samples) == pytest.approx(2.0 * gamma, rel=0.05)


def test_sas_heavy_tails():
    # Lower alpha must produce heavier tails (larger extreme quantile ratio)
    n = 200_000
    q_gauss = np.quantile(np.abs(AlphaStableNoise(2.0, 1.0).sample((n,))), 0.999)
    q_heavy = np.quantile(np.abs(AlphaStableNoise(1.2, 1.0).sample((n,))), 0.999)
    assert q_heavy > 3 * q_gauss


def test_make_noise_factory():
    assert isinstance(make_noise(NoiseConfig(kind="gaussian", variance=2.0)), GaussianNoise)
    sas = make_noise(NoiseConfig(kind="sas", alpha=1.5, gamma=2.0))
    assert isinstance(sas, AlphaStableNoise)
    assert sas.alpha == 1.5


def test_invalid_noise_config_rejected():
    with pytest.raises(ValueError):
        make_noise(NoiseConfig(kind="sas", alpha=2.5))
    with pytest.raises(ValueError):
        make_noise(NoiseConfig(kind="unknown"))
