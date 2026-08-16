import numpy as np
import pytest

from tde_lab.config.settings import SignalConfig


@pytest.fixture(autouse=True)
def seeded_rng():
    """The codebase uses the legacy global NumPy RNG; make every test deterministic."""
    np.random.seed(1234)


@pytest.fixture
def small_config() -> SignalConfig:
    return SignalConfig(frag_length=256, frags=8, tau=0.1, ws=5)


@pytest.fixture
def tiny_config() -> SignalConfig:
    """Small enough for the O(N²)-per-bin robust methods to run fast."""
    return SignalConfig(frag_length=64, frags=2, tau=0.1, ws=3)
