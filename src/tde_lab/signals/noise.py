"""Noise generators: Gaussian and symmetric alpha-stable (SαS)."""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from tde_lab.config.settings import NoiseConfig


class BaseNoise(ABC):
    @abstractmethod
    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        """Return noise samples of the given shape."""

    @property
    @abstractmethod
    def name(self) -> str: ...


class GaussianNoise(BaseNoise):
    """Additive white Gaussian noise with specified variance."""

    def __init__(self, variance: float = 1.0):
        if variance < 0:
            raise ValueError("variance must be >= 0")
        self.variance = variance

    @property
    def name(self) -> str:
        return f"Gaussian(σ²={self.variance})"

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        return np.random.randn(*size) * np.sqrt(self.variance)

    def snr(self, signal_variance: float) -> float:
        if self.variance == 0:
            return float("inf")
        return signal_variance / self.variance


class AlphaStableNoise(BaseNoise):
    """
    Symmetric alpha-stable (SαS) noise via the Chambers-Mallows-Stuck method.

    Reference:
        Kuruoglu, E.E. (1998). Signal Processing in alpha-stable environments.
        PhD Thesis, University of Cambridge.
    """

    def __init__(self, alpha: float = 1.5, gamma: float = 1.0, delta: float = 0.0):
        if not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta

    @property
    def name(self) -> str:
        return f"SαS(α={self.alpha}, γ={self.gamma})"

    def sample(self, size: tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(size))
        noise = self._cms(n)
        return noise.reshape(size)

    def _cms(self, n: int) -> np.ndarray:
        """Chambers-Mallows-Stuck algorithm."""
        alpha = self.alpha
        if alpha == 2:
            # Gaussian special case
            raw = np.sqrt(2) * np.random.randn(n)
        elif alpha == 1:
            # Cauchy special case
            u = np.pi * (np.random.rand(n) - 0.5)
            raw = np.tan(u)
        else:
            u = np.pi * (np.random.rand(n) - 0.5)
            w = -np.log(np.random.rand(n))  # Exp(1)
            raw = (
                np.sin(alpha * u)
                / (np.cos(u) ** (1.0 / alpha))
                * (np.cos((1 - alpha) * u) / w) ** ((1 - alpha) / alpha)
            )

        return self.gamma ** (1.0 / self.alpha) * raw + self.delta


def make_noise(config: NoiseConfig) -> BaseNoise:
    """Factory: build the right noise object from a NoiseConfig."""
    config.validate()
    if config.kind == "gaussian":
        return GaussianNoise(variance=config.variance)
    return AlphaStableNoise(alpha=config.alpha, gamma=config.gamma, delta=config.delta)
