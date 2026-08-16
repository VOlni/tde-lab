"""Synthetic speech-like signal generator."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from tde_lab.config.settings import SignalConfig
from tde_lab.signals.noise import BaseNoise


@dataclass
class SignalPair:
    """Two noisy fragments (frag_length × frags) and the clean reference."""
    sig1: np.ndarray        # shape: (frag_length, frags)
    sig2: np.ndarray        # shape: (frag_length, frags), delayed by sdvig samples
    clean: np.ndarray       # shape: (frag_length, frags), noise-free
    snr: float
    sdvig: int              # actual delay in samples
    config: SignalConfig


class SpeechLikeGenerator:
    """
    Generates a speech-like random process as the output of a moving-average
    (MA) filter driven by white Gaussian noise, then adds independent noise
    to produce two shifted microphone channels.

    Replicates the MATLAB making_signals_gaus / making_signals_SaS logic.
    """

    def __init__(self, config: SignalConfig):
        self.config = config

    def generate(self, noise1: BaseNoise, noise2: BaseNoise) -> SignalPair:
        cfg = self.config
        n_total = cfg.frag_length * cfg.frags + cfg.ws - 1

        # --- clean speech-like signal via MA filter ---
        raw = np.random.randn(n_total)
        sigf = np.array(
            [raw[i : i + cfg.ws].mean() for i in range(n_total - cfg.ws + 1)]
        )
        # normalise to unit variance
        sigf /= np.sqrt(np.var(sigf))

        # reshape into (frag_length, frags) columns
        clean = sigf[: cfg.frag_length * cfg.frags].reshape(cfg.frag_length, cfg.frags, order="F")

        # --- additive noise (independent per channel) ---
        n1 = noise1.sample((cfg.frag_length, cfg.frags))
        n2 = noise2.sample((cfg.frag_length, cfg.frags))

        # gain1 replicates the MATLAB '(sigf + noise) * 0.5' reference-channel
        # scaling; the delayed channel is never scaled
        sig1 = (clean + n1) * cfg.gain1
        noisy2 = clean + n2

        # --- cyclic shift of signal 2 by sdvig samples ---
        sig2 = self._cyclic_shift(noisy2, cfg.sdvig)

        snr = float(np.mean(np.var(clean, axis=0) / np.var(n1, axis=0))) if np.any(n1 != 0) else float("inf")

        return SignalPair(sig1=sig1, sig2=sig2, clean=clean, snr=snr,
                         sdvig=cfg.sdvig, config=cfg)

    @staticmethod
    def _cyclic_shift(signal: np.ndarray, sdvig: int) -> np.ndarray:
        """Cyclic (circular) shift along axis=0 by sdvig samples."""
        if sdvig == 0:
            return signal.copy()
        out = np.empty_like(signal)
        out[:sdvig, :] = signal[-sdvig:, :]
        out[sdvig:, :] = signal[:-sdvig, :]
        return out
