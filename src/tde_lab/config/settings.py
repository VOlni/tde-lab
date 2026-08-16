from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

SPEED_OF_SOUND = 340.0  # m/s
DEFAULT_FRAG_LENGTH = 1024
DEFAULT_FRAGS = 32
DEFAULT_WS = 5           # moving-average window for speech-like signal
DEFAULT_FRAGMENT_DURATION = 0.1  # seconds


@dataclass
class SignalConfig:
    """Parameters for synthetic speech-like signal generation."""
    frag_length: int = DEFAULT_FRAG_LENGTH
    frags: int = DEFAULT_FRAGS
    ws: int = DEFAULT_WS                  # MA window size
    tau: float = 0.4                      # relative delay (fraction of frag_length)
    fragment_duration: float = DEFAULT_FRAGMENT_DURATION  # seconds per fragment
    gain1: float = 1.0                    # scale of the noised reference channel —
                                          # MATLAB w3_05/w3_08 sets used 0.5 / 0.8

    @property
    def sdvig(self) -> int:
        """Delay in samples."""
        return round(self.tau * self.frag_length)

    @property
    def sample_rate(self) -> float:
        return self.frag_length / self.fragment_duration

    @property
    def dt(self) -> float:
        return self.fragment_duration / self.frag_length

    @property
    def time_axis(self):
        import numpy as np
        return np.arange(self.frag_length) * self.dt

    @property
    def lag_axis(self):
        import numpy as np
        return np.linspace(
            -self.fragment_duration / 2,
            self.fragment_duration / 2,
            self.frag_length,
            endpoint=False,
        )


@dataclass
class NoiseConfig:
    """Noise model parameters."""
    kind: str = "gaussian"          # "gaussian" | "sas"
    # Gaussian
    variance: float = 1.0          # noise variance for Gaussian
    # Alpha-stable (SαS)
    alpha: float = 2.0             # stability exponent: (0, 2], 2 = Gaussian
    gamma: float = 1.0             # dispersion (scale)
    delta: float = 0.0             # localization (shift), always 0 for symmetric

    def validate(self) -> None:
        if self.kind not in ("gaussian", "sas"):
            raise ValueError(f"Unknown noise kind: {self.kind!r}")
        if not (0 < self.alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {self.alpha}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")


@dataclass
class AudioConfig:
    """Parameters for WAV-based source."""
    path: str = ""
    frag_length: int = DEFAULT_FRAG_LENGTH
    fragment_duration: float = DEFAULT_FRAGMENT_DURATION
    channel_1: int = 0   # left channel index
    channel_2: int = 1   # right channel index

    @property
    def dt(self) -> float:
        return self.fragment_duration / self.frag_length

    @property
    def lag_axis(self):
        import numpy as np
        return np.linspace(
            -self.fragment_duration / 2,
            self.fragment_duration / 2,
            self.frag_length,
            endpoint=False,
        )


@dataclass
class ExportConfig:
    """Figure export options: formats, resolution, and rc style preset."""
    formats: tuple = ("png",)          # any of: "png", "pdf", "svg"
    dpi: int = 150                     # raster resolution (png); vector formats ignore it
    style: str = "screen"              # rc preset from visualization.style.PRESETS

    def validate(self) -> None:
        allowed = {"png", "pdf", "svg"}
        bad = set(self.formats) - allowed
        if bad:
            raise ValueError(f"Unsupported figure formats: {sorted(bad)} (allowed: {sorted(allowed)})")
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")


@dataclass
class MethodConfig:
    """Per-method optional parameters."""
    trim_percent: float = 25.0    # for AlphaTrimmedRDFT
    dct_beta: float = 2.7         # DCTPreFilter threshold multiplier
    cwmedian_window: int = 5      # CWMedian filter window size
    mic_distance: float = 1.0     # metres, for DOA angle calculation
    # distance-metric methods
    lag_limit: Optional[int] = None   # ± search window in samples (None = all lags;
                                      # MATLAB used 100 — window must contain the true delay)
    normal_halfwidth: int = 5         # classification window around clean-curve argmin


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "experiment"
    signal: SignalConfig = field(default_factory=SignalConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    methods: List[str] = field(default_factory=lambda: ["standard"])
    parallel: bool = False
    save_plots: bool = True
    save_csv: bool = True
    output_dir: str = "output"

    # Sweep parameters (None = no sweep)
    sweep_param: Optional[str] = None        # "variance" | "alpha" | "gamma"
    sweep_values: Optional[List[float]] = None
