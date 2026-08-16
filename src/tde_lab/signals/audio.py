"""WAV loader — treats two selected channels as the two microphone signals."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from tde_lab.config.settings import AudioConfig


@dataclass
class AudioPair:
    """Fragmented stereo audio ready for TDE analysis."""
    sig1: np.ndarray        # shape: (frag_length, n_frags)
    sig2: np.ndarray        # shape: (frag_length, n_frags)
    sample_rate: int
    n_frags: int
    path: str
    raw1: np.ndarray = field(default=None, repr=False)   # unfragmented segment
    raw2: np.ndarray = field(default=None, repr=False)
    config: AudioConfig | None = None

    @property
    def frag_length(self) -> int:
        return self.sig1.shape[0]

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def lag_axis(self) -> np.ndarray:
        """Lag axis in seconds, derived from the file's real sample rate."""
        fl = self.frag_length
        return (np.arange(fl) - fl // 2) * self.dt

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.frag_length) * self.dt


def load_wav_pair(
    path: str,
    *,
    frag_length: int = 1024,
    channels: tuple[int, int] = (0, 1),
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> AudioPair:
    """
    Load two channels of a WAV file, cut the [start_s, start_s + duration_s]
    segment, and split it into non-overlapping fragments of frag_length.
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for WAV loading. "
            "Install it with: pip install soundfile"
        )

    target = str(Path(path).expanduser().resolve())
    data, sr = sf.read(target, always_2d=True)

    ch_a, ch_b = channels
    if data.shape[1] <= max(ch_a, ch_b):
        raise ValueError(
            f"Requested channels {channels}, but the file has only "
            f"{data.shape[1]} channel(s). A two-microphone recording is required."
        )

    start = int(round(start_s * sr))
    stop = len(data) if duration_s is None else start + int(round(duration_s * sr))
    if start >= len(data):
        raise ValueError(f"start_s={start_s}s is beyond the file end "
                         f"({len(data) / sr:.2f}s).")
    segment = data[start:stop]

    ch1 = segment[:, ch_a].astype(np.float64)
    ch2 = segment[:, ch_b].astype(np.float64)

    n_frags = len(ch1) // frag_length
    if n_frags == 0:
        raise ValueError(
            f"Selected segment too short ({len(ch1)} samples) for "
            f"frag_length={frag_length}."
        )

    trimmed1 = ch1[: n_frags * frag_length]
    trimmed2 = ch2[: n_frags * frag_length]

    return AudioPair(
        sig1=trimmed1.reshape(frag_length, n_frags, order="F"),
        sig2=trimmed2.reshape(frag_length, n_frags, order="F"),
        sample_rate=sr,
        n_frags=n_frags,
        path=target,
        raw1=trimmed1,
        raw2=trimmed2,
    )


class WAVLoader:
    """
    Config-object interface kept for backward compatibility; delegates to
    load_wav_pair().
    """

    def __init__(self, config: AudioConfig):
        self.config = config

    def load(self, path: str | None = None) -> AudioPair:
        target = path or self.config.path
        if not target:
            raise ValueError("No WAV path provided.")
        pair = load_wav_pair(
            target,
            frag_length=self.config.frag_length,
            channels=(self.config.channel_1, self.config.channel_2),
        )
        pair.config = self.config
        return pair
