"""Wavelet-based voice activity detection.

Port of VAD_wavelet/{method1,method2,run_length}.m:

- method1: db3 DWT cascade; a frame is voiced when the A3/A4 subband powers
  exceed an adaptive noise estimate (alpha=beta=1.8) that is re-learned from
  every unvoiced frame.  The first n_init frames are assumed to be noise.
- method2: db4 subband energies (D1..D4 + A4) with silence, stationarity and
  background-level flags plus burst/hangover smoothing.  Its T1..T4 thresholds
  are absolute (signal-scale dependent) — expose them via VADConfig.
- run_length: 8-state run-length smoother that removes short bursts and
  bridges short gaps, with the MATLAB retroactive y[i-1], y[i-2] patches.

MATLAB quirks preserved: frames are M+1 samples (x(k:k+M) inclusive), the
method2 background update iterates i = 2 and 4 only (`for i=[2,L]`), and its
difference measure is |sum(dE)|/sqrt(L).  pywt's default symmetric padding
approximates MATLAB's 'sym' extension; energies are threshold-relative so the
tiny boundary differences don't change decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

_MIN_FRAME = 16     # frames shorter than this are treated as unvoiced tails


@dataclass
class VADConfig:
    method: str = "method1"        # "method1" | "method2"
    frame_ms: float = 30.0
    smooth: bool = True            # run-length post-smoothing
    # method1
    alpha: float = 1.8
    beta: float = 1.8
    n_init: int = 4
    # method2
    t1: float = 1e-10              # silence
    t2: float = 1e-4               # stationarity
    t3: float = 1e-5               # subframe background level
    t4: float = 1e-5               # background level
    burst_const: int = 3
    hang_const: int = 5


def _cascade(frame: np.ndarray, wavelet: str, levels: int = 4):
    """DWT cascade on successive approximations → ([A1..A4], [D1..D4])."""
    approxs, details = [], []
    a = frame
    for _ in range(levels):
        a, d = pywt.dwt(a, wavelet)
        approxs.append(a)
        details.append(d)
    return approxs, details


def vad_method1(
    x: np.ndarray,
    frame_len: int,
    *,
    alpha: float = 1.8,
    beta: float = 1.8,
    n_init: int = 4,
    wavelet: str = "db3",
) -> np.ndarray:
    """Adaptive A3/A4-power detector.  Returns a per-frame bool array."""
    x = np.asarray(x, dtype=float)
    m = frame_len
    length = len(x)

    # noise estimate from the first n_init frames
    dw3 = dw4 = 0.0
    k = 0
    while k < n_init * m:
        approxs, _ = _cascade(x[k:k + m + 1], wavelet)
        dw3 += np.mean(approxs[2] ** 2) / n_init
        dw4 += np.mean(approxs[3] ** 2) / n_init
        k += m

    flags = [False] * n_init
    while k < length:
        frame = x[k:min(length, k + m + 1)]
        k += m
        if len(frame) < _MIN_FRAME:
            flags.append(False)
            continue
        approxs, _ = _cascade(frame, wavelet)
        dy3 = float(np.mean(approxs[2] ** 2))
        dy4 = float(np.mean(approxs[3] ** 2))
        voiced = (dy3 + dy4) > (alpha * dw3 + beta * dw4)
        flags.append(bool(voiced))
        if not voiced:                     # re-learn the noise level
            dw3, dw4 = dy3, dy4
    return np.array(flags, dtype=bool)


def vad_method2(
    x: np.ndarray,
    frame_len: int,
    *,
    wavelet: str = "db4",
    t1: float = 1e-10,
    t2: float = 1e-4,
    t3: float = 1e-5,
    t4: float = 1e-5,
    alpha: float = 0.5,
    burst_const: int = 3,
    hang_const: int = 5,
) -> np.ndarray:
    """Subband-energy detector with hangover.  Returns a per-frame bool array."""
    x = np.asarray(x, dtype=float)
    m = frame_len
    levels = 4
    half = m // 2                          # P = 2 subframes

    e_pre = np.zeros(levels + 1)
    b = np.zeros(levels + 1)
    b_pre = np.zeros(levels + 1)
    delta_pre = 0.0
    burstcount = 0
    hangcount = -1

    flags = [False]                        # MATLAB seeds VAD with one zero
    k = 0
    length = len(x)
    while k + m < length:
        frame = x[k:k + m + 1]
        approxs, details = _cascade(frame, wavelet, levels)

        e = np.empty(levels + 1)
        for i in range(levels):
            e[i] = float(np.sum(details[i] ** 2))
        e[levels] = float(np.sum(approxs[levels - 1] ** 2))

        f_sil = e.sum() < t1
        # literal MATLAB: sqrt(sum(dE)^2 / L) = |sum(dE)| / sqrt(L)
        delta = float(np.sqrt(np.sum(e[:levels] - e_pre[:levels]) ** 2 / levels))
        f_stat = (delta < t2) and (delta_pre < t2)

        for i in (1, 3):                   # MATLAB `for i=[2,L]` — bins 2 and 4 only
            if b_pre[i] > e[i]:
                b[i] = e[i]
            else:
                b[i] = alpha * b_pre[i] + (1 - alpha) * e[i]

        a1_1, _ = pywt.dwt(x[k:k + half + 1], wavelet)
        _, d2_1 = pywt.dwt(a1_1, wavelet)
        a1_2, _ = pywt.dwt(x[k + half:k + m + 1], wavelet)
        _, d2_2 = pywt.dwt(a1_2, wavelet)
        e2_1 = float(np.sum(d2_1 ** 2))
        e2_2 = float(np.sum(d2_2 ** 2))
        f_b2 = (e2_1 - b[1] < t3) and (e2_2 - b[1] < t3)
        f_bl = (e[levels - 1] - b[levels - 1]) < t4

        voiced = not (f_sil or (f_b2 and f_bl and f_stat))

        # hangover
        burstcount = burstcount + 1 if voiced else 0
        if burstcount > burst_const:
            hangcount = hang_const
            burstcount = burst_const
        flags.append(bool(voiced or hangcount >= 0))
        if hangcount >= 0:
            hangcount -= 1

        e_pre = e.copy()
        delta_pre = delta
        b_pre = b.copy()
        k += m
    return np.array(flags, dtype=bool)


def run_length_smooth(flags: np.ndarray, n_state: int = 8) -> np.ndarray:
    """8-state run-length smoother — literal port incl. retroactive patches.

    Bursts shorter than n_state/2 frames are suppressed; gaps shorter than
    n_state/2 frames inside speech are bridged (and un-bridged retroactively
    when the gap turns out to end the speech run).
    """
    x = np.asarray(flags).astype(int)
    y = np.zeros(len(x), dtype=int)
    half = n_state // 2
    state = 1
    for i in range(len(x)):
        if 1 <= state <= half - 1:
            if x[i] == 1:
                state += 1
            else:
                state = 1
            y[i] = 0
        elif state == half:
            if x[i] == 1:
                state += 1
                y[i] = 1
                y[i - 1] = 1
                y[i - 2] = 1
            else:
                state = 1
                y[i] = 0
        elif half + 1 <= state <= n_state - 1:
            if x[i] == 0:
                state += 1
            else:
                state = half + 1
            y[i] = 1
        else:                              # state == n_state
            if x[i] == 0:
                state = 1
                y[i] = 0
                y[i - 1] = 0
                y[i - 2] = 0
            else:
                state = half + 1
                y[i] = 1
    return y.astype(bool)


def frame_mask_to_samples(mask: np.ndarray, frame_len: int, n_samples: int) -> np.ndarray:
    """Expand a per-frame mask to per-sample (frames tile the signal with
    stride frame_len; the tail keeps the last frame's decision)."""
    samples = np.repeat(np.asarray(mask, dtype=bool), frame_len)
    if len(samples) >= n_samples:
        return samples[:n_samples]
    pad_value = bool(mask[-1]) if len(mask) else False
    return np.concatenate([samples, np.full(n_samples - len(samples), pad_value)])


def detect_voice(x: np.ndarray, fs: float, config: VADConfig | None = None) -> np.ndarray:
    """Sample-level voice mask for a mono signal."""
    cfg = config or VADConfig()
    frame_len = max(int(round(cfg.frame_ms / 1000.0 * fs)), _MIN_FRAME)

    if cfg.method == "method1":
        flags = vad_method1(x, frame_len, alpha=cfg.alpha, beta=cfg.beta,
                            n_init=cfg.n_init)
    elif cfg.method == "method2":
        flags = vad_method2(x, frame_len, t1=cfg.t1, t2=cfg.t2, t3=cfg.t3,
                            t4=cfg.t4, burst_const=cfg.burst_const,
                            hang_const=cfg.hang_const)
    else:
        raise ValueError(f"Unknown VAD method {cfg.method!r}")

    if cfg.smooth:
        flags = run_length_smooth(flags)
    return frame_mask_to_samples(flags, frame_len, len(x))


def fragment_voice_mask(
    x: np.ndarray,
    fs: float,
    frag_length: int,
    *,
    config: VADConfig | None = None,
    min_activity: float = 0.5,
) -> np.ndarray:
    """Per-fragment bool mask: fragment kept when its voiced-sample fraction
    reaches min_activity.  x is the unfragmented signal (n_frags·frag_length)."""
    sample_mask = detect_voice(x, fs, config)
    n_frags = len(x) // frag_length
    frags = sample_mask[: n_frags * frag_length].reshape(frag_length, n_frags, order="F")
    return frags.mean(axis=0) >= min_activity
