"""Matplotlib style presets: interactive screen output vs paper-quality export."""
from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
from cycler import cycler

_PAPER_COMMON = {
    "savefig.dpi": 300,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 7.0,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "figure.autolayout": True,
}

PRESETS: dict[str, dict] = {
    # matplotlib defaults — what the GUI shows interactively
    "screen": {},
    # journal figure: 300 dpi, serif, MATLAB-like weight (set(gca,'FontSize',16) feel)
    "paper": dict(_PAPER_COMMON),
    # same, but safe for grayscale print: luminance-spaced colors; combine with
    # the LINE_STYLES cycle so curves stay distinguishable without color
    "paper_gray": {
        **_PAPER_COMMON,
        "axes.prop_cycle": cycler(color=["0.0", "0.30", "0.45", "0.60", "0.75"]),
    },
}

# (linestyle, marker) cycle mirroring comparison_sas_w3_05.m:
# '-', '--+', ':o', '-.x', '-.+', '-.s', '-.o', ':x', ':+', ':s', ':d'
LINE_STYLES: list[tuple[str, str | None]] = [
    ("-", None),
    ("--", "+"),
    (":", "o"),
    ("-.", "x"),
    ("-.", "+"),
    ("-.", "s"),
    ("-.", "o"),
    (":", "x"),
    (":", "+"),
    (":", "s"),
    (":", "d"),
]


def line_style(index: int) -> dict:
    """Kwargs for the i-th curve of a multi-method comparison plot."""
    ls, marker = LINE_STYLES[index % len(LINE_STYLES)]
    style: dict = {"linestyle": ls}
    if marker is not None:
        style["marker"] = marker
        style["markevery"] = 1
    return style


@contextmanager
def figure_style(preset: str = "screen"):
    """Temporarily apply a named rc preset while figures are created."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown style preset {preset!r}. Choose from: {list(PRESETS)}")
    with mpl.rc_context(PRESETS[preset]):
        yield
