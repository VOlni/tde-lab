"""All plotting functions.  Every function returns a matplotlib Figure — never calls plt.show()."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for both CLI and Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tde_lab.methods.base import MCFResult
from tde_lab.visualization.style import line_style


# ── colour palette (consistent across plots) ────────────────────────────────
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# ── 1. MCF plot for a single result ─────────────────────────────────────────

def _curve_ylabel(result: MCFResult) -> str:
    """Distance methods carry S_e(j) curves (argmin), others |MCF| (argmax)."""
    return "S_e(j)" if result.extra.get("curve_kind") == "distance" else "|MCF|"


def plot_mcf(result: MCFResult, true_delay_s: float | None = None) -> plt.Figure:
    """Plot the correlation / distance curve for one method."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.lags * 1e3, np.abs(result.mcf), linewidth=1.5)

    ax.axvline(result.delay_seconds * 1e3, color="tab:red", linestyle="--",
               label=f"Est. delay = {result.delay_seconds*1e3:.3f} ms")
    if true_delay_s is not None:
        ax.axvline(true_delay_s * 1e3, color="tab:green", linestyle=":",
                   label=f"True delay = {true_delay_s*1e3:.3f} ms")

    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel(_curve_ylabel(result))
    ax.set_title(f"MCF — {result.method_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── 2. All methods MCF comparison (grid) ───────────────────────────────────

def plot_mcf_comparison(
    results: Dict[str, MCFResult],
    true_delay_s: float | None = None,
    ncols: int = 3,
) -> plt.Figure:
    """Grid of MCF subplots — one per method."""
    names = list(results.keys())
    n = len(names)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)

    for idx, name in enumerate(names):
        r = results[name]
        ax = axes[idx // ncols][idx % ncols]
        lags_ms = r.lags * 1e3
        ax.plot(lags_ms, np.abs(r.mcf), color=_color(idx), linewidth=1.2)
        ax.axvline(r.delay_seconds * 1e3, color="tab:red", linestyle="--", linewidth=1,
                   label=f"Est {r.delay_seconds*1e3:.2f} ms")
        if true_delay_s is not None:
            ax.axvline(true_delay_s * 1e3, color="tab:green", linestyle=":", linewidth=1,
                       label=f"True {true_delay_s*1e3:.2f} ms")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Lag (ms)", fontsize=8)
        ax.set_ylabel(_curve_ylabel(r), fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # hide empty axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("MCF Comparison — All Methods", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ── 3. Metrics bar chart (normal_rate + MSE) ───────────────────────────────

def plot_metrics_bar(results: Dict[str, MCFResult]) -> plt.Figure:
    """Side-by-side bar chart of normal_rate and MSE for all methods."""
    names = list(results.keys())
    normal_rates = [results[n].normal_rate * 100 for n in names]
    mse_values   = [results[n].mse for n in names]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(names) * 1.2), 5))

    bars1 = ax1.bar(x, normal_rates, color=[_color(i) for i in range(len(names))], edgecolor="k", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Normal estimate rate (%)")
    ax1.set_title("Normal Estimate Rate")
    ax1.set_ylim(0, 110)
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, normal_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(x, mse_values, color=[_color(i) for i in range(len(names))], edgecolor="k", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("MSE (samples²)")
    ax2.set_title("MSE of Normal Estimates")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, mse_values):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


# ── 4. Sweep: normal_rate vs noise parameter ───────────────────────────────

def plot_sweep(
    sweep_values: List[float],
    sweep_results: Dict[str, List[MCFResult]],
    param_name: str = "Noise parameter",
    metric: str = "normal_rate",   # "normal_rate" | "mse" | "delay_s" | "angle"
) -> plt.Figure:
    """
    Line plot of one metric vs a swept parameter, one line per method.

    sweep_results : {method_name: [MCFResult_for_val_0, ..., MCFResult_for_val_N]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    metric_labels = {
        "normal_rate": "Normal estimate rate (%)",
        "mse":         "MSE (samples²)",
        "delay_s":     "Estimated delay (s)",
        "angle":       "Estimated angle (°)",
    }
    ylabel = metric_labels.get(metric, metric)

    for idx, (name, result_list) in enumerate(sweep_results.items()):
        y_vals = []
        for r in result_list:
            if metric == "normal_rate":
                y_vals.append(r.normal_rate * 100)
            elif metric == "mse":
                y_vals.append(r.mse)
            elif metric == "delay_s":
                y_vals.append(r.delay_seconds)
            elif metric == "angle":
                y_vals.append(r.angle_degrees)
        ax.plot(sweep_values, y_vals, marker="o", label=name, color=_color(idx), linewidth=1.5)

    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {param_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── 5. Signal viewer ────────────────────────────────────────────────────────

def plot_signals(
    sig1: np.ndarray,
    sig2: np.ndarray,
    time_axis: np.ndarray,
    frag_idx: int = 0,
    title: str = "Input signals (fragment 0)",
) -> plt.Figure:
    """Plot one fragment from both channels."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(time_axis * 1e3, sig1[:, frag_idx], linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Channel 1 (mic 1)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis * 1e3, sig2[:, frag_idx], linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("Channel 2 (mic 2)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ── 6. Spectrum viewer ──────────────────────────────────────────────────────

def plot_spectrum(
    sig: np.ndarray,
    sample_rate: float,
    frag_idx: int = 0,
    title: str = "Spectrum (fragment 0)",
) -> plt.Figure:
    """Magnitude spectrum of one fragment."""
    frag = sig[:, frag_idx] if sig.ndim == 2 else sig
    n = len(frag)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag   = np.abs(np.fft.rfft(frag))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, 20 * np.log10(mag + 1e-12), linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── 7. Metric vs gamma for one alpha (MATLAB comparison_sas_*.m replica) ────

def plot_metric_vs_gamma(
    gamma_values: List[float],
    curves: Dict[str, np.ndarray],
    alpha: float,
    ylabel: str = "Pabn",
    title: str | None = None,
) -> plt.Figure:
    """
    One figure per alpha: metric vs gamma, one line per method.

    Replicates the comparison_sas_w3_05.m layout: distinct line style +
    marker per curve, legend top-left, grid on.

    curves : {method_name: array of metric values, same length as gamma_values}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, (name, values) in enumerate(curves.items()):
        ax.plot(gamma_values, values, label=name, linewidth=2, **line_style(idx))

    ax.set_xlabel("gamma")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel}, alpha = {alpha:g}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_pabn_vs_gamma(
    gamma_values: List[float],
    curves: Dict[str, np.ndarray],
    alpha: float,
) -> plt.Figure:
    """Probability of abnormal estimations (%) vs gamma for one alpha."""
    return plot_metric_vs_gamma(
        gamma_values, curves, alpha, ylabel="Pabn",
        title=f"Probability of Abnormal Estimations, alpha = {alpha:g}",
    )


def plot_sigma_vs_gamma(
    gamma_values: List[float],
    curves: Dict[str, np.ndarray],
    alpha: float,
) -> plt.Figure:
    """RMSE of normal estimates (samples) vs gamma for one alpha."""
    return plot_metric_vs_gamma(
        gamma_values, curves, alpha, ylabel="sigma (samples)",
        title=f"RMSE of Normal Estimations, alpha = {alpha:g}",
    )


# ── 8. SaS sweep — alpha × gamma heatmap summary ──────────────────────────

def plot_sas_sweep_lines(
    alpha_values: List[float],
    gamma_values: List[float],
    # nested: {method_name: {alpha: {gamma: MCFResult}}}
    sweep_data: Dict[str, Dict[float, Dict[float, MCFResult]]],
    metric: str = "normal_rate",
) -> plt.Figure:
    """
    For each method: one subplot with lines per gamma value, x-axis = alpha.
    """
    method_names = list(sweep_data.keys())
    n = len(method_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    metric_label = {
        "normal_rate": "Normal rate (%)",
        "mse": "MSE (samples²)",
    }.get(metric, metric)

    for midx, method_name in enumerate(method_names):
        ax = axes[midx // ncols][midx % ncols]
        data = sweep_data[method_name]

        for gidx, g in enumerate(gamma_values):
            y_vals = []
            for a in alpha_values:
                r = data.get(a, {}).get(g)
                if r is None:
                    y_vals.append(np.nan)
                elif metric == "normal_rate":
                    y_vals.append(r.normal_rate * 100)
                else:
                    y_vals.append(r.mse)
            ax.plot(alpha_values, y_vals, marker="o", label=f"γ={g}", color=_color(gidx), linewidth=1.2)

        ax.set_xlabel("α (stability)")
        ax.set_ylabel(metric_label)
        ax.set_title(method_name, fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"SαS Sweep — {metric_label}", fontsize=12)
    fig.tight_layout()
    return fig
