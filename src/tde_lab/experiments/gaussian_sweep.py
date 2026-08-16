"""Gaussian noise sweep experiment.

Sweeps noise variance values, runs all selected methods at each point,
produces line plots of normal_rate and MSE vs noise variance.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from tde_lab.config.settings import SignalConfig, NoiseConfig, ExperimentConfig, ExportConfig
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import GaussianNoise
from tde_lab.methods.base import BaseMethod, MCFResult
from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.visualization.plots import plot_sweep, plot_mcf_comparison, plot_metrics_bar
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import figure_style


DEFAULT_VARIANCE_VALUES = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 5.0]


def run(
    methods: List[BaseMethod],
    signal_config: SignalConfig | None = None,
    variance_values: List[float] | None = None,
    save: bool = True,
    output_dir: str = "output",
    parallel: bool = False,
    export: ExportConfig | None = None,
    progress_cb=None,
) -> Dict[str, List[MCFResult]]:
    """
    Run all methods across a sweep of Gaussian noise variances.

    Returns
    -------
    {method_name: [MCFResult_at_var_0, ..., MCFResult_at_var_N]}
    """
    sig_cfg = signal_config or SignalConfig()
    var_values = variance_values if variance_values is not None else DEFAULT_VARIANCE_VALUES

    # results[method_name][variance_index] = MCFResult
    sweep_results: Dict[str, List[MCFResult]] = {m.name: [] for m in methods}
    last_results: Dict[str, MCFResult] = {}

    for var in var_values:
        noise_cfg = NoiseConfig(kind="gaussian", variance=var)
        noise = GaussianNoise(variance=var)
        gen = SpeechLikeGenerator(sig_cfg)
        pair = gen.generate(noise, noise)

        runner = ExperimentRunner(
            sig1=pair.sig1,
            sig2=pair.sig2,
            clean=pair.clean,
            lags=sig_cfg.lag_axis,
            sdvig=pair.sdvig,
            mic_distance=1.0,
            parallel=parallel,
            progress_cb=progress_cb,
        )
        step_results = runner.run(methods)
        last_results = step_results

        for m in methods:
            sweep_results[m.name].append(step_results[m.name])

    if not save:
        return sweep_results

    export = export or ExportConfig()
    saver = ResultSaver(output_dir, "gaussian_sweep", export)

    with figure_style(export.style):
        # line plots
        fig_nr = plot_sweep(var_values, sweep_results, "Noise variance (σ²)", "normal_rate")
        saver.save_figure(fig_nr, "normal_rate_vs_variance")

        fig_mse = plot_sweep(var_values, sweep_results, "Noise variance (σ²)", "mse")
        saver.save_figure(fig_mse, "mse_vs_variance")

        # MCF comparison at last (highest) noise level
        fig_mcf = plot_mcf_comparison(last_results, true_delay_s=sig_cfg.sdvig * sig_cfg.dt)
        saver.save_figure(fig_mcf, "mcf_comparison_max_noise")

        # bar chart at last point
        fig_bar = plot_metrics_bar(last_results)
        saver.save_figure(fig_bar, "metrics_bar_max_noise")

    saver.save_sweep_csv(var_values, sweep_results, "variance")
    print(f"[gaussian_sweep] Results saved to: {saver.run_dir}")

    return sweep_results
