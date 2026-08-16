"""Single-shot method comparison experiment.

Generates one signal pair with fixed noise parameters and runs all
selected methods, producing MCF + metrics plots side by side.
Also supports WAV input.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from tde_lab.config.settings import SignalConfig, NoiseConfig, MethodConfig, ExportConfig
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import make_noise
from tde_lab.signals.audio import WAVLoader, AudioConfig
from tde_lab.methods.base import BaseMethod, MCFResult
from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.visualization.plots import plot_mcf_comparison, plot_metrics_bar, plot_signals, plot_spectrum
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import figure_style


def run(
    methods: List[BaseMethod],
    signal_config: SignalConfig | None = None,
    noise_config: NoiseConfig | None = None,
    wav_path: Optional[str] = None,
    save: bool = True,
    output_dir: str = "output",
    parallel: bool = False,
    mic_distance: float = 1.0,
    export: ExportConfig | None = None,
    progress_cb=None,
) -> Dict[str, MCFResult]:
    """
    Run all methods on a single signal pair and return results.

    If wav_path is provided the WAV file is used as input (noise_config ignored).
    """
    if wav_path:
        sig1, sig2, lags, sr = _load_wav(wav_path, signal_config)
        true_delay_s = None
        clean = sig1  # no clean reference from WAV
    else:
        sig_cfg = signal_config or SignalConfig()
        n_cfg = noise_config or NoiseConfig()
        noise = make_noise(n_cfg)
        gen = SpeechLikeGenerator(sig_cfg)
        pair = gen.generate(noise, noise)
        sig1, sig2 = pair.sig1, pair.sig2
        clean = pair.clean
        lags = sig_cfg.lag_axis
        sr = sig_cfg.sample_rate
        true_delay_s = pair.sdvig * sig_cfg.dt

    sdvig = pair.sdvig if not wav_path else 0
    runner = ExperimentRunner(
        sig1=sig1,
        sig2=sig2,
        clean=clean,
        lags=lags,
        sdvig=sdvig,
        mic_distance=mic_distance,
        parallel=parallel,
        progress_cb=progress_cb,
    )
    results = runner.run(methods)

    if not save:
        return results

    exp_name = "wav_comparison" if wav_path else "comparison"
    export = export or ExportConfig()
    saver = ResultSaver(output_dir, exp_name, export)

    with figure_style(export.style):
        # signals (first fragment)
        time_axis = np.arange(sig1.shape[0]) * (lags[1] - lags[0]) if len(lags) > 1 else np.arange(sig1.shape[0])
        fig_sig = plot_signals(sig1, sig2, time_axis)
        saver.save_figure(fig_sig, "input_signals")

        fig_spec = plot_spectrum(sig1, sr)
        saver.save_figure(fig_spec, "signal_spectrum")

        fig_mcf = plot_mcf_comparison(results, true_delay_s=true_delay_s)
        saver.save_figure(fig_mcf, "mcf_comparison")

        fig_bar = plot_metrics_bar(results)
        saver.save_figure(fig_bar, "metrics_bar")

    saver.save_csv(results)
    print(f"[comparison] Results saved to: {saver.run_dir}")

    return results


def _load_wav(wav_path: str, signal_config: SignalConfig | None):
    from tde_lab.signals.audio import load_wav_pair
    fl = signal_config.frag_length if signal_config else 1024
    pair = load_wav_pair(wav_path, frag_length=fl)
    # lag axis derived from the file's real sample rate
    return pair.sig1, pair.sig2, pair.lag_axis, pair.sample_rate
