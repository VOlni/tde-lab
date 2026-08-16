"""TDE on real stereo recordings (Signals/*.wav and similar).

Pipeline: load WAV segment → optional VAD fragment gating → optional
pre-filtering (CW-median or DCT thresholding) → per-fragment TDE with any
method set → optional parabolic sub-sample refinement → figures + CSV.

Unlike the synthetic experiments there is no clean reference and no known
true delay: boundaries are derived from channel 1's autocorrelation peak, so
"normal rate" measures per-fragment consistency rather than accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.config.settings import ExportConfig
from tde_lab.methods.base import BaseMethod, MCFResult
from tde_lab.methods.cwmedian import cwmedian_1d
from tde_lab.methods.dct_prefilter import dct_threshold_filter
from tde_lab.methods.subsample import subsample_delay
from tde_lab.signals.audio import AudioPair, load_wav_pair
from tde_lab.visualization.plots import plot_mcf_comparison, plot_metrics_bar, plot_signals
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import figure_style

PREFILTERS = ("cwmedian", "dct")


@dataclass
class AudioTDEReport:
    results: Dict[str, MCFResult]
    pair: AudioPair
    refined_delay_s: Optional[float] = None    # parabolic sub-sample estimate
    refined_delay_samples: Optional[float] = None
    vad_mask: Optional[np.ndarray] = None      # per-fragment bool, None = no VAD
    kept_frags: int = 0
    run_dir: Optional[Path] = None


def _apply_prefilter(sig: np.ndarray, kind: str, cwmedian_window: int, dct_beta: float) -> np.ndarray:
    out = np.empty_like(sig)
    for f in range(sig.shape[1]):
        if kind == "cwmedian":
            out[:, f] = cwmedian_1d(sig[:, f], cwmedian_window)
        else:
            out[:, f] = dct_threshold_filter(sig[:, f], dct_beta)
    return out


def run(
    wav_path: str,
    methods: List[BaseMethod],
    *,
    channels: tuple[int, int] = (0, 1),
    start_s: float = 0.0,
    duration_s: float | None = None,
    frag_length: int = 1024,
    prefilter: str | None = None,          # "cwmedian" | "dct"
    cwmedian_window: int = 5,
    dct_beta: float = 2.7,
    vad: bool = False,
    vad_min_activity: float = 0.5,
    mic_distance: float = 1.0,
    subsample_refine: bool = True,
    parallel: bool = False,
    save: bool = True,
    output_dir: str = "output",
    export: ExportConfig | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> AudioTDEReport:
    if prefilter is not None and prefilter not in PREFILTERS:
        raise ValueError(f"Unknown prefilter {prefilter!r}; choose from {PREFILTERS}")

    pair = load_wav_pair(
        wav_path, frag_length=frag_length, channels=channels,
        start_s=start_s, duration_s=duration_s,
    )
    sig1, sig2 = pair.sig1, pair.sig2

    # ── optional voice-activity gating (per fragment) ────────────────────────
    vad_mask = None
    if vad:
        from tde_lab.preprocessing.vad import fragment_voice_mask
        vad_mask = fragment_voice_mask(
            pair.raw1, pair.sample_rate, frag_length,
            min_activity=vad_min_activity,
        )
        if not vad_mask.any():
            raise ValueError("VAD found no voiced fragments in the selected segment; "
                             "try another segment or disable --vad.")
        sig1 = sig1[:, vad_mask]
        sig2 = sig2[:, vad_mask]

    kept = sig1.shape[1]

    # ── optional robust pre-filtering of both channels ───────────────────────
    if prefilter:
        sig1 = _apply_prefilter(sig1, prefilter, cwmedian_window, dct_beta)
        sig2 = _apply_prefilter(sig2, prefilter, cwmedian_window, dct_beta)

    runner = ExperimentRunner(
        sig1=sig1, sig2=sig2, clean=sig1,       # no clean reference from WAV
        lags=pair.lag_axis, sdvig=0,
        mic_distance=mic_distance, parallel=parallel, progress_cb=progress_cb,
    )
    results = runner.run(methods)

    # ── sub-sample refinement: median of per-fragment parabolic delays ───────
    refined_s = refined_samples = None
    if subsample_refine:
        deltas = np.array([
            subsample_delay(sig2[:, f], sig1[:, f]) for f in range(kept)
        ])
        refined_samples = float(np.median(deltas))
        refined_s = refined_samples * pair.dt

    report = AudioTDEReport(
        results=results, pair=pair,
        refined_delay_s=refined_s, refined_delay_samples=refined_samples,
        vad_mask=vad_mask, kept_frags=kept,
    )

    if save:
        report.run_dir = _save_outputs(report, sig1, sig2, output_dir, export)
    return report


def _save_outputs(report, sig1, sig2, output_dir, export) -> Path:
    export = export or ExportConfig()
    saver = ResultSaver(output_dir, "audio_tde", export)

    with figure_style(export.style):
        fig = plot_signals(sig1, sig2, report.pair.time_axis,
                           title=f"Input fragments — {Path(report.pair.path).name}")
        saver.save_figure(fig, "input_signals")

        fig = plot_mcf_comparison(report.results)
        saver.save_figure(fig, "mcf_comparison")

        fig = plot_metrics_bar(report.results)
        saver.save_figure(fig, "metrics_bar")

    saver.save_csv(report.results)
    _save_summary(saver, report)
    print(f"[audio_tde] Results saved to: {saver.run_dir}")
    return saver.run_dir


def _save_summary(saver: ResultSaver, report: AudioTDEReport) -> None:
    lines = [
        f"file: {report.pair.path}",
        f"sample_rate: {report.pair.sample_rate} Hz",
        f"fragments analysed: {report.kept_frags}",
    ]
    if report.vad_mask is not None:
        lines.append(f"VAD kept {int(report.vad_mask.sum())}/{len(report.vad_mask)} fragments")
    if report.refined_delay_s is not None:
        lines.append(
            f"sub-sample refined delay: {report.refined_delay_samples:.3f} samples "
            f"= {report.refined_delay_s * 1e3:.4f} ms"
        )
    (saver.run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
