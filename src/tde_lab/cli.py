"""CLI entry point — Click-based interface for all experiments."""
from __future__ import annotations

import click

from tde_lab.config.settings import SignalConfig, NoiseConfig, MethodConfig, ExportConfig
from tde_lab.methods import build_method, ALL_METHODS, DEFAULT_KEYS


# ── shared options ───────────────────────────────────────────────────────────

def _methods_option(f):
    return click.option(
        "--methods", "-m", default="standard,hl,median,atrim,adhl,cwmedian",
        show_default=True,
        help="Comma-separated method keys. Use 'all' for every method. "
             f"Available: {','.join(ALL_METHODS)}",
    )(f)


def _dct_option(f):
    return click.option(
        "--dct", is_flag=True, default=False,
        help="Wrap all methods with DCT pre-filter.",
    )(f)


def _parallel_option(f):
    return click.option(
        "--parallel", "-p", is_flag=True, default=False,
        help="Run methods in parallel threads.",
    )(f)


def _save_option(f):
    return click.option(
        "--save/--no-save", default=True, show_default=True,
        help="Save plots and CSV to output/.",
    )(f)


def _output_option(f):
    return click.option(
        "--output-dir", default="output", show_default=True,
        help="Base directory for saved outputs.",
    )(f)


def _signal_options(f):
    f = click.option("--frag-length", default=1024, show_default=True)(f)
    f = click.option("--frags",       default=32,   show_default=True)(f)
    f = click.option("--tau",         default=0.4,  show_default=True,
                     help="Relative delay (0..0.5).")(f)
    f = click.option("--ws",          default=5,    show_default=True,
                     help="MA window size for speech signal.")(f)
    return f


def _lag_limit_option(f):
    return click.option(
        "--lag-limit", default=None, type=int,
        help="± search window (samples) for distance methods; must contain "
             "the true delay. Default: all lags. MATLAB experiments used 100.",
    )(f)


def _export_options(f):
    f = click.option("--fig-format", default="png", show_default=True,
                     help="Comma-separated figure formats: png,pdf,svg.")(f)
    f = click.option("--dpi", default=150, show_default=True,
                     help="Raster DPI for PNG figures.")(f)
    f = click.option("--style", default="screen", show_default=True,
                     type=click.Choice(["screen", "paper", "paper_gray"]),
                     help="Figure style preset (paper = 300 dpi serif for publications).")(f)
    return f


def _make_export_config(fig_format: str, dpi: int, style: str) -> ExportConfig:
    export = ExportConfig(
        formats=tuple(fmt.strip().lower() for fmt in fig_format.split(",") if fmt.strip()),
        dpi=dpi,
        style=style,
    )
    export.validate()
    return export


def _parse_methods(methods_str: str, method_cfg: MethodConfig, with_dct: bool):
    if methods_str.strip().lower() == "all":
        keys = list(DEFAULT_KEYS)
    else:
        keys = [k.strip() for k in methods_str.split(",")]
    return [build_method(k, method_cfg, with_dct=with_dct) for k in keys]


def _make_signal_config(frag_length, frags, tau, ws) -> SignalConfig:
    return SignalConfig(frag_length=frag_length, frags=frags, tau=tau, ws=ws)


# ── CLI root ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """TDE Lab — time-delay estimation research tool.

    Robust TDE/MCF method comparison on synthetic noise-like signals
    (Gaussian / SαS noise sweeps) and real stereo recordings.
    """


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("streamlit_args", nargs=-1, type=click.UNPROCESSED)
def gui(streamlit_args):
    """Launch the Streamlit GUI."""
    import os
    import subprocess
    import sys

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    raise SystemExit(
        subprocess.call(["streamlit", "run", app_path, *streamlit_args])
    )


# ── compare ──────────────────────────────────────────────────────────────────

@cli.command()
@_methods_option
@_dct_option
@_parallel_option
@_save_option
@_output_option
@_signal_options
@_export_options
@click.option("--noise", type=click.Choice(["gaussian", "sas"]), default="gaussian",
              show_default=True)
@click.option("--variance", default=1.0, show_default=True, help="Gaussian noise variance.")
@click.option("--alpha",    default=1.5, show_default=True, help="SαS stability exponent.")
@click.option("--gamma",    default=2.0, show_default=True, help="SαS dispersion.")
@click.option("--mic-distance", default=1.0, show_default=True,
              help="Microphone separation in metres.")
@_lag_limit_option
def compare(methods, dct, parallel, save, output_dir,
            frag_length, frags, tau, ws, fig_format, dpi, style,
            noise, variance, alpha, gamma, mic_distance, lag_limit):
    """Single-shot comparison of all selected methods."""
    from tde_lab.experiments.comparison import run

    method_cfg = MethodConfig(mic_distance=mic_distance, lag_limit=lag_limit)
    method_list = _parse_methods(methods, method_cfg, dct)
    sig_cfg = _make_signal_config(frag_length, frags, tau, ws)

    if noise == "gaussian":
        noise_cfg = NoiseConfig(kind="gaussian", variance=variance)
    else:
        noise_cfg = NoiseConfig(kind="sas", alpha=alpha, gamma=gamma)

    click.echo(f"Running comparison: {[m.name for m in method_list]}")
    results = run(
        methods=method_list,
        signal_config=sig_cfg,
        noise_config=noise_cfg,
        save=save,
        output_dir=output_dir,
        parallel=parallel,
        mic_distance=mic_distance,
        export=_make_export_config(fig_format, dpi, style),
        progress_cb=lambda n: click.echo(f"  → {n}"),
    )

    _print_results_table(results)


# ── wav ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("wav_file", type=click.Path(exists=True))
@_methods_option
@_dct_option
@_parallel_option
@_save_option
@_output_option
@_export_options
@_lag_limit_option
@click.option("--frag-length", default=1024, show_default=True)
@click.option("--mic-distance", default=1.0, show_default=True)
@click.option("--channels", default="0,1", show_default=True,
              help="Two channel indices to use as mic 1 and mic 2.")
@click.option("--start", "start_s", default=0.0, show_default=True,
              help="Segment start (seconds).")
@click.option("--duration", "duration_s", default=None, type=float,
              help="Segment duration (seconds) [default: to end of file].")
@click.option("--prefilter", type=click.Choice(["cwmedian", "dct"]), default=None,
              help="Robust pre-filter applied to both channels.")
@click.option("--vad", is_flag=True, default=False,
              help="Keep only voice-active fragments (wavelet VAD).")
@click.option("--subsample/--no-subsample", "subsample_refine",
              default=True, show_default=True,
              help="Refine the delay to sub-sample precision (parabolic peak interpolation).")
def wav(wav_file, methods, dct, parallel, save, output_dir,
        fig_format, dpi, style, lag_limit, frag_length, mic_distance,
        channels, start_s, duration_s, prefilter, vad, subsample_refine):
    """TDE on a real stereo WAV recording (two mic channels)."""
    from tde_lab.experiments.audio_tde import run

    method_cfg = MethodConfig(mic_distance=mic_distance, lag_limit=lag_limit)
    method_list = _parse_methods(methods, method_cfg, dct)
    ch = tuple(int(c) for c in channels.split(","))
    if len(ch) != 2:
        raise click.BadParameter("--channels needs exactly two indices, e.g. 0,1")

    click.echo(f"Loading WAV: {wav_file} (channels {ch}, start {start_s}s)")
    report = run(
        wav_file,
        method_list,
        channels=ch,
        start_s=start_s,
        duration_s=duration_s,
        frag_length=frag_length,
        prefilter=prefilter,
        vad=vad,
        mic_distance=mic_distance,
        subsample_refine=subsample_refine,
        parallel=parallel,
        save=save,
        output_dir=output_dir,
        export=_make_export_config(fig_format, dpi, style),
        progress_cb=lambda n: click.echo(f"  → {n}"),
    )

    click.echo(f"\nSample rate: {report.pair.sample_rate} Hz, "
               f"fragments analysed: {report.kept_frags}")
    if report.vad_mask is not None:
        click.echo(f"VAD kept {int(report.vad_mask.sum())}/{len(report.vad_mask)} fragments")
    _print_results_table(report.results)
    if report.refined_delay_s is not None:
        click.echo(f"sub-sample refined delay: {report.refined_delay_samples:.3f} samples "
                   f"= {report.refined_delay_s * 1e3:.4f} ms")


# ── sweep gaussian ────────────────────────────────────────────────────────────

@cli.command("sweep-gaussian")
@_methods_option
@_dct_option
@_parallel_option
@_save_option
@_output_option
@_signal_options
@_export_options
@click.option("--variances", default="0,0.01,0.1,0.25,0.5,1,5",
              show_default=True, help="Comma-separated variance values to sweep.")
def sweep_gaussian(methods, dct, parallel, save, output_dir,
                   frag_length, frags, tau, ws, fig_format, dpi, style, variances):
    """Sweep Gaussian noise variance across all selected methods."""
    from tde_lab.experiments.gaussian_sweep import run

    method_cfg = MethodConfig()
    method_list = _parse_methods(methods, method_cfg, dct)
    sig_cfg = _make_signal_config(frag_length, frags, tau, ws)
    var_list = [float(v) for v in variances.split(",")]

    click.echo(f"Gaussian sweep over variances: {var_list}")
    run(
        methods=method_list,
        signal_config=sig_cfg,
        variance_values=var_list,
        save=save,
        output_dir=output_dir,
        parallel=parallel,
        export=_make_export_config(fig_format, dpi, style),
        progress_cb=lambda n: click.echo(f"  → {n}"),
    )
    click.echo("Done.")


# ── sweep sas ─────────────────────────────────────────────────────────────────

@cli.command("sweep-sas")
@_methods_option
@_dct_option
@_parallel_option
@_save_option
@_output_option
@_signal_options
@_export_options
@click.option("--alphas",  default="2.0,1.8,1.6,1.4,1.2", show_default=True)
@click.option("--gammas",  default="0,1,2,3,4,5,6",        show_default=True,
              help="Gamma grid. For alpha=1.2 it is scaled by 0.1 (MATLAB convention).")
@click.option("--realizations", default=None, type=int,
              help="Total fragments per (alpha, gamma) cell [default: --frags].")
@click.option("--chunk-size", default=500, show_default=True,
              help="Fragments per cached chunk.")
@click.option("--resume/--fresh", default=True, show_default=True,
              help="Resume from cached chunks, or discard the cache first.")
@click.option("--cache-dir", default=".cache", show_default=True)
@click.option("--workers", default=1, show_default=True,
              help="Worker processes for chunk jobs (1 = sequential).")
@click.option("--seed", default=None, type=int,
              help="Base RNG seed for reproducible, resume-stable sweeps.")
@click.option("--quick", is_flag=True, default=False,
              help="Tiny sweep for a fast end-to-end check "
                   "(alphas 2.0,1.4; gammas 0,3,6; 200 realizations).")
@_lag_limit_option
def sweep_sas(methods, dct, parallel, save, output_dir,
              frag_length, frags, tau, ws, fig_format, dpi, style, alphas, gammas,
              realizations, chunk_size, resume, cache_dir, workers, seed, quick,
              lag_limit):
    """Sweep α × γ parameter space for SαS noise (chunked, cached, resumable)."""
    from tde_lab.experiments.sas_sweep import run

    method_cfg = MethodConfig(lag_limit=lag_limit)
    method_list = _parse_methods(methods, method_cfg, dct)
    sig_cfg = _make_signal_config(frag_length, frags, tau, ws)
    a_list = [float(v) for v in alphas.split(",")]
    g_list = [float(v) for v in gammas.split(",")]

    if quick:
        a_list = [2.0, 1.4]
        g_list = [0.0, 3.0, 6.0]
        realizations = realizations or 200
        chunk_size = min(chunk_size, 100)

    click.echo(f"SαS sweep: alpha={a_list}, gamma={g_list}, "
               f"realizations={realizations or sig_cfg.frags}, chunk={chunk_size}")
    run(
        methods=method_list,
        signal_config=sig_cfg,
        alpha_values=a_list,
        gamma_values=g_list,
        realizations=realizations,
        chunk_size=chunk_size,
        cache_dir=cache_dir,
        resume=resume,
        workers=workers,
        seed=seed,
        save=save,
        output_dir=output_dir,
        parallel=parallel,
        export=_make_export_config(fig_format, dpi, style),
        show_progress=True,
    )
    click.echo("Done.")


# ── validate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--mat-root", type=click.Path(exists=True), required=True,
              help="Root of the MATLAB research folder (contains DSP_new_approach/).")
@click.option("--methods", "-m", default=None,
              help="Comma-separated method keys to validate [default: all in registry].")
@click.option("--realizations", default=1000, show_default=True,
              help="Fragments per (alpha, gamma) cell for the Python sweeps.")
@click.option("--chunk-size", default=250, show_default=True)
@click.option("--cache-dir", default=".cache", show_default=True)
@_output_option
@_export_options
@click.option("--quick", is_flag=True, default=False,
              help="Fast subset: standard + dist-l1 + dist-canberra, "
                   "alphas 2.0/1.6/1.2, 300 realizations.")
def validate(mat_root, methods, realizations, chunk_size, cache_dir,
             output_dir, fig_format, dpi, style, quick):
    """Statistically validate the port against cached MATLAB result matrices."""
    from tde_lab.validation.compare import run_validation

    method_keys = [k.strip() for k in methods.split(",")] if methods else None
    alpha_values = None
    if quick:
        method_keys = method_keys or ["standard", "dist-l1", "dist-canberra"]
        alpha_values = [2.0, 1.6, 1.2]
        realizations = min(realizations, 300)

    report = run_validation(
        mat_root=mat_root,
        method_keys=method_keys,
        realizations=realizations,
        chunk_size=chunk_size,
        cache_dir=cache_dir,
        output_dir=output_dir,
        alpha_values=alpha_values,
        export=_make_export_config(fig_format, dpi, style),
    )
    click.echo(report.summary())
    click.echo(f"\nDetails and overlay figures: {report.output_dir}")
    if not report.passed:
        raise SystemExit(1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_results_table(results):
    click.echo("\n{:<30} {:>12} {:>10} {:>14} {:>10}".format(
        "Method", "Delay (ms)", "Angle (°)", "Normal (%)", "MSE"))
    click.echo("-" * 80)
    for name, r in results.items():
        mse_str = f"{r.mse:.3f}" if r.mse == r.mse else "nan"
        click.echo("{:<30} {:>12.4f} {:>10.2f} {:>14.1f} {:>10}".format(
            name[:30],
            r.delay_seconds * 1e3,
            r.angle_degrees,
            r.normal_rate * 100,
            mse_str,
        ))
    click.echo("")


if __name__ == "__main__":
    cli()
