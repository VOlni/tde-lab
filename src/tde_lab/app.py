"""Streamlit web GUI for the TDE research tool."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import streamlit as st

from tde_lab.config.settings import SignalConfig, NoiseConfig, MethodConfig, ExportConfig
from tde_lab.methods import ALL_METHODS, build_method
from tde_lab.visualization.plots import (
    plot_mcf_comparison, plot_metrics_bar, plot_sweep,
    plot_signals, plot_spectrum, plot_sas_sweep_lines,
)
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import figure_style


# ── experiment runner functions (defined first) ───────────────────────────────

def _run_single(methods, sig_cfg, noise_cfg, wav_file, mic_distance, parallel,
                saver, progress_cb, audio_opts=None):
    from tde_lab.signals.generator import SpeechLikeGenerator
    from tde_lab.signals.noise import make_noise
    from tde_lab.analysis.runner import ExperimentRunner

    refined_note = None
    if wav_file is not None:
        from tde_lab.experiments.audio_tde import run as run_audio
        audio_opts = audio_opts or {}
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_file.read())
            tmp_path = tmp.name
        try:
            report = run_audio(
                tmp_path, methods,
                frag_length=sig_cfg.frag_length,
                mic_distance=mic_distance,
                parallel=parallel,
                save=False,
                progress_cb=progress_cb,
                **audio_opts,
            )
        finally:
            os.unlink(tmp_path)
        results = report.results
        sig1, sig2 = report.pair.sig1, report.pair.sig2
        time_axis = report.pair.time_axis
        sr = report.pair.sample_rate
        true_delay_s = None
        if report.refined_delay_s is not None:
            refined_note = (
                f"sub-sample refined delay: {report.refined_delay_samples:.3f} samples "
                f"= {report.refined_delay_s * 1e3:.4f} ms"
            )
        if report.vad_mask is not None:
            st.info(f"VAD kept {int(report.vad_mask.sum())}/{len(report.vad_mask)} fragments")
    else:
        noise = make_noise(noise_cfg)
        gen = SpeechLikeGenerator(sig_cfg)
        pair = gen.generate(noise, noise)
        sig1, sig2 = pair.sig1, pair.sig2
        time_axis = np.arange(sig1.shape[0]) * sig_cfg.dt
        true_delay_s = pair.sdvig * sig_cfg.dt
        sr = sig_cfg.sample_rate

        runner = ExperimentRunner(
            sig1=sig1, sig2=sig2, clean=pair.clean, lags=sig_cfg.lag_axis,
            sdvig=pair.sdvig, mic_distance=mic_distance,
            parallel=parallel, progress_cb=progress_cb,
        )
        results = runner.run(methods)

    tab_mcf, tab_metrics, tab_signals, tab_table = st.tabs(
        ["MCF Plots", "Metrics", "Signals", "Results Table"]
    )

    with tab_mcf:
        fig = plot_mcf_comparison(results, true_delay_s=true_delay_s)
        st.pyplot(fig)
        if saver:
            saver.save_figure(fig, "mcf_comparison")

    with tab_metrics:
        fig = plot_metrics_bar(results)
        st.pyplot(fig)
        if saver:
            saver.save_figure(fig, "metrics_bar")

    with tab_signals:
        fig_s = plot_signals(sig1, sig2, time_axis)
        st.pyplot(fig_s)
        if saver:
            saver.save_figure(fig_s, "input_signals")
        fig_sp = plot_spectrum(sig1, sr)
        st.pyplot(fig_sp)
        if saver:
            saver.save_figure(fig_sp, "spectrum")

    with tab_table:
        import pandas as pd
        rows = []
        for name, r in results.items():
            rows.append({
                "Method": name,
                "Delay (ms)": f"{r.delay_seconds * 1e3:.4f}",
                "Angle (°)": f"{r.angle_degrees:.2f}",
                "Normal rate (%)": f"{r.normal_rate * 100:.1f}",
                "MSE": f"{r.mse:.3f}" if r.mse == r.mse else "nan",
                "Time (s)": f"{r.extra.get('elapsed_s', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if refined_note:
            st.success(refined_note)
        if saver:
            saver.save_csv(results)


def _run_gaussian_sweep(methods, sig_cfg, sweep_variances_str, parallel, saver, progress_cb):
    from tde_lab.experiments.gaussian_sweep import run

    var_list = [float(v.strip()) for v in sweep_variances_str.split(",")]
    sweep_results = run(
        methods=methods,
        signal_config=sig_cfg,
        variance_values=var_list,
        save=False,
        parallel=parallel,
        progress_cb=progress_cb,
    )

    tab_nr, tab_mse = st.tabs(["Normal Rate", "MSE"])

    with tab_nr:
        fig = plot_sweep(var_list, sweep_results, "Noise variance (σ²)", "normal_rate")
        st.pyplot(fig)
        if saver:
            saver.save_figure(fig, "normal_rate_vs_variance")

    with tab_mse:
        fig = plot_sweep(var_list, sweep_results, "Noise variance (σ²)", "mse")
        st.pyplot(fig)
        if saver:
            saver.save_figure(fig, "mse_vs_variance")

    if saver:
        saver.save_sweep_csv(var_list, sweep_results, "variance")


def _run_sas_sweep(methods, sig_cfg, alphas_str, gammas_str, parallel, saver, progress_cb):
    from tde_lab.experiments.sas_sweep import run
    from tde_lab.visualization.plots import plot_pabn_vs_gamma, plot_sigma_vs_gamma

    a_list = [float(v.strip()) for v in alphas_str.split(",")]
    g_list = [float(v.strip()) for v in gammas_str.split(",")]

    bar = st.progress(0, text="SαS sweep...")
    result = run(
        methods=methods,
        signal_config=sig_cfg,
        alpha_values=a_list,
        gamma_values=g_list,
        save=False,
        parallel=parallel,
        progress_cb=lambda done, total, desc: bar.progress(
            min(int(done / total * 100), 100), text=f"SαS sweep: {desc}"
        ),
    )
    bar.progress(100, text="SαS sweep done")

    tab_pabn, tab_sigma = st.tabs(["Pabn (abnormal %)", "Sigma (RMSE)"])

    with tab_pabn:
        for alpha in result.alpha_values:
            gammas = result.gamma_grid[alpha]
            curves = {m: result.curve(m, alpha, "pabn_percent") for m in result.method_names}
            fig = plot_pabn_vs_gamma(gammas, curves, alpha)
            st.pyplot(fig)
            if saver:
                saver.save_figure(fig, f"pabn_alpha_{alpha:g}")

    with tab_sigma:
        for alpha in result.alpha_values:
            gammas = result.gamma_grid[alpha]
            curves = {m: result.curve(m, alpha, "sigma") for m in result.method_names}
            fig = plot_sigma_vs_gamma(gammas, curves, alpha)
            st.pyplot(fig)
            if saver:
                saver.save_figure(fig, f"sigma_alpha_{alpha:g}")


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TDE Lab",
    page_icon="🎙",
    layout="wide",
)

st.title("TDE Lab")
st.caption("Time-delay estimation of noise-like signals — robust methods, SαS noise sweeps, DOA")


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    st.subheader("Signal Source")
    source = st.radio("Input type", ["Synthetic", "WAV file"], horizontal=True)

    wav_file = None
    audio_opts = {}
    if source == "WAV file":
        wav_file = st.file_uploader("Upload stereo WAV", type=["wav"])
        col_a, col_b = st.columns(2)
        with col_a:
            audio_start = st.number_input("Start (s)", min_value=0.0, value=0.0, step=0.5)
        with col_b:
            audio_duration = st.number_input(
                "Duration (s, 0 = all)", min_value=0.0, value=0.0, step=0.5)
        audio_prefilter = st.selectbox("Pre-filter", ["none", "cwmedian", "dct"])
        audio_vad = st.toggle("Voice-activity gating (VAD)")
        audio_subsample = st.toggle("Sub-sample refinement (parabolic)", value=True)
        audio_opts = {
            "start_s": audio_start,
            "duration_s": audio_duration or None,
            "prefilter": None if audio_prefilter == "none" else audio_prefilter,
            "vad": audio_vad,
            "subsample_refine": audio_subsample,
        }

    st.subheader("Signal Parameters")
    frag_length = st.select_slider(
        "Fragment length (samples)",
        options=[256, 512, 1024, 2048, 4096],
        value=1024,
    )
    frags = st.slider("Number of fragments", 4, 128, 32, step=4)
    tau = st.slider("Relative delay τ", 0.0, 0.49, 0.4, step=0.01,
                    help="Fraction of fragment length")
    mic_distance = st.slider("Mic distance (m)", 0.1, 5.0, 1.0, step=0.1)

    st.subheader("Noise Model")
    noise_kind = st.radio("Noise type", ["Gaussian", "SaS (Alpha-Stable)"], horizontal=True)

    if noise_kind == "Gaussian":
        variance = st.slider("Noise variance", 0.0, 10.0, 1.0, step=0.1)
        noise_cfg = NoiseConfig(kind="gaussian", variance=variance)
    else:
        alpha_val = st.slider("Stability alpha", 0.5, 2.0, 1.5, step=0.1,
                              help="2 = Gaussian, lower = heavier tails")
        gamma_val = st.slider("Dispersion gamma", 0.0, 10.0, 2.0, step=0.5)
        noise_cfg = NoiseConfig(kind="sas", alpha=alpha_val, gamma=gamma_val)

    st.subheader("Methods")
    method_keys = list(ALL_METHODS.keys())
    selected_keys = st.multiselect(
        "Select methods",
        options=method_keys,
        default=["standard", "median", "hl", "atrim", "adhl", "cwmedian"],
    )
    use_dct = st.toggle("DCT pre-filter (all methods)")

    with st.expander("Advanced options"):
        trim_percent = st.slider("Alpha-trim %", 1.0, 49.0, 25.0, step=1.0)
        dct_beta = st.slider("DCT beta threshold", 1.0, 5.0, 2.7, step=0.1)
        cw_window = st.slider("CWMedian window", 3, 15, 5, step=2)
        parallel = st.toggle("Parallel execution")
        save_outputs = st.toggle("Save outputs to disk", value=True)
        output_dir = st.text_input("Output directory", value="output")

    with st.expander("Figure export"):
        fig_formats = st.multiselect(
            "Formats", ["png", "pdf", "svg"], default=["png"],
            help="PDF/SVG are vector formats for papers.",
        )
        fig_dpi = st.select_slider("PNG DPI", options=[100, 150, 200, 300, 600], value=150)
        fig_style = st.selectbox(
            "Style preset", ["screen", "paper", "paper_gray"],
            help="'paper' = 300 dpi serif; 'paper_gray' adds grayscale-safe colors.",
        )

    st.subheader("Experiment")
    exp_type = st.selectbox(
        "Experiment type",
        ["Single comparison", "Gaussian sweep", "SaS sweep"],
    )

    sweep_variances = "0,0.01,0.1,0.25,0.5,1,5"
    sweep_alphas = "2.0,1.8,1.6,1.4,1.2"
    sweep_gammas = "0,1,2,3,4,5,6"

    if exp_type == "Gaussian sweep":
        sweep_variances = st.text_input("Variance values (comma-sep)", sweep_variances)
    elif exp_type == "SaS sweep":
        sweep_alphas = st.text_input("Alpha values", sweep_alphas)
        sweep_gammas = st.text_input("Gamma values", sweep_gammas)

    run_btn = st.button("Run", type="primary", use_container_width=True)


# ── main panel ────────────────────────────────────────────────────────────────

if not run_btn:
    with st.expander("📖 Description", expanded=True):
        st.markdown(
            """
Time-delay estimation (TDE) of wideband noise-like (voice-like) signals under
Gaussian and symmetric alpha-stable (SαS) noise.

**The experiment:** generate a speech-like signal, apply a known cyclic delay
to a copy, add noise with tail heaviness α and dispersion γ, estimate the
delay with a family of methods, and report error statistics over the α × γ
grid:

- **Pabn** — percent of realizations whose estimate falls outside the
  correlation main lobe ("abnormal" estimates).
- **sigma** — RMSE of the normal estimates (samples).

Under heavy-tailed SαS noise the conventional FFT cross-correlation collapses
while distance-metric estimators — especially fractional-power Euclidean —
keep recovering the true delay. Configure parameters in the sidebar and
click **Run** to reproduce this on your own signal.
"""
        )

    with st.expander("🧮 Methods", expanded=True):
        st.markdown(
            "**Correlation family** — estimate = argmax of the (robust) "
            "cross-correlation:"
        )
        st.markdown(
            """
| key | method |
|---|---|
| `standard` | FFT cross-correlation (the conventional approach) |
| `subsample` | parabolic interpolation of the cross-correlation peak (Jacovitti–Scarano), fractional-sample resolution |
| `median` | robust DFT via per-bin median |
| `atrim` | alpha-trimmed mean robust DFT |
| `hl` | Hodges-Lehmann robust DFT |
| `adhl` | adaptive Hodges-Lehmann |
| `cwmedian` | centre-weighted median robust DFT |
"""
        )
        st.markdown(
            '**Distance family** ("new approach") — estimate = argmin of the '
            "distance curve S_e(j) between the reference and the delayed "
            "channel un-shifted by each trial lag j:"
        )
        st.markdown(
            """
| key | distance |
|---|---|
| `dist-l1` | Σ\\|x−y\\| (Euclidean, b=1) |
| `dist-pow05` / `dist-pow15` | Σ\\|x−y\\|^b, b = 0.5 / 1.5 (element-wise power) |
| `dist-mink1` / `dist-mink2` | Minkowski (Σ\\|x−y\\|^p)^(1/p), p = 1 / 2 |
| `dist-canberra` | Σ\\|x−y\\| / (\\|x\\|+\\|y\\|) |
| `dist-braycurtis` | Σ\\|x−y\\| / Σ\\|x+y\\| |
| `dist-hellinger` | (1/√2)·√Σ\\|√x−√y\\|² (complex-safe) |
| `dist-cosine` | 1 − cos∠(x, y) |
| `dist-pearson` | 1 − corr(x, y) |
| `dist-mahalanobis` | literal immse-based MATLAB port (degenerate; excluded from defaults) |
"""
        )
        st.caption(
            "Any method can be wrapped with a DCT-thresholding pre-filter "
            "(sidebar → Methods → DCT pre-filter)."
        )

    st.info("Configure parameters in the sidebar and click **Run** to start.")
    st.stop()

if not selected_keys:
    st.error("Please select at least one method.")
    st.stop()

method_cfg = MethodConfig(
    trim_percent=trim_percent,
    dct_beta=dct_beta,
    cwmedian_window=cw_window,
    mic_distance=mic_distance,
)
methods = [build_method(k, method_cfg, with_dct=use_dct) for k in selected_keys]
sig_cfg = SignalConfig(frag_length=frag_length, frags=frags, tau=tau)

export_cfg = ExportConfig(formats=tuple(fig_formats) or ("png",), dpi=fig_dpi, style=fig_style)
saver = ResultSaver(output_dir, exp_type.lower().replace(" ", "_"), export_cfg) if save_outputs else None

progress_bar = st.progress(0, text="Starting...")
completed = [0]
total_steps = len(methods)


def progress_cb(method_name: str):
    completed[0] += 1
    pct = int(completed[0] / total_steps * 100)
    progress_bar.progress(min(pct, 100), text=f"Running: {method_name}")


try:
    with figure_style(export_cfg.style):
        if exp_type == "Single comparison":
            _run_single(methods, sig_cfg, noise_cfg, wav_file, mic_distance,
                        parallel, saver, progress_cb, audio_opts=audio_opts)
        elif exp_type == "Gaussian sweep":
            _run_gaussian_sweep(methods, sig_cfg, sweep_variances, parallel, saver, progress_cb)
        elif exp_type == "SaS sweep":
            _run_sas_sweep(methods, sig_cfg, sweep_alphas, sweep_gammas, parallel, saver, progress_cb)
except Exception as exc:
    st.error(f"Error: {exc}")
    raise

progress_bar.progress(100, text="Done!")

if saver:
    st.success(f"Results saved to: `{saver.run_dir}`")
