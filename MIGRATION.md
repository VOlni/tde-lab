# MATLAB → Python migration report

Completed 2026-07-14, tagged **v1.0.0**. This document records what was done
and what was verified, milestone by milestone. The full working plan lived at
`~/.claude/plans/this-is-a-folder-iterative-backus.md`; commit history mirrors
the milestones (M0–M8).

## Origin

Source material: the multi-year MATLAB research in
`~/Workspace/Research/Studies/coding` — primarily `DSP_new_approach`
(distance-metric estimators + cached result matrices), `experiment` /
`DSP_main` (conventional + robust-DFT pipeline), `fitSignal_120825.m`
(sub-sample delay), `TDE_audio` (DCT/CW-median filtering), `VAD_wavelet`,
and the `Signals/` stereo recordings.

Base: the existing `doa_research` Python prototype (~2,500 lines: Click CLI,
Streamlit GUI, CMS alpha-stable noise, robust RDFT methods). It was renamed
to `tde_lab` and preserved unmodified as the git baseline commit.

## What was done

- **M0 — Package.** Restructured into an installable src-layout package:
  `pyproject.toml`, `tde` console script, `python -m tde_lab`, fresh
  Python 3.14 venv (old 3.9 venv was broken), pytest scaffold, git history
  starting from the untouched prototype.

- **M1 — Figure export for papers.** rc style presets `screen` / `paper`
  (300 dpi serif) / `paper_gray` (grayscale-safe colors + the
  `comparison_sas_w3_05.m` line-style cycle); PNG+PDF+SVG multi-format saver;
  `--fig-format/--dpi/--style` on every command and in the GUI; MATLAB-replica
  per-alpha figures `plot_pabn_vs_gamma` / `plot_sigma_vs_gamma`.

- **M2 — Long sweeps.** `tde sweep-sas` runs the α×γ grid in cached chunks
  (`StatsAccumulator` sufficient statistics + `SweepStore` .npz cache under a
  config fingerprint): interrupt any time, rerun the same command to resume.
  Verified: a rerun over 20 cached chunks finishes in 0.8 s instead of 17 s.
  Deterministic `--seed`, optional `--workers N` (process pool), `--quick`
  mode, and the MATLAB small-gamma rule (α=1.2 → γ×0.1) built in.

- **M3 — Distance-metric estimators (the "new approach").** All metrics from
  `estimators.m` ported as a `DistanceMethod` family (argmin of the S_e(j)
  curve over circular lags, optional ±`lag_limit` window, clean-curve
  argmin ±5 classification): Euclidean powers b=0.5/1/1.5 (element-wise, see
  README notes), Minkowski p=1/2, Canberra, Bray-Curtis, Hellinger
  (complex-sqrt), cosine, Pearson, and the literal immse-"Mahalanobis"
  (degenerate; excluded from defaults, as in the MATLAB plots). They share
  MCFResult/runner/plots/CSV/sweeps with the correlation methods via a new
  per-method `compute_boundaries` hook.

- **M4 — fitSignal.** Full port of `fitSignal_120825.m`: PART I (coarse
  integer xcorr, negative-peak negation, Nyquist-bin zeroing, weighted phase
  least squares, range wrap) with automatic PART II fallback (iterative
  parabolic refinement). Recovers fractional delays to 1e-6 samples
  noise-free, handles scaled/negated references and wrap-around. The
  `subsample` method now delegates to it per fragment (median over
  normally-classified fragments).
  *Removed in v1.1.0 (third-party-derived code, M. Nentwig); replaced by
  textbook parabolic peak interpolation — see "Post-release changes".*

- **M5 — Validation against the MATLAB results.** A declarative registry maps
  every cached matrix (`data_w3_05/w3_08`, `minkovskii`, `estimatores/*`,
  `all_conv`) to the exact Python setup that reproduces it (WS=3, reference
  channel gain 0.5/0.8 via `SignalConfig.gain1`, true delay 0, ±100 lags).
  Per-cell binomial tolerance with a 10 pp floor (cached MATLAB cells are
  conditioned on a single clean-signal draw), overlay figures (MATLAB dashed
  vs Python solid), CSV + summary. **`tde validate --quick` passes: all
  strict references at 90–100 % of cells** (conventional FFT, dist-l1,
  Canberra × w3_05/w3_08). Along the way this caught and fixed a real bug:
  sweep classification used the FFT boundary index space for windowed
  distance methods (100 % false-abnormal); regression test added.

- **M6 — Real audio.** `tde wav` rebuilt on a new pipeline: channel/segment
  selection, lag axis from the file's real sample rate (fixed a hardcoded
  0.1 s assumption), cwmedian/DCT pre-filtering, fitSignal sub-sample
  refinement, figures/CSV/summary. Verified on
  `Signals/50 см 1 м центр.wav` (44.1 kHz): centered source recovered at
  ~0.2 ms. GUI gained the matching controls.

- **M7 — Wavelet VAD.** Literal port of `method1` (db3, adaptive A3/A4 power
  threshold), `method2` (db4 subband energies + hangover) and the 8-state
  `run_length` smoother (incl. retroactive patches, verified against exact
  state-machine traces). Wired in as optional `--vad` fragment gating.

- **M8 — Docs & release.** Full README (method table with formulas, CLI
  cookbook, MATLAB↔Python convention notes), CHANGELOG, tag v1.0.0.

## The research finding, reproduced

```bash
tde compare -m standard,dist-l1,dist-pow05,dist-canberra --noise sas --alpha 1.5 --gamma 2
```

At α=1.5, γ=2 the conventional FFT collapses (0 % normal estimates, wild
delay) while every distance metric recovers the true 40.04 ms delay —
Euclidean b=0.5 the most robust (62.5 % normal), matching the published
claim that fractional-power metrics resist heavy-tailed noise.

## State at release

96 fast tests + slow validation tests green; quick CLI validation passes;
resume, real-audio and GUI paths exercised end-to-end.

## Post-release changes

- **v1.1.0** — removed the fitSignal port (`methods/fit_signal.py`) so the
  repository contains no other author's code before publication; the
  `subsample` method and the audio-pipeline refinement now use parabolic
  interpolation of the cross-correlation peak (standard technique,
  Jacovitti & Scarano 1993) — simpler, a few hundredths of a sample accuracy
  instead of ~1e-6, implemented from the literature. Added LICENSE (MIT),
  Streamlit Cloud deployment files + guide, and Docker packaging + guide.

## Known limitations / natural follow-ups

- The CW-median RDFT is O(N²) with Python-loop medians — impractically slow
  on long high-rate recordings; vectorization would be the next step.
- The full validation (`tde validate --mat-root ... --realizations 1000`,
  every registry method, ~30–60 min) is set up but was only executed in the
  quick form and the slow test subset.
- Out of scope by decision: TDE_kharkov block-DCT variants, other
  `DSP_new_approach` circshift ablations, the old `voice-activity-detection`
  coursework.
