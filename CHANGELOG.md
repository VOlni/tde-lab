# Changelog

## v1.1.0 — 2026-07-14

Publication preparation.

- **Removed third-party-derived code**: the fitSignal port
  (`methods/fit_signal.py`, derived from M. Nentwig's fitSignal_120825.m)
  is gone; the `subsample` method and the audio-pipeline refinement now use
  parabolic interpolation of the cross-correlation peak (standard textbook
  technique, Jacovitti & Scarano 1993, implemented from the literature).
  Accuracy: a few hundredths of a sample instead of ~1e-6 on clean signals.
- Added `LICENSE` (MIT).
- Streamlit Community Cloud deployment: `streamlit_app.py` entry point,
  `requirements.txt`, guide in `docs/streamlit-cloud.md`.
- Docker packaging: `Dockerfile` (GUI by default, CLI via command override),
  `.dockerignore`, guide in `docs/docker.md`.
- `tde gui` now forwards unknown options to Streamlit
  (e.g. `--server.address`).

## v1.0.0 — 2026-07-14

First complete release of the MATLAB → Python migration.

- **M0** — `doa_research` prototype restructured into the installable
  `tde-lab` package (src layout, pyproject, `tde` CLI entry point, pytest).
- **M1** — paper-quality figure export: style presets (`screen`, `paper`,
  `paper_gray`), PNG/PDF/SVG, MATLAB-style Pabn/sigma-vs-gamma figures.
- **M2** — chunked, cached, resumable SαS sweeps (StatsAccumulator +
  SweepStore, deterministic seeds, worker processes, small-gamma rule).
- **M3** — distance-metric estimators (Euclidean powers, Minkowski, Canberra,
  Bray-Curtis, Hellinger, cosine, Pearson, literal immse-Mahalanobis) with
  per-method classification boundaries.
- **M4** — full fitSignal port (PART I weighted-phase LSQ + PART II iterative
  fallback); `subsample` method now has true fractional resolution.
- **M5** — statistical validation against the cached MATLAB result matrices
  (registry of 19 references; binomial tolerance; overlay figures).
  All strict references pass at 90–100% of cells.
- **M6** — real-audio pipeline: channel/segment selection at the file's real
  sample rate, CW-median/DCT pre-filtering, fitSignal refinement, extended
  `tde wav` CLI and GUI controls.
- **M7** — wavelet VAD (method1/method2 + 8-state run-length smoother) as
  optional fragment gating for real recordings.
