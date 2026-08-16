# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TDE Lab is a time-delay-estimation (TDE) research framework — the Python
migration of a multi-year MATLAB research line on delay estimation of
wideband noise-like (voice-like) signals under Gaussian and symmetric
alpha-stable (SαS) noise. The experiment: generate a speech-like signal,
apply a known cyclic delay to a copy, add noise with tail-heaviness α and
dispersion γ, estimate the delay with a family of methods, and report error
statistics (**Pabn** = % of estimates outside the correlation main lobe,
**sigma** = RMSE of the normal estimates) over the α×γ grid. See README.md
for the full method table and the research finding this framework reproduces.

## Commands

```bash
# setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gui]"

# tests
pytest                              # fast suite only (96 tests, ~seconds)
pytest -m slow                      # + slow MATLAB statistical validation tests
pytest tests/test_distance_metrics.py::test_name   # single test
ruff check src tests                # lint (line-length 100, configured in pyproject.toml)

# CLI (also: python -m tde_lab)
tde --help
tde compare -m standard,dist-l1,dist-pow05,dist-canberra --noise sas --alpha 1.5 --gamma 2 --no-save
tde sweep-sas --quick -m standard,dist-l1        # ~2 min end-to-end sweep check
tde validate --mat-root /path/to/matlab/research --quick
tde gui                             # Streamlit GUI (forwards unknown args to streamlit)
```

The `slow` marker (deselected by default via `addopts` in pyproject.toml)
covers MATLAB validation sweeps. Those tests read cached MATLAB result
matrices from a hardcoded path (`tests/test_matlab_validation.py: MAT_ROOT`,
a sibling MATLAB research folder) and self-skip via `needs_mat` when that
folder isn't present — expect them to skip in most environments, that's normal.

## Architecture

The package is a pipeline of independent layers under `src/tde_lab/`, wired
together by thin `experiments/*.py` entry points and an even thinner
`cli.py`. Read `methods/base.py` and `analysis/runner.py` first — they
define the shared contract every method and experiment builds on.

```
config/         dataclass configs: SignalConfig, NoiseConfig, MethodConfig,
                ExportConfig, ExperimentConfig (config/settings.py)
signals/        generator.py (MA-filtered white noise → speech-like signal),
                noise.py (Gaussian + CMS alpha-stable sampling), audio.py (WAV)
methods/        one class per estimator, all implementing BaseMethod
preprocessing/  wavelet VAD (method1/method2 + run-length smoothing)
analysis/       ExperimentRunner, MCF boundaries, sweep accumulator + disk cache
experiments/    comparison.py, gaussian_sweep.py, sas_sweep.py, audio_tde.py —
                each exposes a run(...) function that cli.py calls
validation/     registry of cached MATLAB reference matrices + statistical
                comparison against them
visualization/  plots, rc style presets, multi-format figure saver
cli.py          Click CLI — parses options, builds configs/methods, calls
                the matching experiments.*.run()
app.py          Streamlit GUI (same experiments/* backends as the CLI)
```

### Method contract (`methods/base.py`)

Every estimator subclasses `BaseMethod` and implements
`_compute_rdft_single(signal_1d)` (per-fragment robust DFT for correlation
methods) or extends `DistanceMethod` (per-fragment distance curve, in
`methods/distance_base.py`). The shared `compute_mcf()` pipeline does:
robust-DFT-per-fragment → cross-power `SP2·conj(SP1)` → ifft+fftshift → peak
per fragment → normal/abnormal classification against boundaries → aggregate
`MCFResult`. Two method families share this pipeline but differ in what
"peak" means:

- **Correlation family** (`standard`, `subsample`, `median`, `atrim`, `hl`,
  `adhl`, `cwmedian`): estimate = **argmax** of |MCF|.
- **Distance family** (`dist-*`): estimate = **argmin** of the distance curve
  S_e(j); these override `compute_boundaries()` to classify against the
  clean-curve argmin ± `normal_halfwidth` instead of the MCF main lobe —
  mixing up the two boundary conventions was a real bug caught during
  migration (100% false-abnormal for windowed distance methods).

New methods register in `methods/__init__.py`'s `ALL_METHODS` dict and the
`build_method()` factory; `DEFAULT_KEYS` (what `-m all` resolves to)
excludes `dist-mahalanobis` (a literal, deliberately-degenerate MATLAB port).

### Sweep caching (`analysis/sweep_store.py`, `experiments/sas_sweep.py`)

`sweep-sas` runs are chunked and cached to `.cache/<config-fingerprint>/` as
`.npz` files (`StatsAccumulator` sufficient statistics, not raw samples) so
a run can be Ctrl-C'd and resumed by rerunning the same command. The
fingerprint (sha1 of the canonicalized config dict) covers every parameter
that affects the numbers, so a stale cache is never silently reused for a
different config — changing any sweep parameter starts a fresh cache
directory rather than corrupting the old one.

### Validation against MATLAB (`validation/`)

`validation/matlab_data.py` declares a `REGISTRY` mapping cached MATLAB
result matrices to the exact Python config that reproduces them (WS, channel
gain, lag window, etc.). `validation/compare.py` re-runs those sweeps in
Python and compares per-(alpha, gamma) cell against a binomial tolerance
with a 10pp floor — exact parity is impossible (different RNG streams, and
the MATLAB runs conditioned every cell on a single clean-signal draw).
Overlay figures (MATLAB dashed vs Python solid) are the human-level check.

### Output layout

Experiment runs write to `output/<experiment>/<timestamp>/` (figures + CSV);
`--style paper`/`paper_gray` gives publication-ready 300dpi serif/grayscale
figures. Both `output/` and `.cache/` are gitignored — treat their contents
as disposable, regenerable artifacts.

## Conventions carried over from the MATLAB research

These are load-bearing and non-obvious — see README.md's "MATLAB ↔ Python
conventions" section and individual module docstrings for more:

- `circshift(x, i)` ≡ `np.roll(x, i)`; positive delay = channel 2 lags channel 1.
- Euclidean powers are applied **inside** the sum (Σ|x−y|^b), not outside —
  matches the cached MATLAB pow05/pow15 matrices even though a power outside
  the sum wouldn't change the argmin.
- Hellinger uses complex sqrt on negative samples to replicate MATLAB
  (plain numpy sqrt would produce NaN).
- α grid is `{2, 1.8, 1.6, 1.4, 1.2}`; γ grid `[0..6]`, scaled ×0.1 for α=1.2
  (heavier tails at the same dispersion would swamp every estimator).
- Tests seed the legacy global `np.random` RNG (`conftest.py`'s autouse
  `seeded_rng` fixture) rather than a `Generator` instance — the codebase
  itself uses the global RNG throughout, intentionally, for MATLAB parity.
