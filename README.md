# TDE Lab

Time-delay estimation (TDE) research framework — the Python migration of a
multi-year MATLAB research line on delay estimation of wideband noise-like
(voice-like) signals under Gaussian and symmetric alpha-stable (SαS) noise.

**The experiment:** generate a speech-like signal (moving-average-filtered
white noise), apply a known cyclic delay to a copy, add noise with tail
heaviness α and dispersion γ, estimate the delay with a family of methods,
and report error statistics over the α × γ grid:

- **Pabn** — percent of realizations whose estimate falls outside the
  correlation main lobe ("abnormal" estimates),
- **sigma** — RMSE of the normal estimates (samples).

The central research finding, reproducible with one command (see below): under
heavy-tailed SαS noise the conventional FFT cross-correlation collapses while
distance-metric estimators — especially fractional-power Euclidean — keep
recovering the true delay.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gui]"
pytest                        # 96 fast tests
pytest -m slow                # + statistical validation vs cached MATLAB results
```

## Interfaces

```bash
tde --help                    # CLI (also: python -m tde_lab)
tde gui                       # Streamlit GUI
```

## Methods

Correlation family — estimate = **argmax** of the (robust) cross-correlation:

| key | method |
|---|---|
| `standard` | FFT cross-correlation (the conventional approach) |
| `subsample` | parabolic interpolation of the cross-correlation peak (Jacovitti–Scarano), fractional-sample resolution |
| `median` | robust DFT via per-bin median |
| `atrim` | alpha-trimmed mean robust DFT |
| `hl` | Hodges-Lehmann robust DFT |
| `adhl` | adaptive Hodges-Lehmann |
| `cwmedian` | centre-weighted median robust DFT |

Distance family ("new approach") — estimate = **argmin** of the distance curve
S_e(j) between the reference and the delayed channel un-shifted by each trial
lag j:

| key | distance |
|---|---|
| `dist-l1` | Σ\|x−y\| (Euclidean, b=1) |
| `dist-pow05` / `dist-pow15` | Σ\|x−y\|^b, b = 0.5 / 1.5 (element-wise power) |
| `dist-mink1` / `dist-mink2` | Minkowski (Σ\|x−y\|^p)^{1/p}, p = 1 / 2 |
| `dist-canberra` | Σ\|x−y\| / (\|x\|+\|y\|) |
| `dist-braycurtis` | Σ\|x−y\| / Σ\|x+y\| |
| `dist-hellinger` | (1/√2)·√Σ\|√x−√y\|² (complex-safe) |
| `dist-cosine` | 1 − cos∠(x, y) |
| `dist-pearson` | 1 − corr(x, y) |
| `dist-mahalanobis` | literal immse-based MATLAB port (degenerate; excluded from `all`) |

Any method can be wrapped with a DCT-thresholding pre-filter (`--dct`).
`--lag-limit N` restricts the distance search window to ±N samples (MATLAB
experiments used 100; the window must contain the true delay).

## CLI cookbook

```bash
# single comparison, heavy SaS noise — the research finding in one line
tde compare -m standard,dist-l1,dist-pow05,dist-canberra \
    --noise sas --alpha 1.5 --gamma 2 --no-save

# full 10k-realization alpha×gamma sweep with caching + resume;
# Ctrl-C at any point, rerun the same command to continue
tde sweep-sas -m standard,dist-l1,dist-pow05,dist-canberra \
    --ws 3 --tau 0 --lag-limit 100 \
    --realizations 10000 --chunk-size 500 --seed 42 \
    --style paper --fig-format png,pdf
tde sweep-sas --quick -m standard,dist-l1     # 2-minute end-to-end check

# statistical validation against the cached MATLAB matrices
tde validate --mat-root /path/to/matlab/research --quick
tde validate --mat-root /path/to/matlab/research --realizations 1000

# real stereo recording (two microphones)
tde wav "Signals/50 см 1 м центр.wav" -m standard,subsample \
    --start 1 --duration 3 --mic-distance 0.5 \
    --prefilter cwmedian --vad --fig-format png,pdf
```

Figure export: `--style paper` (300 dpi serif) or `paper_gray`
(grayscale-safe), `--fig-format png,pdf,svg`, `--dpi N`. Results land in
`output/<experiment>/<timestamp>/` with figures + CSV.

## MATLAB ↔ Python conventions

Grid and classification conventions follow the MATLAB research exactly:
α ∈ {2, 1.8, 1.6, 1.4, 1.2}; γ ∈ [0..6], scaled ×0.1 for α = 1.2; WS = 3 for
the distance experiments; distance window ±100 lags; normal window =
clean-curve argmin ± 5; the w3_05/w3_08 sets scale the noised **reference**
channel by 0.5/0.8 (`SignalConfig.gain1`).

Porting notes (details in module docstrings):

- `circshift(x, i)` ≡ `np.roll(x, i)`; positive delay = channel 2 lags channel 1.
- Exact RNG parity with MATLAB is impossible → validation is statistical
  (binomial tolerance with a 10 pp floor, because the cached MATLAB cells are
  conditioned on a single clean-signal draw) plus overlay figures.
- Euclidean powers are applied **inside** the sum (Σ|x−y|^b): the MATLAB
  sketch writes `sum(...)^b`, but a power outside the sum cannot change the
  argmin, while the cached pow05/pow15 matrices differ from b=1.
- Hellinger uses complex sqrt to replicate MATLAB on negative samples
  (numpy would produce NaN).
- The "Mahalanobis" via `immse` is not a real Mahalanobis distance; ported
  literally, excluded from defaults — as in the MATLAB comparison plots.
- WAV lag/time axes derive from the file's real sample rate.

## Layout

```
src/tde_lab/
├── config/          dataclass configs (signal, noise, methods, export)
├── signals/         speech-like generator, Gaussian + CMS SαS noise, WAV loader
├── methods/         correlation + distance estimators, sub-sample interp, DCT prefilter
├── preprocessing/   wavelet VAD (method1/method2 + run-length smoothing)
├── analysis/        ExperimentRunner, MCF boundaries, sweep accumulator + cache
├── experiments/     comparison, gaussian/SaS sweeps, real-audio pipeline
├── validation/      registry + statistical comparison vs cached .mat results
└── visualization/   plots, style presets, multi-format saver
```

## Provenance

Migrated from the MATLAB folders `DSP_new_approach` (distance metrics,
cached result matrices), `experiment`/`DSP_main` (conventional + robust DFT
pipeline), `TDE_audio` (DCT/CW-median filtering), and `VAD_wavelet`. The
`doa_research` Python prototype served as the base and remains visible in
the git history. All code is original to this project; algorithms taken
from the literature (CMS alpha-stable sampling, parabolic peak
interpolation) are implemented from their published descriptions and cited
in the module docstrings.

## Deploy & share

- [docs/streamlit-cloud.md](docs/streamlit-cloud.md) — put the GUI online so
  anyone can use it from a browser link (no installation).
- [docs/docker.md](docs/docker.md) — run the GUI or CLI anywhere via Docker:

  ```bash
  # start Docker Desktop first (macOS: open -a Docker), then:
  docker build -t tde-lab .
  docker run --rm -p 8501:8501 tde-lab        # GUI at http://localhost:8501
  ```

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for any noncommercial
use (research, education, personal projects); **commercial use is not
permitted without the author's prior written consent**. For commercial
licensing inquiries, contact oliinyk.vch@pm.me.
