# Explain NaN-at-low-gamma in sweep plots + document possible fixes

## Context

While reviewing `sweep-sas` output, the user noticed that `sigma`-vs-gamma (and
sometimes `Pabn`-vs-gamma) plots show a lot of NaN gaps even at low gamma
values, and different methods' curves break at different x-positions — which
makes the plots visually inconsistent and harder to compare across methods.
The user wants this mechanism explained and documented, along with the
possible remedies, as a written report — **no code changes**, matching the
existing precedent of `reports/meco2021-consistency.md` (a prior
investigation report checked into the repo).

## Root cause (already verified against this repo's code and cached data)

1. **Where the NaN comes from.** `StatsAccumulator.sigma`
   (`src/tde_lab/analysis/accumulator.py:53-60`) returns `nan` whenever a
   sweep cell has fewer than 2 fragments classified as "normal"
   (`n_normal < 2` — variance of 0 or 1 points is undefined). This is a
   deliberate statistical convention (matches the MATLAB
   `sigma`/`mse_norm` behavior for empty/singleton samples), not a bug.

2. **Why it happens even at "low" gamma.** Gamma is a pure *scale* parameter
   of the SαS distribution — it does not bound tail heaviness, which is
   controlled by alpha alone. For low alpha (1.2, 1.4), the probability of a
   large-magnitude outlier is high regardless of how small gamma is, so
   non-robust methods (`standard` FFT, `dist-mink2`, `median`) can already
   have most fragments' peaks land outside the classification window at
   nominally "low" gamma. This is confirmed directly from the cached sweep
   data already sitting in `.cache/f06b6f95c0684b3d/` in this repo (32
   realizations/cell): e.g. `Standard_FFT` at alpha=1.4 already has
   `n_normal=0` by gamma=1; `Minkowski (p=2)` at alpha=1.2 already collapses
   by gamma≈0.3-0.4 (recall alpha=1.2's gamma grid is itself scaled ×0.1 —
   see `SMALL_GAMMA_SCALE` in `experiments/sas_sweep.py`), i.e. "low gamma" is
   relative to an already-small grid.

3. **Why small sample counts make it worse.** `sweep-sas` defaults to
   `sig_cfg.frags` (32) realizations per cell unless `--realizations` is
   passed. With only 32 draws, the "normal" bucket empties past 2 members
   very easily as soon as the abnormal rate rises above ~90%. The
   README/CLI cookbook recommends `--realizations 10000` for real
   experiments precisely because of this — the demo/default numbers are not
   meant for statistical use.

4. **Why plots look inconsistent, not just gappy.** `plot_metric_vs_gamma`
   (`src/tde_lab/visualization/plots.py:227-253`) plots each method's curve
   with a plain `ax.plot(gamma_values, values, ...)`. Matplotlib silently
   skips NaN points and breaks the line there, with no shared indicator — so
   in a single alpha panel, one method's line may run to gamma=6 while
   another (less robust) one vanishes after gamma=1, with nothing on the
   chart explaining why. That asymmetry, not the NaN itself, is what reads
   as "inconsistent."

## Deliverable

A new report, `reports/nan-at-low-gamma.md` (same style/structure as
`reports/meco2021-consistency.md`), containing:

- **Mechanism** section: the 4 points above, with references to the exact
  functions/lines (`StatsAccumulator.sigma`, `gammas_for_alpha`/
  `SMALL_GAMMA_SCALE`, `plot_metric_vs_gamma`) and the concrete cached-data
  example that demonstrates it (Standard_FFT / Minkowski p=2 collapsing at
  low nominal gamma for alpha ≤ 1.4).
- **Proposed mechanisms to avoid/mitigate it** — documented as options, *not
  implemented*:
  1. **Plot rendering fix** — make NaN/undefined regions explicit and
     consistent across curves in `plot_metric_vs_gamma` (e.g. mark the last
     valid point per method, or shade/annotate the region where a method has
     collapsed) instead of a silent line break. Would touch
     `visualization/plots.py` only.
  2. **Statistical robustness** — raise realizations for cells/methods known
     to collapse early (bigger `--realizations`, or an adaptive scheme that
     runs more chunks specifically for cells near the `n_normal<2` edge)
     using the existing chunked/resumable infra in
     `analysis/sweep_store.py` + `experiments/sas_sweep.py`. Reduces NaN
     frequency but can't eliminate it for true full-collapse cells.
  3. **Metric definition change** — reconsider the `n_normal<2` cutoff itself
     (e.g. surface the sample count `n_normal` alongside `sigma` so readers
     can judge confidence, rather than a hard NaN cliff). Trade-off: a
     looser rule risks reporting a spuriously precise sigma from 1-2 lucky
     survivors.
- A short recommendation of which option(s) are lowest-risk/highest-value,
  without implementing any of them.

## Verification

Documentation-only change — verify by proofreading the new report against
the cited code (`accumulator.py`, `sas_sweep.py`, `plots.py`) and the cached
`.cache/f06b6f95c0684b3d` numbers used as the concrete example, and confirm
it reads consistently with the existing `reports/meco2021-consistency.md`
style.
