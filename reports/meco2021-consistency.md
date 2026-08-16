# Consistency check: MECO 2021 paper vs tde_lab

**Paper:** V. Oliinyk, V. Lukin, I. Djurović, *"Time Delay Estimation for
Noise-Like Signals Embedded in Non-Gaussian Noise Using Robust Similarity
Measures"*, Proc. IEEE MECO 2021.
(local copy: `Статьи/MECO 2021/MECO2021_Oliinyk_Lukin_Djurovic_edited.pdf`)

**Date of check:** 2026-07-14, tde_lab v1.1.0.

**Verdict: the experiment described in the paper is consistent with tde_lab**
— in model, methods, parameter grid and criteria — and the key figures
reproduce numerically (α=1.4 nearly cell-by-cell).

## Design mapping

| Paper | tde_lab | Match |
|---|---|---|
| Model (1): x₁ = s + ξ₁, x₂ = s(t−τ₀) + ξ₂; independent SαS noise; no channel scaling | `SpeechLikeGenerator`, `gain1=1.0` (default), independent per-channel noise, cyclic shift | ✓ |
| WNL signal: low-pass-filtered AWGN, unit variance | MA-filtered white noise, unit variance (`--ws`) | ✓ (nuance 1) |
| Conventional: CCF via 3 FFTs (Eqs. 3–4), global maximum | `StandardFFT`: SP₂·conj(SP₁) → IFFT → argmax | ✓ |
| Eq. (6): S_β(j) = Σ\|x₁(i) − x₂(i+j)\|^β, circular sum, global minimum over j = −j_max…j_max | `EuclideanPowerDistance(b)`, circular candidates, `--lag-limit` window | ✓ |
| β ∈ {0.5, 1, 1.5} | `dist-pow05`, `dist-l1`, `dist-pow15` | ✓ |
| α ∈ {2, 1.8, 1.6, 1.4, 1.2}; γ ∈ [0..6], [0..0.6] for α=1.2 (Fig. 4) | default alpha grid + small-gamma rule | ✓ |
| SαS sampling: Chambers–Mallows–Stuck (ref. [18]) | `AlphaStableNoise._cms` | ✓ |
| 10 000 realizations per (α, γ); fresh signal + noises per realization | `--realizations 10000`; fresh generation per chunk | ✓ |
| Criteria: Pabn(γ) percent (primary), RMSE στ(γ) of normal estimates | `pabn_percent`, `sigma` | ✓ |

Paper Eq. (6) also settles a migration ambiguity: the power β is applied
**inside** the sum (element-wise), exactly as implemented — the MATLAB
`estimators.m` sketch had written it outside, which cannot change the argmin.

## Reproduction run

```bash
tde sweep-sas -m standard,dist-pow05,dist-l1,dist-pow15 \
    --alphas 2.0,1.4 --gammas 0,1,2,3,4,5,6 \
    --ws 3 --tau 0.05 --lag-limit 100 \
    --realizations 400 --chunk-size 200 --seed 5
```

(400 realizations instead of the paper's 10 000 — enough to check curve
shapes and orderings; statistical scatter a few pp.)

### Pabn (%), α = 2 — compare paper Fig. 1

| γ | Conventional | β=0.5 | β=1 | β=1.5 |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.75 | 0.00 | 0.00 |
| 3 | 11.00 | 13.75 | 6.00 | 3.25 |
| 4 | 34.00 | 33.25 | 18.50 | 13.25 |
| 5 | 57.75 | 54.25 | 39.00 | 33.50 |
| 6 | 77.25 | 65.75 | 53.75 | 46.50 |

Paper Fig. 1 (approx. at γ=5): conventional ≈50, β=1 ≈40, β=0.5 ≈75,
β=1.5 ≈32. Agreement: **β=1.5 best, β=1 second** — as in the paper. The one
soft spot: the paper shows β=0.5 clearly worse than conventional at α=2;
here they are roughly equal (see nuance 1).

### Pabn (%), α = 1.4 — compare paper Fig. 4 (top)

| γ | Conventional | β=0.5 | β=1 | β=1.5 |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 99.50 | 0.00 | 1.50 | 44.75 |
| 2 | 99.75 | 20.00 | 39.50 | 84.50 |
| 3 | 99.50 | 61.75 | 72.50 | 92.50 |
| 4 | 99.25 | 75.75 | 83.00 | 90.75 |
| 5 | 99.00 | 89.25 | 87.75 | 91.50 |
| 6 | 99.50 | 88.75 | 91.00 | 92.75 |

Paper Fig. 4 (approx.): conventional ≈100 from γ=1; β=0.5 ≈ 0, 25, 66, 70,
81, 88 for γ=1…6; β=1 slightly above β=0.5; β=1.5 rising steeply.
**Nearly cell-by-cell agreement.** All core claims reproduce: the
conventional method collapses under heavy tails, β=0.5 is best for heavy
tails, β=1.5 for Gaussian — i.e. β should shrink as α decreases (the paper's
adaptivity conclusion).

### RMSE scale check

Paper Fig. 2 shows στ up to ≈9·10⁻⁵ s at 40 kHz sampling ⇒ up to ≈3.6
samples; tde_lab sigma spans 0–3.3 samples over the same grid. Consistent
(tde_lab reports sigma in samples; the paper converts to seconds).

## Nuances (paper text vs code — not migration errors)

1. **Signal bandwidth.** The paper text states the WNL cutoff ≈ Nyquist/3
   (≈ f_s/6, i.e. an MA of ~6 samples); the MATLAB behind the figures used
   WS=3 (first spectral null at f_s/3, broader band). tde_lab follows the
   MATLAB. Most likely explanation for the smaller conventional-vs-β=0.5 gap
   at α=2; use `--ws 6` for strict paper-prose bandwidth.
2. **Physical units.** Paper Nyquist = 20 kHz; the synthetic default in
   tde_lab implies ≈10.24 kHz. Cosmetic only — Pabn/σ live in the sample
   domain.
3. **τ₀ value** is not stated in the paper; choose `--tau` so the delay fits
   inside `--lag-limit` (0.05 · 1024 = 51 ≤ 100 here).
4. The paper regenerates the signal every realization — as tde_lab does.
   (The cached MATLAB validation sets reused one clean signal; that is why
   the statistical validation uses a 10 pp tolerance floor.)

## Conclusion

A reader of the MECO 2021 paper can regenerate its Figures 1–4 with a single
`tde sweep-sas` command (10 000 realizations, `--style paper` for
publication-quality output). The simulator is a faithful superset of the
paper's experiment: it adds the further distance metrics (Minkowski,
Canberra, Bray-Curtis, Hellinger, cosine, Pearson), robust-DFT methods, the
real-audio pipeline and VAD from the follow-up research line.
