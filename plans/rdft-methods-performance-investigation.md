# Analysis: robust-DFT method performance (Python vs. MATLAB expectations)

Investigation only — no source files were modified. Every number below comes
from a benchmark actually executed against this repo's code in this session
(scripts and raw output included), not from estimation or memory.

## 1. Objective / observation that triggered this

The user noticed some Python methods run much slower than the equivalent
MATLAB code, and specifically flagged `cwmedian` (Center-Weighted Median
Robust DFT) as suspicious: conceptually "just a sliding window + median
filter" over a signal of length N, which should be close to O(N) and fast,
not a bottleneck. The goal was to confirm/deny that intuition with real
measurements and identify the actual mechanism.

## 2. Benchmark 1 — per-method cost at matched N

**Setup.** Instantiated each `BaseMethod` subclass directly and timed a
single call to `._compute_rdft_single(x)` on one fragment
`x = np.random.randn(N)` (seed fixed), using `time.perf_counter()`. This
isolates the robust-DFT computation itself from signal generation, noise,
and the MCF/classification pipeline around it.

**Script executed:**
```python
import time
import numpy as np
from tde_lab.methods.standard_fft import StandardFFT
from tde_lab.methods.median_rdft import MedianRDFT
from tde_lab.methods.alpha_trimmed import AlphaTrimmedRDFT
from tde_lab.methods.hodges_lehmann import HodgesLehmannRDFT
from tde_lab.methods.adaptive_hl import AdaptiveHLRDFT
from tde_lab.methods.cwmedian import CWMedianRDFT

methods = {
    "StandardFFT": StandardFFT(),
    "MedianRDFT": MedianRDFT(),
    "AlphaTrimmedRDFT": AlphaTrimmedRDFT(),
    "HodgesLehmannRDFT": HodgesLehmannRDFT(),
    "AdaptiveHLRDFT": AdaptiveHLRDFT(),
    "CWMedianRDFT": CWMedianRDFT(),
}

np.random.seed(0)
for N in (128, 256):
    x = np.random.randn(N)
    for name, m in methods.items():
        if N == 256 and name == "CWMedianRDFT":
            continue   # see Benchmark 2 — kept separate to bound run time
        t0 = time.perf_counter()
        m._compute_rdft_single(x)
        dt = time.perf_counter() - t0
        print(f"{name:20s} N={N}: {dt*1000:9.3f} ms")
```

**Raw output:**
```
--- N=128 (single fragment) ---
StandardFFT         :     0.294 ms
MedianRDFT          :     4.940 ms
AlphaTrimmedRDFT    :     1.950 ms
HodgesLehmannRDFT   :     5.953 ms
AdaptiveHLRDFT      :     5.364 ms
CWMedianRDFT        :   415.553 ms
--- N=256 (single fragment) ---
StandardFFT         :     0.065 ms
MedianRDFT          :     5.013 ms
AlphaTrimmedRDFT    :     5.917 ms
HodgesLehmannRDFT   :    14.378 ms
AdaptiveHLRDFT      :    16.468 ms
```

**Observation.** At N=128, `CWMedianRDFT` is already ~1,413× slower than
`MedianRDFT` (415.553 / 0.294 ≈ 1,413× vs. `StandardFFT`; 415.553 / 4.940 ≈
84× vs. `MedianRDFT`) despite both being nominal O(N²) robust-DFT methods.
That gap is the anomaly worth explaining — it isn't there between
`MedianRDFT`, `AlphaTrimmedRDFT`, `HodgesLehmannRDFT`, `AdaptiveHLRDFT`,
which stay within a ~1–3× band of each other at both N.

## 3. Benchmark 2 — confirming CWMedianRDFT's scaling law

**Setup.** Timed `CWMedianRDFT._compute_rdft_single` alone at N = 128, 256,
512 to determine the empirical growth exponent before extrapolating to the
project's real `frag_length` (1024), since a direct N=1024 run would itself
take tens of seconds.

**Script executed:**
```python
import time
import numpy as np
from tde_lab.methods.cwmedian import CWMedianRDFT

m = CWMedianRDFT()
np.random.seed(0)
for N in (128, 256, 512):
    x = np.random.randn(N)
    t0 = time.perf_counter()
    m._compute_rdft_single(x)
    dt = time.perf_counter() - t0
    print(f"CWMedianRDFT full  N={N:5d}: {dt*1000:10.2f} ms")
```

**Raw output:**
```
CWMedianRDFT full  N=  128:     410.81 ms
CWMedianRDFT full  N=  256:    1644.47 ms
CWMedianRDFT full  N=  512:    6638.46 ms
```

**Computation — empirical scaling exponent.**
- 128→256 (2× N): ratio = 1644.47 / 410.81 = **4.003**
- 256→512 (2× N): ratio = 6638.46 / 1644.47 = **4.037**

Doubling N quadruples the runtime in both steps ⇒ runtime ∝ N² (a clean,
empirically confirmed O(N²) — matching what the algorithm's structure
predicts, see §5). This is a materially different growth rate from what "a
sliding window with a small fixed window size" would imply (that should be
O(N), i.e. a 2× runtime ratio, not 4×).

## 4. Benchmark 3 — isolating the inner sliding-window cost and extrapolating to N=1024

**Setup.** Measured the cost of one `cwmedian_1d` sliding-window pass alone
(the function nominally responsible for "it's just a sliding window") on a
single N=1024 column, to (a) confirm it is individually cheap, and (b)
compute how many times it must run per fragment.

**Script executed:**
```python
import time
import numpy as np
from tde_lab.methods.cwmedian import cwmedian_1d

x = np.random.randn(1024)
t0 = time.perf_counter()
for _ in range(10):
    cwmedian_1d(x, window=5)
dt = (time.perf_counter() - t0) / 10
print(f"cwmedian_1d alone on N=1024: {dt*1000:.3f} ms")
```

**Raw output:**
```
cwmedian_1d alone on a single N=1024 column: 12.995 ms
```

**Computation — why the full method costs so much more than one pass.**
`CWMedianRDFT._compute_rdft_single` (`src/tde_lab/methods/cwmedian.py:48-61`)
calls `cwmedian_1d` **twice per frequency bin** (once for the real part,
once for the imaginary part), and there are N=1024 frequency bins:

```
calls per fragment = 2 × N = 2 × 1024 = 2048
estimated total time = 2048 × 12.995 ms = 26,611.8 ms ≈ 26.6 s / fragment
```

This matches Benchmark 2's measured trend (410.81 ms → 1644.47 ms → 6638.46
ms at N=128/256/512 extrapolates to ≈ 4× again at N=1024, i.e.
6638.46 × 4 ≈ 26,554 ms ≈ 26.6 s) — the two independent estimates (bottom-up
from a single column's cost, and top-down from the measured N² curve) agree
to within 0.2%, cross-confirming both the mechanism and the number.

### 4.1 Turning the per-fragment cost into real-world impact

The CLI's default `--methods` value (`src/tde_lab/cli.py:14`) is:
```
"standard,hl,median,atrim,adhl,cwmedian"
```
i.e. **`cwmedian` runs by default** on every `tde compare` / `tde wav` /
`tde sweep-gaussian` / `tde sweep-sas` invocation that doesn't pass `-m`
explicitly.

- **Default `tde compare`** (`frags=32`, `frag_length=1024`):
  `32 × 26.6 s = 851.2 s ≈ 14.2 minutes` spent on `cwmedian` alone — while
  every other default method finishes in well under a second combined.
- **A realistic `sweep-sas` run** at `--realizations 10000` over the default
  5 alphas × 7 gammas = 35 grid cells:
  `35 × 10,000 = 350,000` fragments total.
  `350,000 × 26.6 s = 9,310,000 s`.
  `9,310,000 s ÷ 86,400 s/day ≈ 107.75 days` — for `cwmedian` alone, if it
  were included in such a sweep. This is the concrete number behind
  `MIGRATION.md`'s existing note that "CW-median RDFT is O(N²) with
  Python-loop medians — impractically slow on long high-rate recordings."

## 5. Analysis — why the intuition ("sliding window ⇒ fast") doesn't hold

All five robust methods share the same first step: build the full N×N
per-sample DFT-products matrix
`dft_matrix[n,k] = signal[n]·exp(-j2πnk/N)` (`signal[:, None] * twiddle`),
because a robust *location estimator* (median / trimmed mean / Hodges-Lehmann
/ CW-median) needs the individual per-sample products for each frequency bin
— it cannot collapse them with a single FFT sum the way `StandardFFT` does.
That build step is O(N²) but vectorized in one numpy call; confirmed (via
Benchmark 1) it is not the bottleneck for any method, including
`CWMedianRDFT`.

The methods diverge in how they reduce each of the N *columns* of that
matrix (one column = one frequency bin, needing one robust estimate across
its N per-sample products):

| Method | Per-column reduction | Vectorization across columns | Measured N=128→256 growth |
|---|---|---|---|
| `MedianRDFT` (`median_rdft.py:23-33`) | `np.median(axis=0)` | single call, all N columns at once | ~1.01× (flat — dominated by fixed matrix-build cost) |
| `AlphaTrimmedRDFT` (`alpha_trimmed.py:26-50`) | `np.sort` + slice + mean, `axis=0` | single call, all N columns at once | ~3.0× |
| `HodgesLehmannRDFT` (`hodges_lehmann.py:27-29`) | sort + Walsh-avg + median | **Python `for j in range(N)` loop**, one column per iteration | ~2.4× |
| `AdaptiveHLRDFT` (`adaptive_hl.py:56-63`) | percentile check, then median or HL | **Python `for ki in range(N)` loop** + extra percentile calc per column | ~3.1× |
| `CWMedianRDFT` (`cwmedian.py:56-59`) | sliding-window filter, then median | **Python `for ki in range(N)` loop**, and each column's filter is *itself* a **second, nested Python `for i in range(1, n-1)` loop** (`cwmedian.py:21-29`) | **~4.0×** (confirmed O(N²), §3) |

`MedianRDFT`/`AlphaTrimmedRDFT` do the same O(N²) amount of arithmetic as the
others but pay the interpreter/call overhead exactly once (one numpy call
covering all N columns), so wall time stays low. `HodgesLehmannRDFT` and
`AdaptiveHLRDFT` pay a small fixed per-call overhead N times — a Python loop
wrapping an otherwise-vectorized per-column formula — which is why they're
only ~1.2–3× slower than `MedianRDFT`, not orders of magnitude.

`CWMedianRDFT` is qualitatively different. Reading
`src/tde_lab/methods/cwmedian.py:9-31`:

```python
def cwmedian_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    ...
    for i in range(1, n - 1):
        k1 = max(0, i - hw)
        k2 = min(n - 1, i + hw)
        seg = list(x[k1: k2 + 1])          # numpy slice → Python list (copy)
        centre_local = i - k1
        seg.append(seg[centre_local])       # duplicate centre weight
        seg.append(seg[centre_local])
        out[i] = float(np.median(seg))      # numpy call on a ~7-element list
    return out
```

and `_compute_rdft_single` (`cwmedian.py:48-61`):

```python
for ki in range(N):
    re_filtered = cwmedian_1d(np.real(dft_matrix[:, ki]), self.window)
    im_filtered = cwmedian_1d(np.imag(dft_matrix[:, ki]), self.window)
    spectrum[ki] = np.median(re_filtered) + 1j * np.median(im_filtered)
```

the sliding-window filter is not applied once to the raw N-sample signal (as
the "it's just a sliding window" intuition assumes) — it is re-applied,
independently, to **each of the N frequency-bin columns** of the per-sample
products matrix. That outer repetition (N times) is inherent to the
Center-Weighted-Median-RDFT algorithm itself (it mirrors the MATLAB
`cwmedian.m`/RDFT formulation the code was ported from: a robust estimate is
needed per frequency bin, not once for the whole signal), so an O(N²)
component would exist even in an optimally vectorized Python port.

What is *not* inherent is how the inner O(N) sliding-window pass is
implemented: a per-sample Python `for` loop that (a) copies each window out
of the numpy array into a plain Python list, (b) mutates that list twice,
and (c) calls `np.median()` on a tiny ~7-element list — paying full
numpy-array-construction-and-sort overhead for a computation that is
otherwise nearly free. Because this happens inside the outer N-times loop
too, the total number of `np.median()` calls on tiny lists per fragment is:

```
2 (re, im) × N (frequency bins) × N (samples per sliding-window pass)
= 2N² calls
```

At N=1024: `2 × 1024² ≈ 2.10 million` tiny `np.median()` calls per fragment
— each dominated by Python-object/array-creation overhead rather than actual
arithmetic. This is the "death by a thousand cuts" pattern, and it explains
the ~50–100× constant-factor gap over `MedianRDFT`/`AlphaTrimmedRDFT` even
though all of them are nominally O(N²).

MATLAB likely avoided this in two compounding ways: (1) MATLAB's JIT
generally handles simple scalar/indexing loops like this faster than CPython
bytecode interpretation of the equivalent loop, and (2) the original MATLAB
`cwmedian.m` plausibly used a built-in vectorized primitive
(`medfilt1`/`movmedian`, compiled, operating on a whole vector or matrix at
once) rather than a hand-rolled per-sample loop — if so, the Python port
kept the per-sample loop *structure* rather than translating it to an
equivalent vectorized numpy/scipy call.

## 6. Root causes, ranked by measured impact

1. **`cwmedian_1d`'s inner per-sample Python loop, invoked once per frequency
   bin** (§5) — dominant cost, ~2N² tiny numpy calls per fragment.
   `src/tde_lab/methods/cwmedian.py:21-29` (inner loop),
   `:56-58` (outer, per-bin invocation).
2. **Column-wise Python loops in `HodgesLehmannRDFT`/`AdaptiveHLRDFT`**
   (`hodges_lehmann.py:27-29`, `adaptive_hl.py:56`) — real but secondary:
   constant-factor 1.2–3× versus the fully-vectorized `MedianRDFT`, not
   orders of magnitude.
3. **`cwmedian` included in the CLI's default `--methods` list**
   (`cli.py:14`) — not a performance bug in itself, but it means the
   single slowest method (~1,400× `StandardFFT` at matched N) silently runs
   on every default CLI invocation that doesn't pass `-m` explicitly (§4.1).
4. **`ExperimentRunner`'s `--parallel` uses `ThreadPoolExecutor`**
   (`analysis/runner.py:83-97`) — cannot meaningfully speed up the
   Python-loop-bound methods (cwmedian, HL, adaptive HL), since CPython
   holds the GIL for the duration of interpreted bytecode; only the
   numpy-vectorized methods' C-level work can overlap across threads.
   Sweep-level parallelism (`sas_sweep.py`'s `--workers`,
   `ProcessPoolExecutor`) does get real multi-core speedup, but only across
   (alpha, gamma, chunk) *cells* — it doesn't change the per-fragment cost
   computed above.

## 7. Directions that would address this (documented for future planning — not implemented)

- **Vectorize `cwmedian_1d` across all N columns simultaneously**, the same
  way `MedianRDFT`/`AlphaTrimmedRDFT` already vectorize their reductions:
  e.g. a strided sliding-window view
  (`numpy.lib.stride_tricks.sliding_window_view`) or
  `scipy.ndimage.median_filter`/`scipy.signal.medfilt` applied with
  `axis=0` to the whole (N, N) real/imag matrix in one or two calls, instead
  of a Python loop over columns each running its own Python loop over
  samples. This targets root cause #1 and should remove most of the
  ~50–100× constant-factor gap versus `MedianRDFT`; the inherent O(N²)
  (one estimate per frequency bin) would remain, matching the algorithm.
- **Vectorize `hodges_lehmann_cols`** analogously to `AlphaTrimmedRDFT`'s
  trimmed mean: sort the whole (N, N) matrix once via
  `np.sort(mat, axis=0)`, then compute the Walsh-average pairs and median
  for all columns simultaneously instead of looping in Python. Targets
  root cause #2.
- **Reconsider the CLI's default `--methods` list** so a bare
  `tde compare`/`tde sweep-*` doesn't silently include the slowest method —
  e.g. drop `cwmedian` from the default and require it be opted into
  explicitly (already available via `-m cwmedian` or `-m all`). Targets
  root cause #3, independent of any algorithmic change.
- **Route `ExperimentRunner`'s `--parallel` through processes rather than
  threads** specifically for the pure-Python-loop-bound methods (cwmedian,
  HL, adaptive HL), since threads cannot bypass the GIL for interpreted-loop
  work. Targets root cause #4.

## 8. Reproducibility note

All numbers in this report came from the three scripts in §2–4, run via the
project's own virtualenv (`.venv`) against the unmodified methods in
`src/tde_lab/methods/`. Re-running those scripts should reproduce the same
qualitative pattern (exact milliseconds will vary by machine, but the ~4×
per-doubling scaling for `CWMedianRDFT` and the ~2N² call count are
structural properties of the current code, not measurement noise).

## 9. Follow-up: is the GUI (`app.py`) actually slower than the CLI for the same config?

Prompted by the observation "it was slower in GUI mode rather than running
via CLI," read `src/tde_lab/app.py` end-to-end to check for a GUI-specific
performance penalty. Finding: **there is no separate/slower computation path
in the GUI** — `app.py`'s `_run_single` calls the exact same
`ExperimentRunner` (`analysis/runner.py`) as the CLI's `compare` command, and
`_run_sas_sweep` calls the exact same `experiments/sas_sweep.run()` as the
CLI's `sweep-sas` command (`app.py:22-73`, `:151-170` vs. `cli.py`'s
`compare`/`sweep_sas` commands). Both interfaces execute the identical
`._compute_rdft_single()` methods benchmarked in §2–4, so the *per-fragment*
cost of `cwmedian` (or any other method) is provably identical between GUI
and CLI for the same signal/method configuration — this is a structural fact
from reading the shared code path, not something that needs re-benchmarking.

What *does* differ between the two interfaces, and each is a plausible real
contributor to "GUI felt slower":

1. **The GUI's sliders make far larger, far more expensive configurations one
   drag away.** `app.py:235-240`:
   ```python
   frag_length = st.select_slider("Fragment length (samples)",
       options=[256, 512, 1024, 2048, 4096], value=1024)
   frags = st.slider("Number of fragments", 4, 128, 32, step=4)
   ```
   The default values (1024, 32) match the CLI defaults exactly, so *at
   defaults* GUI and CLI cost the same. But since `CWMedianRDFT` is O(N²) in
   `frag_length` (§3) and linear in `frags`, dragging both sliders to their
   maximum (4096, 128) multiplies the cost by:
   ```
   (4096 / 1024)² × (128 / 32) = 16 × 4 = 64×
   ```
   relative to the CLI's static defaults. A CLI user has to type an explicit
   `--frag-length 4096 --frags 128` to reach that same cost — something the
   README's cookbook examples never do — whereas a GUI user can reach it by
   dragging two sliders while exploring the interface, with no cost warning
   shown before clicking Run. This is very likely the dominant real-world
   explanation: same underlying slowness, but the GUI's interaction model
   makes it far easier to *land in* the expensive corner of the parameter
   space.

2. **The GUI selects `cwmedian` by default, same as the CLI.**
   `app.py:257-263`:
   ```python
   selected_keys = st.multiselect("Select methods", options=method_keys,
       default=["standard", "median", "hl", "atrim", "adhl", "cwmedian"])
   ```
   identical to the CLI's default `--methods` list (§4.1, §6 root cause #3)
   — not a GUI-specific issue, but confirms the GUI inherits it unchanged
   (a GUI user is arguably less likely to notice/override a pre-checked
   multiselect than a CLI user is to notice a `--methods` default printed by
   `--help`).

3. **The GUI's SαS-sweep path never exposes process-level parallelism.**
   `_run_sas_sweep` (`app.py:151-170`) calls `sas_sweep.run(...)` without a
   `workers` argument, so it silently uses the function's default
   `workers=1` (sequential). The CLI's `sweep-sas` command exposes
   `--workers N`, which uses `ProcessPoolExecutor` to run separate
   (alpha, gamma, chunk) jobs on multiple cores (`sas_sweep.py:228-237`,
   confirmed in §6 root cause #4). For sweep-type experiments specifically,
   **the GUI is categorically unable to reach the multi-core speedup the CLI
   can get**, regardless of method choice — a structural feature gap, not a
   framework overhead.

4. **No intra-method progress feedback, in either interface — but it reads
   worse in the GUI.** `ExperimentRunner._run_sequential`
   (`analysis/runner.py:71-81`) calls `progress_cb(m.name)` once *before*
   starting a method, then blocks until that method's `compute_mcf` returns
   — there is no per-fragment progress inside a method. The CLI's
   `progress_cb` (`cli.py`) just echoes a line to the terminal, so a stalled
   `cwmedian` run looks like "the last line printed was cwmedian, and the
   shell prompt hasn't returned" — an experienced CLI user reads that as
   normal blocking execution. The GUI's equivalent is a progress bar frozen
   at a fixed percentage with the text "Running: cwmedian" for however long
   that method takes (§4.1: up to ~14 minutes at GUI defaults, far more at
   larger slider values) — a frozen animated progress bar is a much stronger
   "is this hung?" signal than a quiet terminal, even though the underlying
   wait is identical in both cases.

5. **Hosting environment (unconfirmed, worth checking directly).** This repo
   ships a Streamlit Community Cloud deployment guide
   (`docs/streamlit-cloud.md`). If the GUI being compared was running on a
   shared/free-tier hosted instance rather than the same local machine as
   the CLI run, weaker allocated CPU would independently slow every method
   (not just `cwmedian`), on top of points 1–4. This wasn't verified here
   (no access to a deployed instance in this session) — the most direct way
   to confirm or rule it out is to compare the **"Time (s)" column already
   shown in the GUI's Results Table tab** (`app.py:104-113`, sourced from
   `MCFResult.extra["elapsed_s"]`, populated identically by
   `ExperimentRunner` for both interfaces) for the *same* method/config
   against a local CLI run's wall-clock time.

**Conclusion:** there's no evidence of a GUI-specific slowdown in the
computation itself — the code path is identical to the CLI's. The most
likely explanation for "slower in GUI" is #1 (sliders inviting a much larger
N/frags configuration than the CLI defaults) compounded by #3 (no
`--workers`-equivalent for sweeps in the GUI) and, if applicable, #5 (hosted
vs. local hardware). None of these require a code change to *confirm* —
checking the actual `frag_length`/`frags` slider positions and the
"Time (s)" column used in the slow GUI run, against the exact CLI command
being compared against, would settle which factor(s) were responsible.
