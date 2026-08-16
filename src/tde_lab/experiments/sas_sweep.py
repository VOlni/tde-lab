"""Alpha-Stable (SαS) noise sweep experiment.

Sweeps an alpha × gamma grid with many realizations per cell, in chunks that
are cached to disk (resume-able after interruption).  Produces the MATLAB-style
per-alpha comparison figures: Pabn vs gamma and sigma vs gamma, one line per
method.

Conventions carried over from the MATLAB research:
- alpha grid [2, 1.8, 1.6, 1.4, 1.2]; gamma grid [0..6]
- for alpha = 1.2 the gamma grid is scaled by 0.1 ([0..0.6]) because heavier
  tails at the same dispersion would swamp every estimator
- Pabn = percent of fragments whose peak lands outside the MCF main lobe
- sigma = RMSE of the normal peak positions (samples)
"""
from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from tde_lab.analysis.accumulator import StatsAccumulator
from tde_lab.analysis.runner import ExperimentRunner
from tde_lab.analysis.sweep_store import SweepStore
from tde_lab.config.settings import SignalConfig, ExportConfig
from tde_lab.methods.base import BaseMethod
from tde_lab.signals.generator import SpeechLikeGenerator
from tde_lab.signals.noise import AlphaStableNoise
from tde_lab.visualization.plots import plot_pabn_vs_gamma, plot_sigma_vs_gamma
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import figure_style


DEFAULT_ALPHA_VALUES = [2.0, 1.8, 1.6, 1.4, 1.2]
DEFAULT_GAMMA_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SMALL_GAMMA_ALPHAS = (1.2,)
SMALL_GAMMA_SCALE = 0.1


@dataclass
class SweepResult:
    """Aggregated sweep statistics.

    stats[method_name][(alpha, gamma)] holds the accumulated peak statistics
    for that cell; gamma_grid[alpha] lists the gammas actually used for that
    alpha (the small-gamma rule may rescale them).
    """
    alpha_values: List[float]
    gamma_grid: Dict[float, List[float]]
    stats: Dict[str, Dict[Tuple[float, float], StatsAccumulator]]

    @property
    def method_names(self) -> List[str]:
        return list(self.stats.keys())

    def curve(self, method: str, alpha: float, metric: str = "pabn_percent") -> np.ndarray:
        """Metric values vs gamma for one method at one alpha."""
        cells = self.stats[method]
        return np.array([
            getattr(cells[(alpha, gamma)], metric)
            for gamma in self.gamma_grid[alpha]
        ])


def gammas_for_alpha(
    alpha: float,
    gamma_values: List[float],
    small_gamma_alphas: Tuple[float, ...] = SMALL_GAMMA_ALPHAS,
    small_gamma_scale: float = SMALL_GAMMA_SCALE,
) -> List[float]:
    """MATLAB rule: alpha=1.2 uses the gamma grid scaled by 0.1 ([0..0.6])."""
    if alpha in small_gamma_alphas:
        return [g * small_gamma_scale for g in gamma_values]
    return list(gamma_values)


def _job_seed(base_seed: Optional[int], alpha: float, gamma: float, chunk: int) -> Optional[int]:
    """Distinct deterministic RNG seed per (alpha, gamma, chunk) job."""
    if base_seed is None:
        return None
    key = f"{base_seed}|{alpha:g}|{gamma:g}|{chunk}".encode()
    return int.from_bytes(hashlib.sha1(key).digest()[:4], "big")


def _run_cell_chunk(
    methods: List[BaseMethod],
    sig_cfg: SignalConfig,
    alpha: float,
    gamma: float,
    n_frags: int,
    seed: Optional[int],
    parallel: bool,
) -> Dict[str, StatsAccumulator]:
    """One chunk of realizations for one (alpha, gamma) cell, all methods."""
    if seed is not None:
        np.random.seed(seed)

    noise = AlphaStableNoise(alpha=alpha, gamma=gamma)
    gen = SpeechLikeGenerator(replace(sig_cfg, frags=n_frags))
    pair = gen.generate(noise, noise)

    runner = ExperimentRunner(
        sig1=pair.sig1, sig2=pair.sig2, clean=pair.clean,
        lags=sig_cfg.lag_axis, sdvig=pair.sdvig, parallel=parallel,
    )
    results = runner.run(methods)

    accs: Dict[str, StatsAccumulator] = {}
    for m in methods:
        r = results[m.name]
        peaks = np.asarray(r.extra["peaks"])
        # boundaries live in the method's own peak index space (a windowed
        # distance method's space differs from the fftshifted MCF space)
        left, right = runner.boundaries_for(m)
        acc = StatsAccumulator()
        acc.update(peaks, (peaks >= left) & (peaks <= right))
        accs[m.name] = acc
    return accs


def run(
    methods: List[BaseMethod],
    signal_config: SignalConfig | None = None,
    alpha_values: List[float] | None = None,
    gamma_values: List[float] | None = None,
    *,
    realizations: int | None = None,
    chunk_size: int = 500,
    cache_dir: str = ".cache",
    cache_tag: str = "",
    resume: bool = True,
    workers: int = 1,
    seed: int | None = None,
    small_gamma_alphas: Tuple[float, ...] = SMALL_GAMMA_ALPHAS,
    small_gamma_scale: float = SMALL_GAMMA_SCALE,
    save: bool = True,
    output_dir: str = "output",
    parallel: bool = False,
    export: ExportConfig | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    show_progress: bool = False,
) -> SweepResult:
    """
    Run all methods over the alpha × gamma grid with `realizations` fragments
    per cell (default: signal_config.frags), computed in cached chunks.

    Interrupt at any time; rerunning the same configuration resumes from the
    chunks already on disk.  Pass resume=False to discard the cache first.

    progress_cb : optional callable(done_jobs, total_jobs, description)
    show_progress : print a tqdm bar (CLI)
    """
    sig_cfg = signal_config or SignalConfig()
    a_vals = alpha_values if alpha_values is not None else DEFAULT_ALPHA_VALUES
    g_vals = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
    total = realizations if realizations is not None else sig_cfg.frags

    chunk_sizes = [chunk_size] * (total // chunk_size)
    if total % chunk_size:
        chunk_sizes.append(total % chunk_size)

    store = SweepStore(cache_dir, {
        "experiment": "sas_sweep",
        "frag_length": sig_cfg.frag_length,
        "ws": sig_cfg.ws,
        "tau": sig_cfg.tau,
        "gain1": sig_cfg.gain1,
        "chunk_size": chunk_size,
        "seed": seed,
        "tag": cache_tag,        # distinguishes method variants (e.g. lag_limit)
    })
    if not resume:
        store.clear()
    store.write_manifest()

    gamma_grid = {
        alpha: gammas_for_alpha(alpha, g_vals, small_gamma_alphas, small_gamma_scale)
        for alpha in a_vals
    }

    # (alpha, gamma, chunk_index, n_frags) jobs, split by cache state
    jobs = [
        (alpha, gamma, ci, n)
        for alpha in a_vals
        for gamma in gamma_grid[alpha]
        for ci, n in enumerate(chunk_sizes)
    ]
    cached = [j for j in jobs if all(store.has(m.name, j[0], j[1], j[2]) for m in methods)]
    to_run = [j for j in jobs if j not in cached]

    stats: Dict[str, Dict[Tuple[float, float], StatsAccumulator]] = {
        m.name: {
            (alpha, gamma): StatsAccumulator()
            for alpha in a_vals for gamma in gamma_grid[alpha]
        }
        for m in methods
    }

    for alpha, gamma, ci, _ in cached:
        for m in methods:
            stats[m.name][(alpha, gamma)].merge(store.load(m.name, alpha, gamma, ci))

    done = len(cached)
    total_jobs = len(jobs)

    def _report(alpha: float, gamma: float) -> None:
        if progress_cb:
            progress_cb(done, total_jobs, f"alpha={alpha:g} gamma={gamma:g}")

    progress_iter = None
    if show_progress and to_run:
        from tqdm import tqdm
        progress_iter = tqdm(total=total_jobs, initial=done, unit="chunk", desc="SaS sweep")

    def _absorb(alpha: float, gamma: float, ci: int, accs: Dict[str, StatsAccumulator]) -> None:
        nonlocal done
        for name, acc in accs.items():
            store.save(name, alpha, gamma, ci, acc)
            stats[name][(alpha, gamma)].merge(acc)
        done += 1
        if progress_iter:
            progress_iter.update(1)
        _report(alpha, gamma)

    if workers > 1 and to_run:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_cell_chunk, methods, sig_cfg, alpha, gamma, n,
                            _job_seed(seed, alpha, gamma, ci), False): (alpha, gamma, ci)
                for alpha, gamma, ci, n in to_run
            }
            for future in as_completed(futures):
                alpha, gamma, ci = futures[future]
                _absorb(alpha, gamma, ci, future.result())
    else:
        for alpha, gamma, ci, n in to_run:
            accs = _run_cell_chunk(methods, sig_cfg, alpha, gamma, n,
                                   _job_seed(seed, alpha, gamma, ci), parallel)
            _absorb(alpha, gamma, ci, accs)

    if progress_iter:
        progress_iter.close()

    result = SweepResult(alpha_values=list(a_vals), gamma_grid=gamma_grid, stats=stats)

    if save:
        _save_outputs(result, output_dir, export)

    return result


def _save_outputs(result: SweepResult, output_dir: str, export: ExportConfig | None) -> None:
    export = export or ExportConfig()
    saver = ResultSaver(output_dir, "sas_sweep", export)

    with figure_style(export.style):
        for alpha in result.alpha_values:
            gammas = result.gamma_grid[alpha]
            pabn = {m: result.curve(m, alpha, "pabn_percent") for m in result.method_names}
            saver.save_figure(plot_pabn_vs_gamma(gammas, pabn, alpha), f"pabn_alpha_{alpha:g}")

            sigma = {m: result.curve(m, alpha, "sigma") for m in result.method_names}
            saver.save_figure(plot_sigma_vs_gamma(gammas, sigma, alpha), f"sigma_alpha_{alpha:g}")

    _save_csv(saver, result)
    print(f"[sas_sweep] Results saved to: {saver.run_dir}")


def _save_csv(saver: ResultSaver, result: SweepResult) -> None:
    import csv
    path = saver.run_dir / "sas_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "alpha", "gamma", "method", "realizations",
            "norm_pct", "pabn_pct", "sigma",
        ])
        writer.writeheader()
        for method, cells in result.stats.items():
            for (alpha, gamma), acc in cells.items():
                writer.writerow({
                    "alpha": alpha,
                    "gamma": gamma,
                    "method": method,
                    "realizations": acc.n_total,
                    "norm_pct": f"{acc.norm_percent:.2f}",
                    "pabn_pct": f"{acc.pabn_percent:.2f}",
                    "sigma": f"{acc.sigma:.4f}" if acc.sigma == acc.sigma else "nan",
                })
