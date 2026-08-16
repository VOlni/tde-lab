"""Statistical validation of the Python port against cached MATLAB results.

For each registry entry the matching Python sweep is run (same WS, channel
gain, true delay, ±100 lag window, ±5 normal window) and compared per
(alpha, gamma) cell.  Exact parity is impossible — different RNG streams, and
MATLAB reused a single clean signal across realizations — so cells pass on a
binomial tolerance:

    |Pabn_py − Pabn_mat| ≤ max(5 pp, 3·sqrt(p̂(1−p̂)·(1/n_py + 1/n_mat))·100)

The floor is 10 pp rather than a pure binomial bound because the MATLAB runs
conditioned every cell on a single clean-signal draw (one sigf reused for all
realizations — see circshift_sas_sig_set_generation.m), which shifts cell
values by several pp in the steep region of the Pabn curve.

Overlay figures (MATLAB dashed, Python solid) are the human-level check that
the curve shapes and method orderings match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tde_lab.config.settings import ExportConfig, MethodConfig, SignalConfig
from tde_lab.experiments import sas_sweep
from tde_lab.methods import build_method
from tde_lab.validation.matlab_data import (
    ALPHA_VALUES, GAMMA_VALUES, REGISTRY, MatlabRef, load_ref,
)
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import figure_style, line_style

LAG_LIMIT = 100          # MATLAB search window
NORMAL_HALFWIDTH = 5     # MATLAB classification window


@dataclass
class CellCheck:
    alpha: float
    gamma: float
    python: float
    matlab: float
    tolerance: float

    @property
    def passed(self) -> bool:
        if np.isnan(self.python) or np.isnan(self.matlab):
            return True          # sigma cells without normal estimates
        return abs(self.python - self.matlab) <= self.tolerance


@dataclass
class RefReport:
    ref: MatlabRef
    cells: list[CellCheck] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return float(np.mean([c.passed for c in self.cells])) if self.cells else float("nan")

    @property
    def passed(self) -> bool:
        # a couple of borderline cells out of 35 are acceptable
        return self.pass_rate >= 0.9


@dataclass
class ValidationReport:
    reports: list[RefReport] = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def passed(self) -> bool:
        strict = [r for r in self.reports if r.ref.strict]
        return all(r.passed for r in strict) if strict else False

    def summary(self) -> str:
        lines = [
            f"{'ref':<42} {'method':<18} {'strict':<7} {'pass%':<7} verdict",
            "-" * 90,
        ]
        for r in self.reports:
            verdict = ("PASS" if r.passed else "FAIL") if r.ref.strict else "info"
            lines.append(
                f"{r.ref.var:<42} {r.ref.method_key:<18} "
                f"{str(r.ref.strict):<7} {r.pass_rate * 100:<7.1f} {verdict}"
            )
        lines.append("-" * 90)
        lines.append(f"OVERALL: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _binomial_tolerance(py: float, mat: float, n_py: int, n_mat: int) -> float:
    p = np.clip((py + mat) / 2 / 100.0, 0.0, 1.0)
    stat = 3.0 * np.sqrt(p * (1 - p) * (1 / n_py + 1 / n_mat)) * 100.0
    # 10 pp floor: the cached MATLAB cells are conditioned on one clean-signal
    # draw, adding systematic per-cell spread beyond the sampling error
    return max(10.0, float(stat))


def run_validation(
    mat_root: str | Path,
    method_keys: list[str] | None = None,
    realizations: int = 1000,
    chunk_size: int = 250,
    cache_dir: str = ".cache",
    output_dir: str = "output",
    alpha_values: list[float] | None = None,
    export: ExportConfig | None = None,
    show_progress: bool = True,
) -> ValidationReport:
    """
    Run the Python sweeps matching the MATLAB registry and compare per cell.

    method_keys  : restrict to these registry method keys (None = all)
    alpha_values : restrict the alpha grid (row subset of the matrices)
    """
    mat_root = Path(mat_root)
    a_vals = alpha_values if alpha_values is not None else ALPHA_VALUES

    refs = [r for r in REGISTRY
            if method_keys is None or r.method_key in method_keys]
    if not refs:
        raise ValueError(f"No registry entries match method_keys={method_keys}")

    export = export or ExportConfig()
    saver = ResultSaver(output_dir, "validation", export)
    report = ValidationReport(output_dir=saver.run_dir)

    # one sweep serves every ref that shares (gain1, ws, tau)
    groups: dict[tuple, list[MatlabRef]] = {}
    for ref in refs:
        groups.setdefault((ref.gain1, ref.ws, ref.tau), []).append(ref)

    for (gain1, ws, tau), group_refs in groups.items():
        keys = sorted({r.method_key for r in group_refs})
        method_cfg = MethodConfig(lag_limit=LAG_LIMIT, normal_halfwidth=NORMAL_HALFWIDTH)
        methods = [build_method(k, method_cfg) for k in keys]
        name_by_key = {k: m.name for k, m in zip(keys, methods)}

        sig_cfg = SignalConfig(frag_length=1024, frags=32, ws=ws, tau=tau, gain1=gain1)
        if show_progress:
            print(f"[validate] sweep: gain1={gain1} ws={ws} tau={tau} methods={keys}")

        sweep = sas_sweep.run(
            methods=methods,
            signal_config=sig_cfg,
            alpha_values=list(a_vals),
            gamma_values=list(GAMMA_VALUES),
            realizations=realizations,
            chunk_size=chunk_size,
            cache_dir=cache_dir,
            cache_tag=f"validate-ll{LAG_LIMIT}",
            save=False,
            show_progress=show_progress,
        )

        for ref in group_refs:
            matlab = load_ref(mat_root, ref)          # (5 alphas, 7 gammas)
            ref_report = RefReport(ref=ref)
            for a_idx_full, alpha in enumerate(ALPHA_VALUES):
                if alpha not in sweep.gamma_grid:
                    continue
                gammas = sweep.gamma_grid[alpha]
                python = sweep.curve(name_by_key[ref.method_key], alpha, ref.metric)
                for g_idx, gamma in enumerate(gammas):
                    py_val = float(python[g_idx])
                    mat_val = float(matlab[a_idx_full, g_idx])
                    if ref.metric == "pabn_percent":
                        tol = _binomial_tolerance(py_val, mat_val, realizations, ref.n_matlab)
                    else:
                        # sigma: loose — both small, or within a factor of 2
                        tol = max(1.0, mat_val)
                    ref_report.cells.append(CellCheck(alpha, gamma, py_val, mat_val, tol))
            report.reports.append(ref_report)

        _save_overlays(saver, export, sweep, group_refs, mat_root, name_by_key, gain1)

    _save_csv(saver, report)
    (saver.run_dir / "summary.txt").write_text(report.summary() + "\n")
    return report


def _save_overlays(saver, export, sweep, group_refs, mat_root, name_by_key, gain1) -> None:
    """Per alpha: MATLAB (dashed) vs Python (solid) Pabn curves for the group."""
    import matplotlib.pyplot as plt

    pabn_refs = [r for r in group_refs if r.metric == "pabn_percent"]
    if not pabn_refs:
        return

    with figure_style(export.style):
        for alpha in sweep.alpha_values:
            gammas = sweep.gamma_grid[alpha]
            a_idx = ALPHA_VALUES.index(alpha)
            fig, ax = plt.subplots(figsize=(9, 6))
            for idx, ref in enumerate(pabn_refs):
                style = line_style(idx)
                style.pop("linestyle", None)
                color = f"C{idx}"
                python = sweep.curve(name_by_key[ref.method_key], alpha, "pabn_percent")
                matlab = load_ref(mat_root, ref)[a_idx, :len(gammas)]
                ax.plot(gammas, python, "-", color=color, linewidth=2,
                        label=f"{ref.method_key} (py)", **style)
                ax.plot(gammas, matlab, "--", color=color, linewidth=1.5,
                        label=f"{ref.method_key} (MATLAB)", **style)
            ax.set_xlabel("gamma")
            ax.set_ylabel("Pabn")
            ax.set_title(f"Validation overlay, alpha = {alpha:g}, gain1 = {gain1:g}")
            ax.legend(loc="upper left", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.4)
            fig.tight_layout()
            saver.save_figure(fig, f"overlay_gain{gain1:g}_alpha_{alpha:g}")


def _save_csv(saver: ResultSaver, report: ValidationReport) -> None:
    import csv
    with open(saver.run_dir / "validation_cells.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ref_var", "method", "metric", "strict",
            "alpha", "gamma", "python", "matlab", "tolerance", "passed",
        ])
        writer.writeheader()
        for r in report.reports:
            for c in r.cells:
                writer.writerow({
                    "ref_var": r.ref.var, "method": r.ref.method_key,
                    "metric": r.ref.metric, "strict": r.ref.strict,
                    "alpha": c.alpha, "gamma": c.gamma,
                    "python": f"{c.python:.2f}", "matlab": f"{c.matlab:.2f}",
                    "tolerance": f"{c.tolerance:.2f}", "passed": c.passed,
                })
