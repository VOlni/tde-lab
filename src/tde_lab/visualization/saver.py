"""Save figures and CSV results to output/<experiment>/<timestamp>/."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from tde_lab.config.settings import ExportConfig
from tde_lab.methods.base import MCFResult


def _make_run_dir(base: str, experiment_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / experiment_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class ResultSaver:
    """
    Saves figures (PNG and/or vector PDF/SVG) and CSV metrics tables
    for one experiment run.

    Usage
    -----
    saver = ResultSaver("output", "gaussian_sweep", ExportConfig(formats=("png", "pdf")))
    saver.save_figure(fig, "mcf_comparison")
    saver.save_csv(results)
    print(saver.run_dir)
    """

    def __init__(
        self,
        base_dir: str = "output",
        experiment_name: str = "experiment",
        export: ExportConfig | None = None,
    ):
        self.run_dir = _make_run_dir(base_dir, experiment_name)
        self.export = export or ExportConfig()
        self.export.validate()

    def save_figure(self, fig: plt.Figure, name: str, dpi: int | None = None) -> list[Path]:
        """Save a Figure in every configured format.  Returns the saved paths."""
        # sanitise filename
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        paths = []
        for fmt in self.export.formats:
            path = self.run_dir / f"{safe}.{fmt}"
            fig.savefig(path, dpi=dpi or self.export.dpi, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
        return paths

    def save_csv(self, results: Dict[str, MCFResult], filename: str = "metrics.csv") -> Path:
        """Write a CSV with one row per method."""
        path = self.run_dir / filename
        fieldnames = [
            "method", "delay_samples", "delay_ms", "angle_degrees",
            "normal_rate_pct", "mse", "elapsed_s",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, r in results.items():
                writer.writerow({
                    "method":           name,
                    "delay_samples":    r.delay_samples,
                    "delay_ms":         f"{r.delay_seconds * 1e3:.4f}",
                    "angle_degrees":    f"{r.angle_degrees:.4f}",
                    "normal_rate_pct":  f"{r.normal_rate * 100:.2f}",
                    "mse":              f"{r.mse:.4f}" if not (r.mse != r.mse) else "nan",
                    "elapsed_s":        f"{r.extra.get('elapsed_s', 0):.4f}",
                })
        return path

    def save_sweep_csv(
        self,
        sweep_values: list,
        sweep_results: Dict[str, list],
        param_name: str,
        filename: str = "sweep_metrics.csv",
    ) -> Path:
        """CSV for sweep experiments: rows = (param_value, method, metrics...)."""
        path = self.run_dir / filename
        fieldnames = [param_name, "method", "normal_rate_pct", "mse", "delay_ms", "angle_degrees"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for method_name, result_list in sweep_results.items():
                for val, r in zip(sweep_values, result_list):
                    writer.writerow({
                        param_name:        val,
                        "method":          method_name,
                        "normal_rate_pct": f"{r.normal_rate * 100:.2f}",
                        "mse":             f"{r.mse:.4f}" if not (r.mse != r.mse) else "nan",
                        "delay_ms":        f"{r.delay_seconds * 1e3:.4f}",
                        "angle_degrees":   f"{r.angle_degrees:.4f}",
                    })
        return path
