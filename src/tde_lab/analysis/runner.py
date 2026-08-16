"""ExperimentRunner — runs one or many methods sequentially or in parallel."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from tde_lab.methods.base import BaseMethod, MCFResult
from tde_lab.analysis.metrics import mcf_boundaries


class ExperimentRunner:
    """
    Runs a list of methods against a fixed signal pair.

    Parameters
    ----------
    sig1, sig2    : (frag_length, n_frags) arrays
    clean         : (frag_length, n_frags) noise-free signal (for boundary calc)
    lags          : time-lag axis (seconds)
    sdvig         : true delay in samples (used for MCF boundary calculation).
                    Pass 0 for WAV input where the true delay is unknown —
                    boundaries will be estimated from the autocorrelation peak.
    mic_distance  : metres, for DOA angle
    parallel      : run methods concurrently with ThreadPoolExecutor
    progress_cb   : optional callable(method_name: str) called before each run
    """

    def __init__(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
        clean: np.ndarray,
        lags: np.ndarray,
        sdvig: int = 0,
        mic_distance: float = 1.0,
        parallel: bool = False,
        progress_cb: Callable[[str], None] | None = None,
    ):
        self.sig1 = sig1
        self.sig2 = sig2
        self.clean = clean
        self.sdvig = sdvig
        self.lags = lags
        self.mic_distance = mic_distance
        self.parallel = parallel
        self.progress_cb = progress_cb or (lambda _: None)

        # Default MCF main-lobe boundaries (correlation-method index space).
        # Distance methods get their own boundaries via boundaries_for().
        self.boundaries = mcf_boundaries(clean[:, 0], sdvig=sdvig)
        self._boundary_cache: dict[str, tuple[int, int]] = {}

    def boundaries_for(self, method: BaseMethod) -> tuple[int, int]:
        """Classification boundaries in the method's own peak index space."""
        key = method.name
        if key not in self._boundary_cache:
            self._boundary_cache[key] = method.compute_boundaries(
                self.clean[:, 0], self.sdvig
            )
        return self._boundary_cache[key]

    def run(self, methods: list[BaseMethod]) -> dict[str, MCFResult]:
        """Run all methods and return {method_name: MCFResult}."""
        if self.parallel and len(methods) > 1:
            return self._run_parallel(methods)
        return self._run_sequential(methods)

    def _run_sequential(self, methods: list[BaseMethod]) -> dict[str, MCFResult]:
        results: dict[str, MCFResult] = {}
        for m in methods:
            self.progress_cb(m.name)
            t0 = time.perf_counter()
            result = m.compute_mcf(
                self.sig1, self.sig2, self.lags, self.boundaries_for(m), self.mic_distance
            )
            result.extra["elapsed_s"] = time.perf_counter() - t0
            results[m.name] = result
        return results

    def _run_parallel(self, methods: list[BaseMethod]) -> dict[str, MCFResult]:
        results: dict[str, MCFResult] = {}
        with ThreadPoolExecutor(max_workers=len(methods)) as executor:
            futures = {
                executor.submit(self._timed_run, m): m.name
                for m in methods
            }
            for future in as_completed(futures):
                name = futures[future]
                self.progress_cb(name)
                try:
                    results[name] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Method {name!r} failed: {exc}") from exc
        return results

    def _timed_run(self, method: BaseMethod) -> MCFResult:
        t0 = time.perf_counter()
        result = method.compute_mcf(
            self.sig1, self.sig2, self.lags, self.boundaries_for(method), self.mic_distance
        )
        result.extra["elapsed_s"] = time.perf_counter() - t0
        return result
