"""Disk cache for chunked sweep results — enables resume after interruption.

Layout:  <cache_dir>/<fingerprint>/<method>__a<alpha>__g<gamma>__c<chunk>.npz
plus a manifest.json recording the configuration the fingerprint was built
from.  The fingerprint covers every parameter that changes the numbers, so a
stale cache can never be silently reused for a different experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from tde_lab.analysis.accumulator import StatsAccumulator


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def config_fingerprint(config: dict) -> str:
    """Stable short hash of a canonicalised config dict."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


class SweepStore:
    def __init__(self, cache_dir: str | Path, config: dict):
        self.config = config
        self.root = Path(cache_dir) / config_fingerprint(config)

    def chunk_path(self, method: str, alpha: float, gamma: float, chunk: int) -> Path:
        return self.root / f"{_slug(method)}__a{alpha:g}__g{gamma:g}__c{chunk:04d}.npz"

    def has(self, method: str, alpha: float, gamma: float, chunk: int) -> bool:
        return self.chunk_path(method, alpha, gamma, chunk).exists()

    def save(self, method: str, alpha: float, gamma: float, chunk: int,
             acc: StatsAccumulator) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.chunk_path(method, alpha, gamma, chunk)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:   # file handle stops np.savez appending ".npz"
            np.savez(f, **acc.to_dict())
        os.replace(tmp, path)  # atomic — a crash never leaves a torn chunk

    def load(self, method: str, alpha: float, gamma: float, chunk: int) -> StatsAccumulator:
        with np.load(self.chunk_path(method, alpha, gamma, chunk)) as data:
            return StatsAccumulator.from_dict({k: data[k].item() for k in data.files})

    def write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps(self.config, indent=2, sort_keys=True, default=str))

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
