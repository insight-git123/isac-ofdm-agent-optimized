"""Small, dependency-light optimization utilities for the ISAC-OFDM pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Any
import numpy as np

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    workers: int = 1
    cache_size: int = 32

def rng(seed: int = 2026) -> np.random.Generator:
    """Return an isolated generator so experiments do not mutate global RNG state."""
    return np.random.default_rng(seed)

def summarize(values: Iterable[float]) -> dict[str, float]:
    """Vectorized summary used by scans and reports."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        raise ValueError("values must not be empty")
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95))}

def cached(func: Callable[..., Any], maxsize: int = 32) -> Callable[..., Any]:
    """Apply a bounded cache to pure parameter-to-result functions."""
    return lru_cache(maxsize=maxsize)(func)

def batch_run(experiment: Callable[[int], float], seeds: Iterable[int]) -> dict[str, float]:
    """Run scalar experiments and return stable aggregate statistics."""
    return summarize(experiment(int(s)) for s in seeds)
