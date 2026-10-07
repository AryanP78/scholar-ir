"""Net score = relevance + static quality (Scoring lecture, "net-score(q,d) = g(d) + cosine(q,d)").

We re-rank only the top-N relevance candidates (authority never creates candidates):
    R(q,d)   = min-max normalised BM25F score within the candidate set   (in [0,1])
    additive        net = (1 - lambda) * R + lambda * g(d)
    multiplicative  net = R * (1 + lambda * g(d))
Normalising R inside the candidate set puts relevance and g(d) on the same [0,1] scale, so lambda
is directly interpretable as "how much of the score is authority".
"""
from __future__ import annotations

import numpy as np


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    lo, hi = x.min(), x.max()
    return np.ones_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def net_score(R: np.ndarray, g: np.ndarray, lam: float, mode: str = "additive") -> np.ndarray:
    if mode == "additive":
        return (1.0 - lam) * R + lam * g
    if mode == "multiplicative":
        return R * (1.0 + lam * g)
    raise ValueError(f"unknown net-score mode {mode!r}")
