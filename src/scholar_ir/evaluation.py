"""Evaluation metrics, pooling and significance tests (Evaluation lecture).

Relevance grades: 0 = irrelevant, 1 = relevant, 2 = highly relevant. Binary metrics treat
grade >= relevant_grade as relevant. Recall is measured against the POOLED relevant set (the union
of judged-relevant docs from the systems' top-15 lists); true recall over the whole corpus is
unknowable without exhaustive judging. Unjudged documents count as non-relevant.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

Qrels = dict[str, dict[int, int]]  # query_id -> {doc_id: grade}


def load_qrels(path) -> Qrels:
    df = pd.read_csv(path)
    q: Qrels = defaultdict(dict)
    for qid, d, g in zip(df["query_id"], df["doc_id"], df["grade"]):
        if pd.notna(g):
            q[str(qid)][int(d)] = int(g)
    return dict(q)


def precision_at(ranked: Sequence[int], rel: set[int], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at(ranked: Sequence[int], rel: set[int], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in rel) / len(rel) if rel else 0.0


def average_precision(ranked: Sequence[int], rel: set[int]) -> float:
    """AP = mean of precision at the rank of each relevant doc (relevant docs never retrieved add 0)."""
    if not rel:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / len(rel)


def reciprocal_rank(ranked: Sequence[int], rel: set[int]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: Sequence[int], grades: dict[int, int], k: int) -> float:
    """nDCG@k with gain = 2^grade - 1 and log2(rank+1) discount; ideal from all judged grades."""
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(ranked: Sequence[int], grades: dict[int, int], rel_grade: int = 1) -> dict[str, float]:
    rel = {d for d, g in grades.items() if g >= rel_grade}
    return {
        "P@5": precision_at(ranked, rel, 5), "P@10": precision_at(ranked, rel, 10),
        "R@10": recall_at(ranked, rel, 10), "R@20": recall_at(ranked, rel, 20),
        "MAP": average_precision(ranked, rel), "nDCG@10": ndcg_at(ranked, grades, 10),
        "MRR": reciprocal_rank(ranked, rel),
        "judged@10": sum(1 for d in ranked[:10] if d in grades) / 10,
    }


METRICS = ["P@5", "P@10", "R@10", "R@20", "MAP", "nDCG@10", "MRR"]


def bias_diagnostic(ranked: Sequence[int], years: np.ndarray, ref_year: int, window: int) -> dict[str, float]:
    top = [d for d in ranked[:10]]
    ys = years[top] if top else np.array([])
    return {"median_year@10": float(np.median(ys)) if len(ys) else float("nan"),
            f"frac_last{window}y@10": float(np.mean(ys > ref_year - window)) if len(ys) else float("nan")}


# ---- significance -------------------------------------------------------------------------------

def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap over queries for mean(a - b): two-sided p-value and 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    obs = d.mean()
    # p-value: how often a resampled mean (centred under H0) is as extreme as observed
    p = float(np.mean(np.abs(means - obs) >= abs(obs))) if len(d) else float("nan")
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_bootstrap": p}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import wilcoxon as _w
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(_w(a, b, zero_method="wilcox").pvalue)


# ---- agreement ----------------------------------------------------------------------------------

def cohens_kappa(x: Sequence[int], y: Sequence[int]) -> float:
    x, y = list(x), list(y)
    cats = sorted(set(x) | set(y))
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---- pooling ------------------------------------------------------------------------------------

def make_pool(runs: dict[str, dict[str, list[int]]], depth: int) -> dict[str, list[int]]:
    """Union of each system's top-`depth` docs per query (deduplicated, order-free)."""
    pool: dict[str, set[int]] = defaultdict(set)
    for sys_runs in runs.values():
        for qid, ranked in sys_runs.items():
            pool[qid].update(ranked[:depth])
    return {q: sorted(v) for q, v in pool.items()}


def mean_table(per_query: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in per_query.columns if c in METRICS or c.startswith("median_year") or c.startswith("frac_") or c == "judged@10"]
    return per_query.groupby(list(by))[cols].mean().round(4).reset_index()
