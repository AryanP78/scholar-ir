"""Static quality g(d) in [0,1] from citations (Scoring lecture: "static quality scores").

Raw citation counts grow with age, so blending them into the score favours old papers regardless
of relevance. Variants (c = citations, age = max(1, Y - year + 1), Y = reference year):

  A0 raw          g0 = log(1+c) / log(1+c_max)
  A1 per-year     g1 = log(1+c/age) / log(1+max(c/age))
  A2 cohort       g2 = mid-rank percentile of c among papers first submitted in the SAME year
                       (adjacent years pooled until the cohort has >= min_cohort_size papers)
  A3 PageRank     PageRank on the in-corpus citation graph, then the same cohort percentile

Papers without citation data are excluded from cohort statistics and receive the cohort median
(0.5) or 0, per config `citations.missing_policy`; they are flagged in `has_citation_data`.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger("authority")


def raw_log(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    top = math.log1p(c.max()) if len(c) and c.max() > 0 else 1.0
    return np.log1p(c) / top


def per_year(c: np.ndarray, years: np.ndarray, ref_year: int) -> np.ndarray:
    age = np.maximum(1, ref_year - np.asarray(years) + 1)
    rate = np.asarray(c, dtype=np.float64) / age
    top = math.log1p(rate.max()) if len(rate) and rate.max() > 0 else 1.0
    return np.log1p(rate) / top


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    """(#less + 0.5 * #equal) / n for every value — ties share the mid-rank."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return v
    srt = np.sort(v)
    less = np.searchsorted(srt, v, side="left")
    leq = np.searchsorted(srt, v, side="right")
    return (less + 0.5 * (leq - less)) / len(v)


def cohorts(years: np.ndarray, valid: np.ndarray, min_size: int) -> dict[int, list[int]]:
    """Map each year to the list of years pooled into its cohort (grow symmetrically until large enough)."""
    uniq = sorted(set(int(y) for y in years))
    counts = {y: int(((years == y) & valid).sum()) for y in uniq}
    out = {}
    for y in uniq:
        pool, lo, hi = [y], y, y
        while sum(counts.get(t, 0) for t in pool) < min_size:
            cand = []
            if lo - 1 >= uniq[0]:
                cand.append(lo - 1)
            if hi + 1 <= uniq[-1]:
                cand.append(hi + 1)
            if not cand:
                break
            for t in cand:
                pool.append(t)
            lo, hi = min(pool), max(pool)
        out[y] = sorted(pool)
    return out


def cohort_percentile(values: np.ndarray, years: np.ndarray, valid: np.ndarray, min_size: int,
                      missing_value: float = 0.5) -> np.ndarray:
    """Percentile of each paper's value within its publication-year cohort (A2)."""
    values = np.asarray(values, dtype=np.float64)
    years = np.asarray(years)
    valid = np.asarray(valid, dtype=bool)
    g = np.full(len(values), missing_value, dtype=np.float64)
    for y, pool in cohorts(years, valid, min_size).items():
        ref = values[np.isin(years, pool) & valid]
        target = (years == y) & valid
        if not target.any() or len(ref) == 0:
            continue
        srt = np.sort(ref)
        v = values[target]
        less = np.searchsorted(srt, v, side="left")
        leq = np.searchsorted(srt, v, side="right")
        g[target] = (less + 0.5 * (leq - less)) / len(ref)
    return g


def pagerank(n: int, edges_src: np.ndarray, edges_dst: np.ndarray, damping: float = 0.85,
             tol: float = 1e-10, max_iter: int = 100) -> tuple[np.ndarray, int]:
    """PageRank by power iteration on a citing -> cited graph. Dangling nodes (no in-corpus
    references) spread their mass uniformly. Returns (scores summing to 1, iterations)."""
    src = np.asarray(edges_src, dtype=np.int64)
    dst = np.asarray(edges_dst, dtype=np.int64)
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    dangling = outdeg == 0
    pr = np.full(n, 1.0 / n)
    w = np.zeros(len(src))
    if len(src):
        w = 1.0 / outdeg[src]
    it = 0
    for it in range(1, max_iter + 1):
        new = np.bincount(dst, weights=pr[src] * w, minlength=n) if len(src) else np.zeros(n)
        new = damping * (new + pr[dangling].sum() / n) + (1 - damping) / n
        delta = np.abs(new - pr).sum()
        pr = new
        if delta < tol:
            break
    return pr, it


def compute_authority(papers: pd.DataFrame, cites: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """All authority variants as columns aligned with doc_id."""
    acfg = cfg["authority"]
    df = papers[["doc_id", "year"]].merge(cites, on="doc_id", how="left")
    df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)
    df["has_citation_data"] = df["has_citation_data"].fillna(False).astype(bool)
    ref_year = int(df["year"].max())
    c = df["cited_by_count"].to_numpy()
    years = df["year"].to_numpy()
    valid = df["has_citation_data"].to_numpy()
    missing = 0.5 if cfg["citations"]["missing_policy"] == "median" else 0.0
    g0 = raw_log(c)
    g1 = per_year(c, years, ref_year)
    if missing == 0.0:
        g0 = np.where(valid, g0, 0.0)
        g1 = np.where(valid, g1, 0.0)
    else:  # median of the valid papers, so missing papers are neither boosted nor buried
        g0 = np.where(valid, g0, np.median(g0[valid]) if valid.any() else 0.0)
        g1 = np.where(valid, g1, np.median(g1[valid]) if valid.any() else 0.0)
    g2 = cohort_percentile(c, years, valid, acfg["min_cohort_size"], missing)
    out = pd.DataFrame({"doc_id": df["doc_id"], "year": years, "cited_by_count": c, "has_citation_data": valid,
                        "g_raw": g0, "g_per_year": g1, "g_cohort": g2})
    refs = df["referenced_doc_ids"] if "referenced_doc_ids" in df else pd.Series([[]] * len(df))
    src = np.concatenate([np.full(len(r), d) for d, r in zip(df["doc_id"], refs) if r is not None and len(r)]) \
        if any(r is not None and len(r) for r in refs) else np.empty(0, dtype=np.int64)
    dst = np.concatenate([np.asarray(r) for r in refs if r is not None and len(r)]) if len(src) else np.empty(0, dtype=np.int64)
    n = len(df)
    pr, iters = pagerank(n, src, dst, acfg["pagerank_damping"], acfg["pagerank_tol"], acfg["pagerank_max_iter"])
    out["pagerank"] = pr
    out["g_pagerank_cohort"] = cohort_percentile(pr, years, valid, acfg["min_cohort_size"], missing)
    out.attrs.update({"reference_year": ref_year, "pagerank_iterations": iters, "graph_edges": int(len(src))})
    log.info("authority: ref_year=%d, coverage=%.1f%%, graph edges=%d, PageRank iters=%d",
             ref_year, 100 * valid.mean(), len(src), iters)
    return out


AUTHORITY_COLUMNS = {"A0": "g_raw", "A1": "g_per_year", "A2": "g_cohort", "A3": "g_pagerank_cohort"}
