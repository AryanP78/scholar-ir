"""Inexact top-K: champion lists and index elimination (Scoring lecture, Sec. 7.1.2-7.1.3).

Both produce a contender set A (K < |A| << N); the rankers then compute exact scores for docs in A
only. Their cost/quality trade-off is measured in experiments.py (recall@10 vs. the exact ranking,
and latency).
"""
from __future__ import annotations

import numpy as np

from .index import Index
from .ranking import QueryTerm


def champion_candidates(index: Index, qterms: list[QueryTerm]) -> np.ndarray:
    """A = union over query terms of the term's champion list (top-r postings by tf). Terms whose
    df <= r contribute their whole postings list (it *is* their champion list)."""
    parts = []
    for qt in qterms:
        z = qt.zone
        champs = index.champions.get(z, {})
        if qt.term in champs:
            parts.append(champs[qt.term])
        else:
            parts.append(index.zone(z).doc_list(qt.term))
    if not parts:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(parts)).astype(np.int32)


def eliminate_terms(index: Index, qterms: list[QueryTerm], idf_threshold: float) -> list[QueryTerm]:
    """High-idf query terms only: drop terms whose idf (ln, BM25 form) is below the threshold.
    Never drops every term — the highest-idf term is always kept."""
    if not qterms:
        return qterms
    idfs = [index.zone(q.zone).idf_ln(q.term) for q in qterms]
    kept = [q for q, i in zip(qterms, idfs) if i >= idf_threshold]
    return kept or [qterms[int(np.argmax(idfs))]]


def many_terms_candidates(index: Index, qterms: list[QueryTerm], min_fraction: float) -> np.ndarray:
    """Docs containing at least ceil(min_fraction * n) of the n query terms ("3 of 4" soft AND)."""
    n = len(qterms)
    need = max(1, int(np.ceil(min_fraction * n)))
    lists = [index.zone(q.zone).doc_list(q.term) for q in qterms]
    lists = [l for l in lists if len(l)]
    if not lists:
        return np.empty(0, dtype=np.int32)
    allp = np.concatenate(lists)
    counts = np.bincount(allp, minlength=index.N)
    return np.flatnonzero(counts >= need).astype(np.int32)
