"""Ranked retrieval: tf-idf cosine (SMART lnc.ltc), BM25, BM25F, weighted zone scoring, heap top-K.

All scorers are term-at-a-time: for each query term they walk its postings list and add the term's
contribution into a score accumulator. Only documents that appear in some query term's postings ever
receive a score — the collection is never scanned. An optional candidate set A (from Boolean
constraints, filters, champion lists or index elimination) restricts accumulation to A by looking
each doc of A up in the postings list (binary search), which costs O(|A| log L) instead of O(L).
"""
from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .index import Index, ZoneIndex
from .query_parser import ParsedQuery, positive_clauses


@dataclass(frozen=True)
class QueryTerm:
    term: str     # analysed (stemmed) term
    zone: str     # all | title | abstract
    qtf: int      # frequency of the term in the query


def query_terms(index: Index, pq: ParsedQuery) -> list[QueryTerm]:
    """Positive (non-negated) terms of the parsed query, analysed with the index's analyzer."""
    counts: Counter = Counter()
    for leaf in positive_clauses(pq.root):
        for t in index.analyzer.terms(leaf.text):
            counts[(t, leaf.zone)] += 1
    return [QueryTerm(t, z, n) for (t, z), n in counts.items()]


def restrict(doc_ids: np.ndarray, tf: np.ndarray, cand: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Postings (doc_ids, tf) restricted to the sorted candidate array `cand` (None = no restriction)."""
    if cand is None or len(doc_ids) == 0:
        return doc_ids, tf
    if len(cand) == 0:
        return doc_ids[:0], tf[:0]
    pos = np.searchsorted(doc_ids, cand)
    ok = pos < len(doc_ids)
    ok[ok] = doc_ids[pos[ok]] == cand[ok]
    return cand[ok].astype(np.int32), tf[pos[ok]]


# ---- heap-based top-K (Scoring lecture) ----------------------------------------------------------

def heap_topk(doc_ids: Iterable[int], scores: Iterable[float], k: int) -> list[tuple[int, float]]:
    """Top-K by score with a size-K min-heap: O(J log K) for J scored docs. Ties -> lower doc id."""
    heap: list[tuple[float, int]] = []
    for d, s in zip(doc_ids, scores):
        item = (float(s), -int(d))
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heappushpop(heap, item)
    return [(-nd, s) for s, nd in sorted(heap, reverse=True)]


def topk_from_accumulator(acc: np.ndarray, touched: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Heap top-K over the docs that received a score (the 'J docs with nonzero score')."""
    docs = np.flatnonzero(touched)
    return heap_topk(docs.tolist(), acc[docs].tolist(), k)


# ---- SMART lnc.ltc cosine ------------------------------------------------------------------------

def tfidf_lnc_ltc(index: Index, qterms: list[QueryTerm], cand: np.ndarray | None = None,
                  detail: bool = False):
    """cos(q,d) with documents weighted lnc (1+log10 tf, no idf, cosine-normalised) and the query
    weighted ltc ((1+log10 qtf) * log10(N/df), cosine-normalised)."""
    N = index.N
    acc = np.zeros(N)
    touched = np.zeros(N, dtype=bool)
    weights = {}
    for qt in qterms:
        zi = index.zone(qt.zone)
        df = zi.get_df(qt.term)
        if df == 0:
            continue
        weights[qt] = (1 + math.log10(qt.qtf)) * math.log10(N / df)
    qnorm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    for qt, w in weights.items():
        wq = w / qnorm
        d, tf = restrict(*index.zone(qt.zone).postings(qt.term), cand)
        acc[d] += wq * (1 + np.log10(tf)) / index.lnc_norm[d]
        touched[d] = True
    if detail:
        return acc, touched, {qt.term + ("" if qt.zone == "all" else f"@{qt.zone}"): w / qnorm for qt, w in weights.items()}
    return acc, touched


# ---- BM25 -----------------------------------------------------------------------------------------

def bm25_idf(N: int, df: int) -> float:
    return math.log(1.0 + (N - df + 0.5) / (df + 0.5))


def bm25(index: Index, qterms: list[QueryTerm], k1: float, b: float, cand: np.ndarray | None = None,
         zone: str = "all", idf_fn: Callable[[int, int], float] = bm25_idf):
    """Okapi BM25 over one zone's postings (the flat `all` zone by default)."""
    N = index.N
    acc = np.zeros(N)
    touched = np.zeros(N, dtype=bool)
    for qt in qterms:
        zi = index.zone(qt.zone if qt.zone != "all" else zone)
        df = zi.get_df(qt.term)
        if df == 0:
            continue
        idf = idf_fn(N, df)
        d, tf = restrict(*zi.postings(qt.term), cand)
        tf = tf.astype(np.float64)
        norm = k1 * (1 - b + b * zi.doc_len[d] / zi.avgdl)
        acc[d] += qt.qtf * idf * tf * (k1 + 1) / (tf + norm)
        touched[d] = True
    return acc, touched


# ---- BM25F (zones combined before saturation) ----------------------------------------------------

def bm25f(index: Index, qterms: list[QueryTerm], k1: float, w: dict[str, float], b: dict[str, float],
          cand: np.ndarray | None = None, return_parts: bool = False):
    """BM25F: tf'_td = sum_z w_z * tf_tdz / (1 - b_z + b_z * dl_dz/avgdl_z);
    score = sum_t idf(t) * tf'_td / (k1 + tf'_td), idf from the document-level (`all`) df."""
    N = index.N
    acc = np.zeros(N)
    touched = np.zeros(N, dtype=bool)
    parts = {}
    for qt in qterms:
        df = index.zone("all").get_df(qt.term)
        if df == 0:
            continue
        idf = bm25_idf(N, df)
        zones = ("title", "abstract") if qt.zone == "all" else (qt.zone,)
        tfp = np.zeros(N)
        hit = np.zeros(N, dtype=bool)
        for z in zones:
            zi = index.zone(z)
            d, tf = restrict(*zi.postings(qt.term), cand)
            if len(d) == 0:
                continue
            tfp[d] += w[z] * tf / (1 - b[z] + b[z] * zi.doc_len[d] / zi.avgdl)
            hit[d] = True
        docs = np.flatnonzero(hit)
        contrib = qt.qtf * idf * tfp[docs] / (k1 + tfp[docs])
        acc[docs] += contrib
        touched[docs] = True
        if return_parts:
            parts[qt] = (docs, contrib)
    return (acc, touched, parts) if return_parts else (acc, touched)


def linear_zone(index: Index, qterms: list[QueryTerm], k1: float, b: dict[str, float], g: dict[str, float],
                cand: np.ndarray | None = None):
    """Weighted zone scoring: score = g_title * BM25_title + g_abstract * BM25_abstract."""
    acc = np.zeros(index.N)
    touched = np.zeros(index.N, dtype=bool)
    for z in ("title", "abstract"):
        zq = [q for q in qterms if q.zone in ("all", z)]
        a, t = bm25(index, [QueryTerm(q.term, "all", q.qtf) for q in zq], k1, b[z], cand, zone=z)
        acc += g[z] * a
        touched |= t
    return acc, touched
