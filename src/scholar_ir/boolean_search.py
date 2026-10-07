"""Boolean, phrase and proximity retrieval over the positional zone indexes (Boolean retrieval lecture).

* Postings intersection by linear merge, optionally with skip pointers at sqrt(len) spacing.
* Query optimisation: conjunctions are processed in order of increasing document frequency.
* Phrase queries by positional intersection: pos(t_i) = pos(t_1) + offset_i.
* NEAR/k by positional windows; parametric filters by intersection with year/category postings.
Every operation counts the postings entries it touches, so the efficiency effect of df-ordering
and skip pointers can be measured rather than asserted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .index import Index, ZoneIndex
from .query_parser import And, Near, Not, Or, ParsedQuery, Phrase, Seq, Term


@dataclass
class Stats:
    touched: int = 0          # postings entries read (merge steps)
    skips_taken: int = 0
    lists: list[tuple[str, int]] = field(default_factory=list)  # (term, df) in processing order


# ---- merge algorithms (pure Python, as in the lecture pseudocode) ---------------------------------

def intersect(p1: Sequence[int], p2: Sequence[int], stats: Stats | None = None) -> list[int]:
    """INTERSECT(p1, p2): linear merge of two sorted postings lists."""
    out, i, j = [], 0, 0
    n1, n2 = len(p1), len(p2)
    steps = 0
    while i < n1 and j < n2:
        steps += 1
        a, b = p1[i], p2[j]
        if a == b:
            out.append(a); i += 1; j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    if stats:
        stats.touched += steps
    return out


def intersect_with_skips(p1: Sequence[int], p2: Sequence[int], stats: Stats | None = None) -> list[int]:
    """Intersection with skip pointers every floor(sqrt(L)) entries (implicit pointers: i -> i+s)."""
    out, i, j = [], 0, 0
    n1, n2 = len(p1), len(p2)
    s1, s2 = max(1, int(math.sqrt(n1))), max(1, int(math.sqrt(n2)))
    steps = skips = 0
    while i < n1 and j < n2:
        steps += 1
        a, b = p1[i], p2[j]
        if a == b:
            out.append(a); i += 1; j += 1
        elif a < b:
            if i % s1 == 0 and i + s1 < n1 and p1[i + s1] <= b:
                while i % s1 == 0 and i + s1 < n1 and p1[i + s1] <= b:
                    i += s1; skips += 1; steps += 1
            else:
                i += 1
        else:
            if j % s2 == 0 and j + s2 < n2 and p2[j + s2] <= a:
                while j % s2 == 0 and j + s2 < n2 and p2[j + s2] <= a:
                    j += s2; skips += 1; steps += 1
            else:
                j += 1
    if stats:
        stats.touched += steps
        stats.skips_taken += skips
    return out


def union(p1: Sequence[int], p2: Sequence[int], stats: Stats | None = None) -> list[int]:
    out, i, j = [], 0, 0
    n1, n2 = len(p1), len(p2)
    while i < n1 and j < n2:
        a, b = p1[i], p2[j]
        if a == b:
            out.append(a); i += 1; j += 1
        elif a < b:
            out.append(a); i += 1
        else:
            out.append(b); j += 1
    out.extend(p1[i:]); out.extend(p2[j:])
    if stats:
        stats.touched += n1 + n2
    return out


def difference(p1: Sequence[int], p2: Sequence[int], stats: Stats | None = None) -> list[int]:
    """p1 AND NOT p2."""
    out, i, j = [], 0, 0
    n1, n2 = len(p1), len(p2)
    while i < n1:
        a = p1[i]
        while j < n2 and p2[j] < a:
            j += 1
        if j < n2 and p2[j] == a:
            i += 1
            continue
        out.append(a); i += 1
    if stats:
        stats.touched += n1 + j
    return out


def intersect_many(lists: list[list[int]], stats: Stats | None = None, order_by_df: bool = True,
                   use_skips: bool = False) -> list[int]:
    """AND of several postings lists; with order_by_df, start from the shortest (query optimisation)."""
    if not lists:
        return []
    lists = sorted(lists, key=len) if order_by_df else list(lists)
    fn = intersect_with_skips if use_skips else intersect
    result = lists[0]
    for nxt in lists[1:]:
        if not result:
            break
        result = fn(result, nxt, stats)
    return result


# ---- positional operators -------------------------------------------------------------------------

def _zone(index: Index, z: str) -> ZoneIndex:
    return index.zone(z if z in ("title", "abstract") else "all")


def phrase_docs(index: Index, text: str, zone: str, stats: Stats | None = None,
                return_positions: bool = False):
    """Docs (and optionally match start positions) where the analysed phrase terms occur at the
    query-side offsets. Single-term phrases reduce to a postings list."""
    zi = _zone(index, zone)
    terms = index.analyzer.phrase_terms(text)
    if not terms:
        return ([], {}) if return_positions else []
    lists = [zi.doc_list(t).tolist() for t, _ in terms]
    if stats is not None:
        stats.lists += [(t, len(l)) for (t, _), l in zip(terms, lists)]
    cands = intersect_many(lists, stats)
    if len(terms) == 1:
        if return_positions:
            return cands, {d: [(int(p), int(p)) for p in zi.positions_in(terms[0][0], d)] for d in cands}
        return cands
    out, spans = [], {}
    last_off = terms[-1][1]
    for d in cands:
        starts = set(zi.positions_in(terms[0][0], d).tolist())
        for t, off in terms[1:]:
            pos = zi.positions_in(t, d)
            if stats is not None:
                stats.touched += len(pos)
            starts &= {int(p) - off for p in pos}
            if not starts:
                break
        if starts:
            out.append(d)
            spans[d] = [(s, s + last_off) for s in sorted(starts)]
    return (out, spans) if return_positions else out


def _occurrences(index: Index, node, stats: Stats) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    """Docs plus occurrence spans (start, end) for a Term or Phrase node."""
    if isinstance(node, Term):
        return phrase_docs(index, node.text, node.zone, stats, return_positions=True)
    if isinstance(node, Phrase):
        return phrase_docs(index, node.text, node.zone, stats, return_positions=True)
    raise ValueError("NEAR operands must be words or phrases")


def smallest_window(position_lists: list[list[int]]) -> int | None:
    """Smallest window (max-min) containing one occurrence of every list (Query term proximity)."""
    if not position_lists or any(len(p) == 0 for p in position_lists):
        return None
    events = sorted((p, i) for i, ps in enumerate(position_lists) for p in ps)
    need, have, counts = len(position_lists), 0, [0] * len(position_lists)
    best, lo = None, 0
    for hi, (p, i) in enumerate(events):
        if counts[i] == 0:
            have += 1
        counts[i] += 1
        while have == need:
            w = p - events[lo][0]
            best = w if best is None or w < best else best
            j = events[lo][1]
            counts[j] -= 1
            if counts[j] == 0:
                have -= 1
            lo += 1
    return best


def near_docs(index: Index, node: Near, stats: Stats) -> list[int]:
    if node.right is None:  # "a b c" NEAR/k : the words within k positions, any order
        text = node.left.text if isinstance(node.left, (Term, Phrase)) else ""
        zone = getattr(node.left, "zone", "all")
        zi = _zone(index, zone)
        terms = [t for t, _ in index.analyzer.phrase_terms(text)]
        cands = intersect_many([zi.doc_list(t).tolist() for t in terms], stats)
        out = []
        for d in cands:
            w = smallest_window([zi.positions_in(t, d).tolist() for t in terms])
            if w is not None and w <= node.k + max(0, len(terms) - 2):
                out.append(d)
        return out
    ld, lspans = _occurrences(index, node.left, stats)
    rd, rspans = _occurrences(index, node.right, stats)
    out = []
    for d in intersect(ld, rd, stats):
        ok = any(max(s1, s2) - min(e1, e2) <= node.k
                 for s1, e1 in lspans[d] for s2, e2 in rspans[d])
        if ok:
            out.append(d)
    return out


# ---- AST evaluation -------------------------------------------------------------------------------

def evaluate(index: Index, node, stats: Stats | None = None, order_by_df: bool = True,
             use_skips: bool = False) -> list[int]:
    """Evaluate a Boolean AST to a sorted list of doc ids."""
    stats = stats or Stats()
    if node is None:
        return list(range(index.N))
    if isinstance(node, Term):
        zi = _zone(index, node.zone)
        terms = index.analyzer.terms(node.text)
        if not terms:
            return []
        lists = [zi.doc_list(t).tolist() for t in terms]
        stats.lists += [(t, len(l)) for t, l in zip(terms, lists)]
        # a hyphenated/multi-token word behaves like a phrase of its parts
        return lists[0] if len(lists) == 1 else phrase_docs(index, node.text, node.zone, stats)
    if isinstance(node, Phrase):
        return phrase_docs(index, node.text, node.zone, stats)
    if isinstance(node, Near):
        return near_docs(index, node, stats)
    if isinstance(node, Not):
        return difference(list(range(index.N)), evaluate(index, node.child, stats, order_by_df, use_skips), stats)
    if isinstance(node, And):
        return intersect_many([evaluate(index, c, stats, order_by_df, use_skips) for c in node.children],
                              stats, order_by_df, use_skips)
    if isinstance(node, Or):
        out: list[int] = []
        for c in node.children:
            out = union(out, evaluate(index, c, stats, order_by_df, use_skips), stats)
        return out
    if isinstance(node, Seq):
        required = [c for c in node.children if not isinstance(c, (Term, Not))]
        optional = [c for c in node.children if isinstance(c, Term)]
        excluded = [c.child for c in node.children if isinstance(c, Not)]
        if required:
            res = intersect_many([evaluate(index, c, stats, order_by_df, use_skips) for c in required],
                                 stats, order_by_df, use_skips)
        elif optional:
            res = []
            for c in optional:
                res = union(res, evaluate(index, c, stats, order_by_df, use_skips), stats)
        else:
            res = list(range(index.N))
        for c in excluded:
            res = difference(res, evaluate(index, c, stats, order_by_df, use_skips), stats)
        return res
    raise TypeError(f"unknown node {node!r}")


def filter_docs(index: Index, pq: ParsedQuery) -> np.ndarray | None:
    """Parametric-index filter as a sorted doc-id array (None = no filter)."""
    f = pq.filters
    if f.is_empty():
        return None
    sets = []
    if f.year_min is not None or f.year_max is not None:
        sets.append(index.year_range_docs(f.year_min, f.year_max))
    if f.categories:
        sets.append(index.category_docs(f.categories))
    out = sets[0]
    for s in sets[1:]:
        out = np.intersect1d(out, s, assume_unique=True)
    return out


def boolean_search(index: Index, pq: ParsedQuery, order_by_df: bool = True, use_skips: bool = False
                   ) -> tuple[list[int], Stats]:
    """Unranked Boolean retrieval: AST evaluation intersected with parametric filters."""
    stats = Stats()
    docs = evaluate(index, pq.root, stats, order_by_df, use_skips) if pq.root is not None else list(range(index.N))
    flt = filter_docs(index, pq)
    if flt is not None:
        docs = intersect(docs, flt.tolist(), stats)
    return docs, stats
