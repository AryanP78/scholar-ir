"""Inverted, positional, zone and parametric indexes, built from scratch.

Layout of one ZoneIndex (the classic "dictionary + postings" split, stored as flat arrays):

    dictionary   vocab: term -> term_id ; df[term_id] ; ptr[term_id] (pointer into the postings arrays)
    postings     doc_ids[ptr[t]:ptr[t+1]]   sorted doc ids of term t
                 tf[ptr[t]:ptr[t+1]]        term frequency of t in each doc
                 positions[pos_ptr[i]:pos_ptr[i+1]]  token positions of posting i  (positional index)
    lengths      doc_len[d] (indexed tokens in this zone), avgdl, N

Zones: `title`, `abstract`, and a flat `all` view (title then abstract, with a positional gap so that
phrases never straddle the zone boundary). Parametric indexes map year -> sorted doc ids and
category -> sorted doc ids. Skip pointers are implicit at sqrt(len) spacing (Term vocabulary and
postings lecture); variable-byte gap compression is implemented and measured (size report only).
"""
from __future__ import annotations

import logging
import math
import pickle
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .preprocess import Analyzer

log = logging.getLogger("index")

ZONE_GAP = 50  # positional gap between title and abstract in the flat `all` zone


@dataclass
class ZoneIndex:
    """Positional inverted index for one zone."""
    name: str
    N: int
    vocab: dict[str, int]
    terms: list[str]
    df: np.ndarray        # int32 [T]
    ptr: np.ndarray       # int64 [T+1]
    doc_ids: np.ndarray   # int32 [P]
    tf: np.ndarray        # int32 [P]
    pos_ptr: np.ndarray   # int64 [P+1]
    positions: np.ndarray  # int32 [total tokens]
    doc_len: np.ndarray   # int32 [N]
    avgdl: float

    # ---- dictionary lookups -------------------------------------------------------------------
    def term_id(self, term: str) -> int | None:
        return self.vocab.get(term)

    def get_df(self, term: str) -> int:
        t = self.vocab.get(term)
        return int(self.df[t]) if t is not None else 0

    def span(self, term: str) -> tuple[int, int]:
        t = self.vocab.get(term)
        if t is None:
            return 0, 0
        return int(self.ptr[t]), int(self.ptr[t + 1])

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """(doc_ids, tf) views of the postings list of `term` (empty if not in the dictionary)."""
        a, b = self.span(term)
        return self.doc_ids[a:b], self.tf[a:b]

    def doc_list(self, term: str) -> np.ndarray:
        a, b = self.span(term)
        return self.doc_ids[a:b]

    def positions_in(self, term: str, doc_id: int) -> np.ndarray:
        """Positions of `term` in `doc_id` (binary search inside the postings list)."""
        a, b = self.span(term)
        if a == b:
            return np.empty(0, dtype=np.int32)
        i = a + int(np.searchsorted(self.doc_ids[a:b], doc_id))
        if i >= b or self.doc_ids[i] != doc_id:
            return np.empty(0, dtype=np.int32)
        return self.positions[self.pos_ptr[i]:self.pos_ptr[i + 1]]

    def raw_postings(self, term: str, limit: int | None = None) -> list[tuple[int, int, list[int]]]:
        """Human-readable postings: [(doc_id, tf, [positions]), ...] — used in the demo."""
        a, b = self.span(term)
        if limit is not None:
            b = min(b, a + limit)
        return [(int(self.doc_ids[i]), int(self.tf[i]),
                 self.positions[self.pos_ptr[i]:self.pos_ptr[i + 1]].tolist()) for i in range(a, b)]

    def idf_ln(self, term: str) -> float:
        """BM25 idf: ln(1 + (N - df + 0.5)/(df + 0.5))."""
        df = self.get_df(term)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def idf_log10(self, term: str) -> float:
        df = self.get_df(term)
        return math.log10(self.N / df) if df else 0.0

    @property
    def n_postings(self) -> int:
        return int(len(self.doc_ids))

    def nbytes(self) -> int:
        return int(self.df.nbytes + self.ptr.nbytes + self.doc_ids.nbytes + self.tf.nbytes
                   + self.pos_ptr.nbytes + self.positions.nbytes + self.doc_len.nbytes)


def build_zone_index(name: str, token_stream: Iterable[list[tuple[str, int]]], N: int) -> ZoneIndex:
    """Sort-based index construction: collect (term_id, doc_id, position) triples, sort, then cut
    the sorted run into postings (one per (term, doc)) and the postings into per-term lists."""
    vocab: dict[str, int] = {}
    tids, docs, poss = array("i"), array("i"), array("i")
    doc_len = np.zeros(N, dtype=np.int32)
    for d, toks in enumerate(token_stream):
        doc_len[d] = len(toks)
        for term, pos in toks:
            t = vocab.get(term)
            if t is None:
                t = vocab[term] = len(vocab)
            tids.append(t); docs.append(d); poss.append(pos)
    tid = np.frombuffer(tids, dtype=np.int32)
    doc = np.frombuffer(docs, dtype=np.int32)
    pos = np.frombuffer(poss, dtype=np.int32)
    # Re-number terms alphabetically so the dictionary is sorted (like a real lexicon).
    terms_sorted = sorted(vocab)
    remap = np.empty(len(vocab), dtype=np.int32)
    for new, term in enumerate(terms_sorted):
        remap[vocab[term]] = new
    tid = remap[tid] if len(tid) else tid
    order = np.lexsort((pos, doc, tid))
    tid, doc, pos = tid[order], doc[order], pos[order]
    M = len(tid)
    if M:
        change = np.empty(M, dtype=bool)
        change[0] = True
        change[1:] = (tid[1:] != tid[:-1]) | (doc[1:] != doc[:-1])
        starts = np.flatnonzero(change)
    else:
        starts = np.empty(0, dtype=np.int64)
    pos_ptr = np.append(starts, M).astype(np.int64)
    p_doc = doc[starts].astype(np.int32)
    p_tid = tid[starts]
    tf = np.diff(pos_ptr).astype(np.int32)
    T = len(terms_sorted)
    ptr = np.searchsorted(p_tid, np.arange(T + 1)).astype(np.int64)
    df = np.diff(ptr).astype(np.int32)
    avgdl = float(doc_len.mean()) if N else 0.0
    return ZoneIndex(name=name, N=N, vocab={t: i for i, t in enumerate(terms_sorted)}, terms=terms_sorted,
                     df=df, ptr=ptr, doc_ids=p_doc, tf=tf, pos_ptr=pos_ptr, positions=pos.astype(np.int32),
                     doc_len=doc_len, avgdl=avgdl)


# ---- variable-byte gap compression (Index compression; measured, not used at query time) ----------

def vb_encode_number(n: int) -> bytes:
    out = []
    while True:
        out.insert(0, n % 128)
        if n < 128:
            break
        n //= 128
    out[-1] += 128
    return bytes(out)


def vb_encode(numbers: Sequence[int]) -> bytes:
    return b"".join(vb_encode_number(int(n)) for n in numbers)


def vb_decode(stream: bytes) -> list[int]:
    nums, n = [], 0
    for byte in stream:
        if byte < 128:
            n = 128 * n + byte
        else:
            nums.append(128 * n + byte - 128)
            n = 0
    return nums


def compressed_size(zi: ZoneIndex) -> dict[str, int]:
    """Bytes for doc-id postings: raw int32 vs gap + variable-byte encoding (vectorised byte count)."""
    gaps = zi.doc_ids.astype(np.int64).copy()
    starts = zi.ptr[:-1][zi.df > 0]
    inner = np.ones(len(gaps), dtype=bool)
    inner[starts] = False
    gaps[1:][inner[1:]] = np.diff(zi.doc_ids.astype(np.int64))[inner[1:]]
    # bytes needed per number in VB: ceil(bits/7), min 1
    nbytes = np.maximum(1, np.ceil(np.log2(gaps + 1) / 7)).astype(np.int64)
    return {"docid_raw_int32_bytes": int(zi.doc_ids.nbytes), "docid_gap_vb_bytes": int(nbytes.sum()),
            "tf_raw_int32_bytes": int(zi.tf.nbytes),
            "tf_vb_bytes": int(np.maximum(1, np.ceil(np.log2(zi.tf.astype(np.int64) + 1) / 7)).sum())}


# ---- the full index -------------------------------------------------------------------------------

@dataclass
class Index:
    zones: dict[str, ZoneIndex]
    N: int
    doc_year: np.ndarray                  # int16 [N]
    doc_cats: list[list[str]]
    arxiv_ids: list[str]
    year_postings: dict[int, np.ndarray]  # parametric index
    cat_postings: dict[str, np.ndarray]   # parametric index
    lnc_norm: np.ndarray                  # float64 [N], L2 norm of (1+log10 tf) doc vectors in `all`
    champions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    analyzer: Analyzer = field(default_factory=Analyzer)
    build_info: dict = field(default_factory=dict)

    def zone(self, name: str) -> ZoneIndex:
        return self.zones[name]

    # parametric filters --------------------------------------------------------------------------
    def year_range_docs(self, y_min: int | None, y_max: int | None) -> np.ndarray:
        ys = [y for y in self.year_postings if (y_min is None or y >= y_min) and (y_max is None or y <= y_max)]
        if not ys:
            return np.empty(0, dtype=np.int32)
        return np.sort(np.concatenate([self.year_postings[y] for y in ys]))

    def category_docs(self, cats: Sequence[str]) -> np.ndarray:
        arrs = [self.cat_postings[c] for c in cats if c in self.cat_postings]
        if not arrs:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(arrs))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "Index":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def zone_tokens(papers, analyzer: Analyzer) -> tuple[list, list, list]:
    """Analyze every document once; build the flat `all` stream from the two zones."""
    title_toks, abs_toks, all_toks = [], [], []
    for title, abstract in zip(papers["title"], papers["abstract"]):
        tt = analyzer.analyze(title)
        at = analyzer.analyze(abstract)
        offset = (max((p for _, p in tt), default=-1) + 1) + ZONE_GAP
        title_toks.append(tt)
        abs_toks.append(at)
        all_toks.append(tt + [(t, p + offset) for t, p in at])
    return title_toks, abs_toks, all_toks


def champion_lists(zi: ZoneIndex, r: int) -> dict[str, np.ndarray]:
    """Champion list per term: the r postings with highest tf (ties -> lower doc id), stored sorted
    by doc id. Terms with df <= r need no champion list (the full list is the champion list)."""
    champs: dict[str, np.ndarray] = {}
    for t in np.flatnonzero(zi.df > r):
        a, b = int(zi.ptr[t]), int(zi.ptr[t + 1])
        tf = zi.tf[a:b]
        # stable top-r by tf: sort by (-tf, doc order)
        top = np.argsort(-tf, kind="stable")[:r]
        champs[zi.terms[t]] = np.sort(zi.doc_ids[a:b][top])
    return champs


def build_index(papers, cfg: dict, analyzer: Analyzer | None = None, with_champions: bool = True) -> Index:
    """Build all indexes for a papers DataFrame (doc_id must equal the row number)."""
    analyzer = analyzer or Analyzer.from_config(cfg)
    N = len(papers)
    assert (papers["doc_id"].to_numpy() == np.arange(N)).all(), "doc_id must be 0..N-1 in order"
    t0 = time.time()
    title_toks, abs_toks, all_toks = zone_tokens(papers, analyzer)
    log.info("analysis: %.1fs", time.time() - t0)
    zones = {}
    for name, stream in (("title", title_toks), ("abstract", abs_toks), ("all", all_toks)):
        t1 = time.time()
        zones[name] = build_zone_index(name, stream, N)
        log.info("zone %-8s: %7d terms, %9d postings (%.1fs)", name, len(zones[name].terms),
                 zones[name].n_postings, time.time() - t1)
    allz = zones["all"]
    w = 1.0 + np.log10(allz.tf.astype(np.float64))
    lnc_norm = np.sqrt(np.bincount(allz.doc_ids, weights=w * w, minlength=N))
    lnc_norm[lnc_norm == 0] = 1.0
    doc_year = papers["year"].to_numpy().astype(np.int16)
    year_postings = {int(y): np.flatnonzero(doc_year == y).astype(np.int32) for y in np.unique(doc_year)}
    doc_cats = [list(c) for c in papers["categories"]]
    cat_lists: dict[str, list[int]] = {}
    for d, cats in enumerate(doc_cats):
        for c in cats:
            cat_lists.setdefault(c, []).append(d)
    cat_postings = {c: np.asarray(v, dtype=np.int32) for c, v in cat_lists.items()}
    idx = Index(zones=zones, N=N, doc_year=doc_year, doc_cats=doc_cats, arxiv_ids=list(papers["arxiv_id"]),
                year_postings=year_postings, cat_postings=cat_postings, lnc_norm=lnc_norm, analyzer=analyzer)
    if with_champions:
        r = cfg["index"]["champion_r"]
        idx.champions = {z: champion_lists(zones[z], r) for z in ("all", "title", "abstract")}
    idx.build_info = {"build_seconds": round(time.time() - t0, 1), "N": N,
                      "analyzer": {"stem": analyzer.stem, "stopwords": analyzer.stopwords}}
    return idx
