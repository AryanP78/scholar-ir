"""Topic-local citation authority with HITS (Kleinberg 1999) — the final-project novelty.

Global citation counts reward fame: a paper with 13,000 citations outranks the papers that matter for
the query even when it is only loosely on topic. Here authority is computed PER QUERY, on the citation
graph around that query's own results, exactly as HITS does for web pages:

  1. root set  R  = the top-N BM25F candidates for the query (N = ranking.candidate_pool, 200);
  2. base set  B  = R ∪ every paper a root paper cites (inside OR outside our 80k corpus)
                    ∪ up to `max_citers` corpus papers that cite each root paper;
  3. graph     A  = citation edges u -> v with u, v in B (only corpus papers have known reference lists),
                    each weighted by w(u) = how relevant the CITING paper u is to the query (its BM25F
                    score / the best root score, clipped to [0, 1]); this is the relevance-weighted HITS
                    of Bharat & Henzinger (SIGIR 1998) that stops "topic drift": papers pulled into the
                    base set from a neighbouring field cannot vote a famous off-topic paper up;
  4. HITS          a <- Aᵀh,  h <- A a,  each L2-normalised, until the change is below `tol`;
  5. g_topic(d)    = rank percentile of a(d) among the root papers (0 when nobody in B cites d),
     g_hub(d)      = the same for h(d) (hubs are papers that cite many good authorities: surveys).

A root paper is a topic authority when the papers *about this query* cite it, so a famous paper from a
neighbouring field gains nothing unless the topic's own literature points to it.

Data: data/processed/refs_full.parquet (doc_id, openalex_id, refs) from scripts/export_refs.py —
the full OpenAlex reference lists, including references to papers outside the corpus.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger("hits")


@dataclass
class HITSResult:
    authority: np.ndarray      # a(d) for each root doc (L2-normalised over the base set)
    hub: np.ndarray            # h(d) for each root doc
    base_size: int
    edges: int
    iterations: int


class CitationGraph:
    """Whole-corpus citation graph in CSR form. Nodes 0..N-1 are corpus docs (doc_id); nodes >= N are
    papers outside the corpus that some corpus paper cites."""

    def __init__(self, n_docs: int, refs: pd.DataFrame, max_citers: int = 100):
        self.N = n_docs
        self.max_citers = max_citers
        oa_to_doc = {int(o): int(d) for d, o in zip(refs["doc_id"], refs["openalex_id"]) if pd.notna(o)}
        ext: dict[int, int] = {}
        src, dst = [], []
        for d, rl in zip(refs["doc_id"].to_numpy(), refs["refs"]):
            if rl is None or len(rl) == 0:
                continue
            for w in rl:
                w = int(w)
                v = oa_to_doc.get(w)
                if v is None:
                    v = ext.setdefault(w, n_docs + len(ext))
                if v != d:
                    src.append(int(d))
                    dst.append(v)
        self.n_nodes = n_docs + len(ext)
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        ones = np.ones(len(src), dtype=np.float32)
        self.out = sparse.csr_matrix((ones, (src, dst)), shape=(self.n_nodes, self.n_nodes))
        self.out.data[:] = 1.0  # duplicate references count once
        self.inc = self.out.T.tocsr()  # inc[v] = corpus papers citing v
        self.n_edges = int(self.out.nnz)
        self.n_external = len(ext)
        self.internal_edges = int((dst < n_docs).sum())
        log.info("citation graph: %d corpus docs, %d outside papers, %d edges (%d inside the corpus)",
                 n_docs, self.n_external, self.n_edges, self.internal_edges)

    @classmethod
    def load(cls, path: Path, n_docs: int, max_citers: int = 100) -> "CitationGraph":
        return cls(n_docs, pd.read_parquet(path), max_citers)

    # ---- per-query HITS -------------------------------------------------------------------------
    def base_set(self, root: np.ndarray) -> np.ndarray:
        parts = [np.asarray(root, dtype=np.int64)]
        for d in root:
            parts.append(self.out.indices[self.out.indptr[d]:self.out.indptr[d + 1]])
            cit = self.inc.indices[self.inc.indptr[d]:self.inc.indptr[d + 1]]
            if len(cit) > self.max_citers:   # Kleinberg's cap on in-links per root page (deterministic)
                cit = np.sort(cit)[: self.max_citers]
            parts.append(cit)
        return np.unique(np.concatenate(parts))

    def hits(self, root: np.ndarray, weight: np.ndarray | None = None, max_iter: int = 100,
             tol: float = 1e-8, exclude: int | None = None) -> HITSResult:
        """weight: optional relevance of every corpus doc to the query (length N, any scale >= 0);
        None = classic unweighted HITS. exclude: a doc removed from the base set (used by the automatic
        citation-recommendation test so a held-out paper's own references cannot vote)."""
        root = np.asarray(root, dtype=np.int64)
        B = self.base_set(root)
        if exclude is not None:
            B = B[B != exclude]
        A = self.out[B][:, B].tocsr()              # sub-graph induced by the base set
        if weight is not None:
            w = np.zeros(len(B))
            inside = B < self.N
            w[inside] = np.asarray(weight, dtype=np.float64)[B[inside]]
            top = float(np.max(np.asarray(weight, dtype=np.float64)[root])) if len(root) else 0.0
            w = np.clip(w / top, 0.0, 1.0) if top > 0 else w
            A = sparse.diags(w) @ A                # row u scaled by the citing paper's relevance
            A.eliminate_zeros()
        n = len(B)
        if A.nnz == 0 or n == 0:
            z = np.zeros(len(root))
            return HITSResult(z, z.copy(), n, 0, 0)
        AT = A.T.tocsr()
        h = np.ones(n) / np.sqrt(n)
        a = np.zeros(n)
        it = 0
        for it in range(1, max_iter + 1):
            a_new = AT @ h
            na = np.linalg.norm(a_new)
            a_new = a_new / na if na > 0 else a_new
            h_new = A @ a_new
            nh = np.linalg.norm(h_new)
            h_new = h_new / nh if nh > 0 else h_new
            delta = np.abs(a_new - a).sum() + np.abs(h_new - h).sum()
            a, h = a_new, h_new
            if delta < tol:
                break
        pos = np.searchsorted(B, root)
        ok = (pos < n) & (B[np.minimum(pos, n - 1)] == root)
        return HITSResult(np.where(ok, a[np.minimum(pos, n - 1)], 0.0), np.where(ok, h[np.minimum(pos, n - 1)], 0.0),
                          n, int(A.nnz), it)


def percentile_score(x: np.ndarray) -> np.ndarray:
    """Rank percentile in (0, 1] among positive values; exactly 0 for zero (nobody in the base set
    cites the paper). Ties share their average rank."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(x))
    pos = x > 0
    if pos.any():
        out[pos] = pd.Series(x[pos]).rank(method="average", pct=True).to_numpy()
    return out
