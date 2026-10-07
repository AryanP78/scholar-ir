"""The search engine: query parser -> Boolean/filters -> ranker -> (authority net score) -> heap top-K.

Systems compared in the evaluation (config-driven; see SYSTEMS):
    S1  flat tf-idf cosine, SMART lnc.ltc            (baseline)
    S2  flat BM25
    S3  BM25F over title/abstract zones
    S3L weighted (linear) zone scoring, ablation of S3
    S4  S3 + raw citation authority A0, fixed lambda
    S5  S3 + per-year authority A1, fixed lambda
    S6  S3 + cohort-percentile authority A2, fixed lambda
    S7  S6 + query-adaptive lambda                   (proposed system)
    S8  S7 with PageRank cohort-percentile authority A3
    RAND random order of the BM25F candidate pool    (sanity baseline)
"""
from __future__ import annotations

import copy
import json
import re
import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import champion as champ
from .adaptive import IntentFeatures, adaptive_lambda, compute_features, p_foundational
from .authority import AUTHORITY_COLUMNS
from .boolean_search import Stats, evaluate, filter_docs
from .config import load_config, p
from .index import Index
from .net_score import minmax, net_score
from .query_parser import ParsedQuery, parse
from .ranking import (QueryTerm, bm25, bm25f, heap_topk, linear_zone, query_terms, tfidf_lnc_ltc,
                      topk_from_accumulator)

log = logging.getLogger("engine")
_YEAR_TOKEN = re.compile(r"^(19|20)\d{2}$")  # a bare year in free text is an intent cue / filter, not a topic word

SYSTEMS: dict[str, dict[str, Any]] = {
    "S1": {"base": "tfidf", "label": "tf-idf lnc.ltc (flat)"},
    "S2": {"base": "bm25", "label": "BM25 (flat)"},
    "S3": {"base": "bm25f", "label": "BM25F (zones)"},
    "S3L": {"base": "linear", "label": "linear zone BM25"},
    "S4": {"base": "bm25f", "authority": "A0", "lambda": "fixed", "label": "BM25F + raw citations"},
    "S5": {"base": "bm25f", "authority": "A1", "lambda": "fixed", "label": "BM25F + per-year citations"},
    "S6": {"base": "bm25f", "authority": "A2", "lambda": "fixed", "label": "BM25F + cohort percentile"},
    "S7": {"base": "bm25f", "authority": "A2", "lambda": "adaptive", "label": "BM25F + cohort + adaptive λ"},
    "S8": {"base": "bm25f", "authority": "A3", "lambda": "adaptive", "label": "S7 with PageRank authority"},
    "RAND": {"base": "random", "label": "random order of candidates"},
}


EXTRA = 5  # extra results fetched so that collapsing duplicates still returns k


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Result:
    rank: int
    doc_id: int
    arxiv_id: str
    title: str
    year: int
    score: float
    relevance: float | None = None     # R(q,d) (normalised BM25F) for net-score systems
    authority: float | None = None     # g(d)
    citations: int | None = None
    explain: dict | None = None


@dataclass
class SearchResponse:
    query: str
    parsed: str
    system: str
    results: list[Result]
    n_candidates: int
    lam: float | None = None
    p_found: float | None = None
    features: IntentFeatures | None = None
    query_terms: list[QueryTerm] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    boolean_stats: Stats | None = None

    @property
    def doc_ids(self) -> list[int]:
        return [r.doc_id for r in self.results]

    def table(self) -> pd.DataFrame:
        rows = [{"rank": r.rank, "arxiv_id": r.arxiv_id, "year": r.year, "score": round(r.score, 4),
                 "R": None if r.relevance is None else round(r.relevance, 3),
                 "g": None if r.authority is None else round(r.authority, 3),
                 "citations": r.citations, "title": r.title[:90]} for r in self.results]
        return pd.DataFrame(rows)


class SearchEngine:
    def __init__(self, index: Index, papers: pd.DataFrame, authority: pd.DataFrame | None, cfg: dict,
                 tuned: dict | None = None):
        self.index = index
        self.papers = papers
        self.cfg = deep_merge(cfg, tuned or {})
        self.tuned = tuned or {}
        self.auth = authority
        self.g = {}
        if authority is not None:
            a = authority.sort_values("doc_id")
            assert len(a) == index.N
            for key, col in AUTHORITY_COLUMNS.items():
                if col in a:
                    self.g[key] = a[col].to_numpy(dtype=np.float64)
            self.citations = a["cited_by_count"].to_numpy()
            self.has_cite = a["has_citation_data"].to_numpy()
        else:
            self.citations = np.zeros(index.N, dtype=int)
            self.has_cite = np.zeros(index.N, dtype=bool)
        self.meta_stems = {t for w in self.cfg["adaptive"].get("intent_only_terms", [])
                           for t in index.analyzer.terms(w)}
        self.titles = papers["title"].tolist()
        # "content seen?" fingerprint: exact duplicate re-submissions (same normalised title) are
        # collapsed at result-assembly time (29 such pairs in the 80k corpus)
        self.title_key = papers["title"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().tolist()
        self.years = papers["year"].to_numpy()

    # ---- loading -------------------------------------------------------------------------------
    @classmethod
    def load(cls, cfg: dict | None = None, use_tuned: bool = True) -> "SearchEngine":
        cfg = cfg or load_config()
        index = Index.load(p(cfg, "index"))
        papers = pd.read_parquet(p(cfg, "papers"), columns=["doc_id", "arxiv_id", "title", "abstract", "year", "categories"])
        auth_path = p(cfg, "processed_dir") / "authority.parquet"
        authority = pd.read_parquet(auth_path) if auth_path.exists() else None
        tuned = None
        tuned_path = p(cfg, "results_dir") / "tuning" / "params.json"
        if use_tuned and tuned_path.exists():
            tuned = json.loads(tuned_path.read_text()).get("params")
        return cls(index, papers, authority, cfg, tuned)

    # ---- pieces --------------------------------------------------------------------------------
    def candidate_set(self, pq: ParsedQuery) -> tuple[np.ndarray | None, Stats | None]:
        """Boolean constraints ∩ parametric filters (None = unconstrained free text)."""
        stats = None
        cand = None
        if pq.constrained:
            stats = Stats()
            cand = np.asarray(evaluate(self.index, pq.root, stats), dtype=np.int32)
        flt = filter_docs(self.index, pq)
        if flt is not None:
            cand = flt if cand is None else np.intersect1d(cand, flt, assume_unique=True)
        return cand, stats

    def _base_scores(self, base: str, qterms: list[QueryTerm], cand, ov: dict):
        rc = self.cfg["ranking"]
        if base == "tfidf":
            return tfidf_lnc_ltc(self.index, qterms, cand)
        if base == "bm25":
            return bm25(self.index, qterms, ov.get("k1", rc["bm25"]["k1"]), ov.get("b", rc["bm25"]["b"]), cand)
        if base in ("bm25f", "random"):
            w = dict(rc["bm25f"]["w"])
            if "w_title" in ov:
                w["title"] = ov["w_title"]
            b = dict(rc["bm25f"]["b"])
            if "b_title" in ov:
                b["title"] = ov["b_title"]
            if "b_abstract" in ov:
                b["abstract"] = ov["b_abstract"]
            return bm25f(self.index, qterms, ov.get("k1", rc["bm25f"]["k1"]), w, b, cand)
        if base == "linear":
            return linear_zone(self.index, qterms, rc["bm25"]["k1"], rc["bm25f"]["b"], rc["linear_zone"]["g"], cand)
        raise ValueError(base)

    def intent(self, pq: ParsedQuery, qterms: list[QueryTerm], top_scores: list[float],
               llm_intent: str | None = None) -> tuple[IntentFeatures, float]:
        ac = self.cfg["adaptive"]
        idfs = [self.index.zone("all").idf_ln(q.term) for q in qterms]
        feats = compute_features(pq.raw, idfs, top_scores, ac, ac["norm"], llm_intent)
        return feats, p_foundational(feats, ac["weights"], ac.get("use_llm_feature", False) and llm_intent is not None)

    def _collapse_duplicates(self, top: list, k: int) -> list:
        seen, out = set(), []
        for d, s in top:
            key = self.title_key[d]
            if key in seen:
                continue
            seen.add(key)
            out.append((d, s))
            if len(out) == k:
                break
        return out

    # ---- main entry point ----------------------------------------------------------------------
    def search(self, query: str | ParsedQuery, system: str = "S7", k: int = 10, explain: bool = False,
               mode: str = "exact", llm_intent: str | None = None, overrides: dict | None = None) -> SearchResponse:
        """Run one query. mode: exact | champion | eliminate. overrides: ablation knobs
        (k1, b, w_title, b_title, b_abstract, lambda, combine, authority, adaptive, pool)."""
        ov = overrides or {}
        spec = dict(SYSTEMS[system])
        if "authority" in ov:
            spec["authority"] = ov["authority"]
            spec.setdefault("lambda", "fixed")
        if "adaptive" in ov:
            spec["lambda"] = "adaptive" if ov["adaptive"] else "fixed"
        t = {}
        t0 = time.perf_counter()
        pq = query if isinstance(query, ParsedQuery) else parse(query)
        qterms = query_terms(self.index, pq)
        topical = [q for q in qterms if q.term not in self.meta_stems and not _YEAR_TOKEN.match(q.term)]
        qterms = topical or qterms  # intent-only words never rank (unless they are the whole query)
        cand, bstats = self.candidate_set(pq)
        t["parse_boolean"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        if mode == "champion":
            A = champ.champion_candidates(self.index, qterms)
            cand = A if cand is None else np.intersect1d(cand, A, assume_unique=True)
        elif mode == "eliminate":
            ec = self.cfg["ranking"]["index_elimination"]
            qterms = champ.eliminate_terms(self.index, qterms, ec["idf_threshold"])
            if len(qterms) >= ec["min_terms_for_match_rule"]:
                A = champ.many_terms_candidates(self.index, qterms, ec["min_match_fraction"])
                cand = A if cand is None else np.intersect1d(cand, A, assume_unique=True)
        acc, touched = self._base_scores(spec["base"], qterms, cand, ov)
        if pq.constrained and cand is not None and mode == "exact":
            touched[cand] = True  # Boolean-required docs that matched no ranked term still qualify (score 0)
        n_cand = int(touched.sum())
        t["score"] = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        resp = SearchResponse(query=pq.raw, parsed=pq.to_string(), system=system, results=[],
                              n_candidates=n_cand, query_terms=qterms, boolean_stats=bstats)
        auth_key = spec.get("authority")
        pool_n = ov.get("pool", self.cfg["ranking"]["candidate_pool"])
        if spec["base"] == "random":
            pool = topk_from_accumulator(acc, touched, pool_n)
            rng = np.random.default_rng(zlib.crc32(pq.raw.encode()))
            order = rng.permutation(len(pool))
            top = [(pool[i][0], float(len(pool) - r)) for r, i in enumerate(order[:k + EXTRA])]
        elif auth_key is None:
            top = topk_from_accumulator(acc, touched, k + EXTRA)
            if system == "S3":  # intent features are diagnostic for S3 too (shown in the demo)
                resp.features, resp.p_found = self.intent(pq, qterms, [s for _, s in top[:10]], llm_intent)
        else:
            if auth_key not in self.g:
                raise RuntimeError("authority scores not built; run `make index` after `make citations`")
            pool = topk_from_accumulator(acc, touched, pool_n)
            docs = np.array([d for d, _ in pool], dtype=np.int64)
            rel = np.array([s for _, s in pool])
            R = minmax(rel)
            g = self.g[auth_key][docs] if len(docs) else np.empty(0)
            feats, pf = self.intent(pq, qterms, rel[:10].tolist(), llm_intent)
            resp.features, resp.p_found = feats, pf
            ac = self.cfg["adaptive"]
            if "lambda" in ov:
                lam = float(ov["lambda"])
            elif spec.get("lambda") == "adaptive":
                lam = adaptive_lambda(pf, ac["lambda_min"], ac["lambda_max"])
            else:
                ns = self.cfg["net_score"]
                lam = float((ns.get("fixed_lambda_by_authority") or {}).get(auth_key, ns["fixed_lambda"]))
            resp.lam = lam
            net = net_score(R, g, lam, ov.get("combine", self.cfg["net_score"]["combine"]))
            top = heap_topk(docs.tolist(), net.tolist(), k + EXTRA)
            rmap = dict(zip(docs.tolist(), R.tolist()))
            gmap = dict(zip(docs.tolist(), g.tolist()))
        t["rank"] = (time.perf_counter() - t2) * 1e3
        t["total"] = (time.perf_counter() - t0) * 1e3

        top = self._collapse_duplicates(top, k)
        for r, (d, s) in enumerate(top, 1):
            res = Result(rank=r, doc_id=int(d), arxiv_id=self.index.arxiv_ids[d], title=self.titles[d],
                         year=int(self.years[d]), score=float(s), citations=int(self.citations[d]))
            if auth_key is not None and spec["base"] != "random":
                res.relevance, res.authority = rmap[d], gmap[d]
            resp.results.append(res)
        resp.timings_ms = t
        if explain:
            from .explain import explain_response
            explain_response(self, resp, spec, ov)
        return resp
