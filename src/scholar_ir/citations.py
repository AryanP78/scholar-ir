"""Phase 1b: citation data (the authority source) from the OpenAlex API.

The arXiv dump has no citation counts. OpenAlex merges an arXiv preprint into its published version,
and the merged record usually carries the venue DOI, so a lookup by arXiv DOI alone misses exactly
the most-cited papers (probe, 2026-10-07: "Attention Is All You Need", BERT and DPR were all missed
by DOI). We therefore match in three passes, cheapest first:

  pass 1  filter=doi:10.48550/arxiv.<id>|...                 (batches of 100; exact)
  pass 2  filter=locations.landing_page_url:http://arxiv.org/abs/<id>|...  (batches of 100)
          -> results are assigned to requested papers by TITLE similarity, which also rejects
             mis-merged OpenAlex records (the probe found one for BERT)
  pass 3  filter=title.search:<title> for what is still missing (one call each, 10x the cost;
          capped by --max-title-search)
Landing-page and title-search matches must pass a title-similarity check (token Jaccard >=
citations.min_title_jaccard). Exact arXiv-DOI hits are always kept (the DOI is the paper's identity),
but flagged `doi_title_mismatch` when OpenAlex's title disagrees (corrupted metadata).

Politeness / reproducibility: every raw response is cached on disk (data/cache/openalex/<pass>/),
so re-runs never re-query a cached request and the fetch is resumable; fixed sleep between requests;
exponential backoff on 429/5xx; optional API key from $OPENALEX_API_KEY (never written to disk/logs).

Usage:
    python -m scholar_ir.citations --test          # 5 well-known papers through all passes
    python -m scholar_ir.citations                 # full corpus (resumable)
    python -m scholar_ir.citations --build-only    # rebuild citations.parquet from the cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import load_config, load_dotenv, p, setup_logging

log = logging.getLogger("citations")

TEST_PAPERS = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering",
    "1301.3781": "Efficient Estimation of Word Representations in Vector Space",
}


class BudgetExhausted(RuntimeError):
    pass


_NORM = re.compile(r"[^a-z0-9 ]+")


def title_tokens(title: str) -> set[str]:
    return set(_NORM.sub(" ", (title or "").lower()).split())


def title_jaccard(a: str, b: str) -> float:
    """Jaccard coefficient of the two titles' word sets (Jaccard, tf-idf lecture)."""
    ta, tb = title_tokens(a), title_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def arxiv_doi(arxiv_id: str, prefix: str) -> str:
    return f"{prefix}{arxiv_id}".lower()


def _key(parts: Iterable[str]) -> str:
    return hashlib.sha1("|".join(sorted(parts)).encode()).hexdigest()[:20]


class OpenAlexClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg["citations"]
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "scholar-ir/0.1 (CSD358 course project)"
        self.api_key = os.environ.get("OPENALEX_API_KEY")
        self.calls = 0
        self.cost = 0.0

    def get(self, params: dict, cache_file: Path, extra: dict | None = None) -> dict:
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        c = self.cfg
        for attempt in range(c["max_retries"]):
            try:
                r = self.session.get(c["api_base"], params=params, timeout=c["timeout_seconds"])
            except requests.RequestException as exc:
                wait = c["backoff_base"] ** attempt
                log.warning("network error %s; retry in %.0fs", exc, wait)
                time.sleep(wait)
                continue
            if r.status_code == 200:
                data = r.json()
                data.update(extra or {})
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data))
                self.calls += 1
                self.cost += float((data.get("meta") or {}).get("cost_usd") or 0)
                time.sleep(c["sleep_seconds"])
                return data
            if r.status_code == 429 and ("budget" in r.text.lower() or attempt >= 3):
                raise BudgetExhausted(r.text[:300])
            if r.status_code in (429, 500, 502, 503, 504):
                wait = c["backoff_base"] ** attempt
                log.warning("HTTP %s; retry in %.0fs", r.status_code, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenAlex HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError("OpenAlex: retries exhausted")


def _chunks(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ---- the three passes -----------------------------------------------------------------------------

def pass_doi(client: OpenAlexClient, ids: list[str], cache: Path) -> None:
    c = client.cfg
    for batch in _chunks(sorted(ids), c["batch_size"]):
        dois = "|".join(arxiv_doi(i, c["doi_prefix"]) for i in batch)
        client.get({"filter": f"doi:{dois}", "per-page": c["per_page"], "select": c["select"]},
                   cache / "doi" / f"{_key(batch)}.json", {"_requested_ids": batch})


def pass_landing(client: OpenAlexClient, ids: list[str], cache: Path) -> None:
    c = client.cfg
    for batch in _chunks(sorted(ids), c["batch_size"]):
        urls = "|".join(f"{c['landing_prefix']}{i}" for i in batch)
        client.get({"filter": f"locations.landing_page_url:{urls}", "per-page": c["per_page"],
                    "select": c["select"]}, cache / "landing" / f"{_key(batch)}.json", {"_requested_ids": batch})


def pass_title(client: OpenAlexClient, todo: list[tuple[str, str]], cache: Path, max_calls: int) -> None:
    for aid, title in todo[:max_calls]:
        q = _NORM.sub(" ", title.lower()).strip()
        if not q:
            continue
        client.get({"filter": f"title.search:{q}", "per-page": 5, "select": client.cfg["select"]},
                   cache / "title" / f"{_key([aid])}.json", {"_requested_ids": [aid]})


# ---- matching cached responses back to papers -----------------------------------------------------

def collect_matches(papers: pd.DataFrame, cfg: dict) -> dict[str, dict]:
    """arxiv_id -> OpenAlex work (+ match_method), from every cached response."""
    c = cfg["citations"]
    cache = p(cfg, "cache_dir") / "openalex"
    titles = dict(zip(papers["arxiv_id"], papers["title"]))
    years = dict(zip(papers["arxiv_id"], papers["year"]))
    min_j = c["min_title_jaccard"]
    matches: dict[str, dict] = {}

    doi_to_id = {arxiv_doi(a, c["doi_prefix"]): a for a in titles}
    for f in sorted((cache / "doi").glob("*.json")):
        for w in json.loads(f.read_text()).get("results", []):
            doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
            aid = doi_to_id.get(doi)
            if aid and aid not in matches:
                # The arXiv DOI identifies the paper exactly, so the hit is kept even when OpenAlex's title
                # disagrees (probe 2026-10-07: 10.48550/arxiv.2005.11401, the RAG paper, carries a wrong
                # title but RAG's ~3k citations; its title-search "match" was a duplicate with 18).
                # Such hits are flagged so the coverage report can count them.
                j = title_jaccard(w.get("title") or "", titles[aid])
                matches[aid] = {**w, "match_method": "doi" if j >= c["min_doi_title_jaccard"] else "doi_title_mismatch",
                                "title_jaccard": round(j, 3)}

    def assign(results: list[dict], requested: list[str], method: str) -> None:
        open_ids = [a for a in requested if a in titles and a not in matches]
        for w in results:
            best, best_j = None, 0.0
            for a in open_ids:
                j = title_jaccard(w.get("title") or "", titles[a])
                if j > best_j:
                    best, best_j = a, j
            # the year check guards near-miss titles; an (almost) identical title is accepted even when
            # OpenAlex dates the merged record later (probe: "Attention Is All You Need" is dated 2025)
            year_ok = best is not None and (w.get("publication_year") is None or best_j >= c["exact_title_jaccard"]
                                            or abs(int(w["publication_year"]) - int(years[best])) <= c["max_year_gap"])
            if best is not None and best_j >= min_j and year_ok:
                matches[best] = {**w, "match_method": method, "title_jaccard": round(best_j, 3)}
                open_ids.remove(best)

    for sub, method in (("landing", "landing_page"), ("title", "title_search")):
        for f in sorted((cache / sub).glob("*.json")):
            data = json.loads(f.read_text())
            assign(data.get("results", []), data.get("_requested_ids", []), method)
    return matches


def build_table(papers: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join matches to doc_ids; keep only references that point inside the corpus (for PageRank)."""
    matches = collect_matches(papers, cfg)
    rows = []
    for doc_id, aid in zip(papers["doc_id"], papers["arxiv_id"]):
        w = matches.get(aid)
        rows.append({
            "doc_id": int(doc_id),
            "openalex_id": w["id"] if w else None,
            "cited_by_count": int(w.get("cited_by_count") or 0) if w else 0,
            "oa_publication_year": w.get("publication_year") if w else None,
            "match_method": w["match_method"] if w else "none",
            "refs": (w.get("referenced_works") or []) if w else [],
            "has_citation_data": w is not None,
        })
    df = pd.DataFrame(rows)
    df["oa_cited_by_count"] = df["cited_by_count"]
    # Semantic Scholar counts (canonical record incl. all versions), if fetched: preferred for the count
    from .citations_s2 import collect as s2_collect
    s2 = s2_collect(papers, cfg)
    df["s2_citations"] = pd.NA
    df["count_source"] = np.where(df["has_citation_data"], "openalex", "none")
    if len(s2) and cfg["citations"].get("count_source", "s2_then_openalex") == "s2_then_openalex":
        s2 = s2[s2.s2_ok & s2.s2_citations.notna()].drop_duplicates("arxiv_id").set_index("arxiv_id")
        aid = papers.set_index("doc_id").loc[df["doc_id"], "arxiv_id"].to_numpy()
        hit = pd.Series(aid).isin(s2.index).to_numpy()
        df.loc[hit, "s2_citations"] = s2.loc[aid[hit], "s2_citations"].astype(int).to_numpy()
        df.loc[hit, "cited_by_count"] = s2.loc[aid[hit], "s2_citations"].astype(int).to_numpy()
        df.loc[hit, "count_source"] = "semantic_scholar"
        df.loc[hit, "has_citation_data"] = True
    # one OpenAlex work matched by two arXiv ids (rare duplicate submissions): keep the first only
    dup = df["openalex_id"].notna() & df["openalex_id"].duplicated()
    df.loc[dup, ["openalex_id", "oa_publication_year", "match_method"]] = [None, None, "duplicate"]
    df.loc[dup, "refs"] = pd.Series([[] for _ in range(int(dup.sum()))], index=df.index[dup], dtype=object)
    in_corpus = {oid: d for oid, d in zip(df["openalex_id"], df["doc_id"]) if oid}
    df["referenced_doc_ids"] = [[in_corpus[r] for r in refs if r in in_corpus] for refs in df["refs"]]
    df["n_references"] = df["refs"].str.len()
    return df.drop(columns=["refs"])


def coverage_report(df: pd.DataFrame, papers: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = df.merge(papers[["doc_id", "year"]], on="doc_id")
    rep = m.groupby("year").agg(papers=("doc_id", "size"), matched=("has_citation_data", "sum"),
                                median_citations=("cited_by_count", "median"),
                                mean_citations=("cited_by_count", "mean")).reset_index()
    rep["coverage_pct"] = (100 * rep["matched"] / rep["papers"]).round(1)
    total = pd.DataFrame([{"year": "ALL", "papers": len(m), "matched": int(m.has_citation_data.sum()),
                           "median_citations": m.cited_by_count.median(), "mean_citations": m.cited_by_count.mean(),
                           "coverage_pct": round(100 * m.has_citation_data.mean(), 1)}])
    rep = pd.concat([rep, total], ignore_index=True)
    out = p(cfg, "results_dir") / "data"
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "citation_coverage.csv", index=False)
    m["match_method"].value_counts().rename_axis("method").reset_index(name="papers") \
        .to_csv(out / "citation_match_methods.csv", index=False)
    m["count_source"].value_counts().rename_axis("count_source").reset_index(name="papers") \
        .to_csv(out / "citation_count_sources.csv", index=False)
    both = m[m["s2_citations"].notna() & m["openalex_id"].notna()]
    if len(both):
        pd.DataFrame([{"papers_in_both": len(both),
                       "median_openalex_arxiv_record": float(both["oa_cited_by_count"].median()),
                       "median_semantic_scholar": float(both["s2_citations"].astype(float).median()),
                       "share_s2_higher": round(float((both["s2_citations"].astype(float) > both["oa_cited_by_count"]).mean()), 3),
                       "spearman": round(float(both["oa_cited_by_count"].rank().corr(both["s2_citations"].astype(float).rank())), 3)}]) \
            .to_csv(out / "citation_source_comparison.csv", index=False)
    with open(out / "citation_graph_stats.json", "w") as fh:
        json.dump({"in_corpus_edges": int(df["referenced_doc_ids"].str.len().sum()),
                   "papers_with_in_corpus_refs": int((df["referenced_doc_ids"].str.len() > 0).sum())}, fh, indent=2)
    return rep


def fetch_all(papers: pd.DataFrame, cfg: dict, max_title_search: int, cache: Path | None = None) -> None:
    client = OpenAlexClient(cfg)
    cache = cache or p(cfg, "cache_dir") / "openalex"
    ids = papers["arxiv_id"].tolist()
    try:
        log.info("pass 1/3: arXiv DOI lookup for %d papers", len(ids))
        pass_doi(client, ids, cache)
        missing = [a for a in ids if a not in collect_matches(papers, cfg)]
        log.info("pass 2/3: arXiv landing-page lookup for %d unmatched papers", len(missing))
        pass_landing(client, missing, cache)
        matched = collect_matches(papers, cfg)
        todo = [(a, t) for a, t in zip(papers["arxiv_id"], papers["title"]) if a not in matched]
        log.info("pass 3/3: title search for up to %d of %d still unmatched", max_title_search, len(todo))
        pass_title(client, todo, cache, max_title_search)
    except BudgetExhausted as exc:
        log.error("OpenAlex daily budget exhausted (%s). Re-run later or set OPENALEX_API_KEY; "
                  "cached requests are skipped on re-run.", exc)
    log.info("new requests this run: %d (reported cost $%.4f)", client.calls, client.cost)


def run_test(cfg: dict) -> None:
    papers = pd.DataFrame({"doc_id": range(len(TEST_PAPERS)), "arxiv_id": list(TEST_PAPERS),
                           "title": list(TEST_PAPERS.values()), "year": [2017, 2018, 2020, 2020, 2013]})
    cfg = {**cfg, "paths": {**cfg["paths"], "cache_dir": str(p(cfg, "cache_dir") / "_test")}}
    fetch_all(papers, cfg, max_title_search=5, cache=p(cfg, "cache_dir") / "openalex")
    for aid, w in sorted(collect_matches(papers, cfg).items()):
        print(f"  {aid:<12} via {w['match_method']:<13} cited_by={w.get('cited_by_count'):<7} "
              f"year={w.get('publication_year')} refs={len(w.get('referenced_works') or [])}  {w.get('title','')[:50]!r}")
    missing = set(TEST_PAPERS) - set(collect_matches(papers, cfg))
    print("unmatched:", sorted(missing) or "none")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--max-title-search", type=int, default=None,
                    help="cap on pass-3 title searches (default from config)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.test:
        run_test(cfg)
        return
    papers = pd.read_parquet(p(cfg, "papers"), columns=["doc_id", "arxiv_id", "title", "year"])
    if not args.build_only:
        fetch_all(papers, cfg, args.max_title_search if args.max_title_search is not None
                  else cfg["citations"]["max_title_search"])
    df = build_table(papers, cfg)
    df.to_parquet(p(cfg, "citations"), index=False)
    rep = coverage_report(df, papers, cfg)
    log.info("citation coverage:\n%s", rep.to_string(index=False))


if __name__ == "__main__":
    main()
