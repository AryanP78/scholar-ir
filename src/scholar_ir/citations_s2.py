"""Second citation source: Semantic Scholar (Academic Graph API), looked up by arXiv id.

Why: in OpenAlex an arXiv preprint and its published version are often separate works, and the
citations go to the published one, so the arXiv-DOI record of a well-cited paper can show only a
handful of citations (pass-1 coverage report: median 0-2 citations per year). Semantic Scholar
resolves an arXiv id to the canonical paper record, whose citationCount includes the citations
of all its versions. OpenAlex stays the source of the in-corpus citation graph (PageRank).

    POST https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,year,citationCount
         {"ids": ["ARXIV:1706.03762", ...]}            (up to 500 ids per request)

Cached per batch under data/cache/s2/, resumable, polite (sleep between requests, backoff on 429).
Optional key: $S2_API_KEY (x-api-key header).

    python -m scholar_ir.citations_s2 --test
    python -m scholar_ir.citations_s2            # then: python -m scholar_ir.citations --build-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time

import pandas as pd
import requests

from .citations import TEST_PAPERS, title_jaccard
from .config import load_config, load_dotenv, p, setup_logging

log = logging.getLogger("citations_s2")
URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,citationCount"  # the minimum: count + title for the sanity check


def _key(ids):
    return hashlib.sha1("|".join(sorted(ids)).encode()).hexdigest()[:20]


def fetch(ids: list[str], cache_dir, cfg) -> list:
    c = cfg["citations"]["s2"]
    f = cache_dir / f"{_key(ids)}.json"
    if f.exists():
        return json.loads(f.read_text())["data"]
    headers = {"User-Agent": "scholar-ir/0.1 (CSD358 course project)"}
    if os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    for attempt in range(c["max_retries"]):
        try:
            r = requests.post(URL, params={"fields": FIELDS}, json={"ids": [f"ARXIV:{i}" for i in ids]},
                              headers=headers, timeout=120)
        except requests.RequestException as exc:
            log.warning("network error %s", exc)
            time.sleep(c["backoff_base"] ** (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"ids": ids, "data": data}))
            time.sleep(c["sleep_seconds"])
            return data
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(120, c["backoff_base"] ** (attempt + 1))
            log.warning("HTTP %s; retry in %.0fs", r.status_code, wait)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Semantic Scholar HTTP {r.status_code}: {r.text[:300]}")
    log.warning("Semantic Scholar: batch skipped after %d retries (rate limited); re-run to fill it in", c["max_retries"])
    return None


def collect(papers: pd.DataFrame, cfg) -> pd.DataFrame:
    """arxiv_id -> s2 citation count (title-checked), from the cache."""
    cache = p(cfg, "cache_dir") / "s2"
    titles = dict(zip(papers["arxiv_id"], papers["title"]))
    min_j = cfg["citations"]["s2"]["min_title_jaccard"]
    rows = []
    for f in cache.glob("*.json"):
        blob = json.loads(f.read_text())
        for aid, rec in zip(blob["ids"], blob["data"]):
            if not rec or aid not in titles:
                continue
            j = title_jaccard(rec.get("title") or "", titles[aid])
            rows.append({"arxiv_id": aid, "s2_citations": rec.get("citationCount"), 
                         "s2_title_jaccard": round(j, 3), "s2_ok": j >= min_j})
    return pd.DataFrame(rows, columns=["arxiv_id", "s2_citations", "s2_title_jaccard", "s2_ok"])


def main(argv=None):
    setup_logging()
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.test:
        ids = list(TEST_PAPERS)
        data = fetch(ids, p(cfg, "cache_dir") / "s2_test", cfg)
        for aid, rec in zip(ids, data):
            print(f"  {aid:<12} " + (f"citations={rec.get('citationCount'):<7} {rec.get('title','')[:60]!r}"
                                       if rec else "NOT FOUND"))
        return
    papers = pd.read_parquet(p(cfg, "papers"), columns=["arxiv_id", "title"])
    ids = sorted(papers["arxiv_id"])
    bs = cfg["citations"]["s2"]["batch_size"]
    batches = [ids[i:i + bs] for i in range(0, len(ids), bs)]
    cache = p(cfg, "cache_dir") / "s2"
    t0 = time.time()
    missing = 0
    for i, b in enumerate(batches, 1):
        if fetch(b, cache, cfg) is None:
            missing += 1
            time.sleep(60)  # back off hard after a fully rate-limited batch
        if i % 10 == 0:
            log.info("%d/%d batches (%.0fs), %d skipped so far", i, len(batches), time.time() - t0, missing)
    if missing:
        log.warning("%d batches were skipped because of rate limiting: run this command again to fetch them", missing)
    df = collect(papers, cfg)
    log.info("Semantic Scholar: %d of %d papers found, %d pass the title check; median citations %.0f",
             len(df), len(papers), int(df.s2_ok.sum()), df.loc[df.s2_ok, "s2_citations"].median())
    log.info("now run: python -m scholar_ir.citations --build-only   (merges both sources)")


if __name__ == "__main__":
    main()
