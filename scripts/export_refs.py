"""Export every paper's FULL OpenAlex reference list (inside and outside the corpus) from the
cached OpenAlex responses, so the topic-local HITS graph can use references to papers outside
our 80,000. Reads only data/cache/openalex/ (no network). Output: data/processed/refs_full.parquet
with columns doc_id, refs (list of OpenAlex work numbers, e.g. W2963403868 -> 2963403868).

    python scripts/export_refs.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scholar_ir.citations import collect_matches  # noqa: E402
from scholar_ir.config import load_config, p  # noqa: E402


def wnum(url: str) -> int:
    return int(url.rsplit("/", 1)[-1].lstrip("W"))


def main():
    cfg = load_config()
    cache = p(cfg, "cache_dir") / "openalex"
    n_files = sum(1 for _ in cache.rglob("*.json")) if cache.exists() else 0
    print(f"OpenAlex cache: {cache}  ({n_files} files)")
    if n_files == 0:
        sys.exit("No OpenAlex cache found here, so the full reference lists are not on this computer.")
    papers = pd.read_parquet(p(cfg, "papers"), columns=["doc_id", "arxiv_id", "title", "year"])
    matches = collect_matches(papers, cfg)
    rows = []
    for doc_id, aid in zip(papers["doc_id"], papers["arxiv_id"]):
        w = matches.get(aid)
        if w:
            rows.append({"doc_id": int(doc_id), "openalex_id": wnum(w["id"]),
                         "refs": [wnum(r) for r in (w.get("referenced_works") or [])]})
    df = pd.DataFrame(rows)
    out = p(cfg, "processed_dir") / "refs_full.parquet"
    df.to_parquet(out, index=False)
    n_refs = int(df["refs"].str.len().sum())
    print(f"papers with OpenAlex record: {len(df):,}   total references: {n_refs:,}   "
          f"papers with >=1 reference: {int((df['refs'].str.len() > 0).sum()):,}")
    print(f"written: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
