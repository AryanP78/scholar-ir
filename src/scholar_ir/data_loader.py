"""Phase 1: build the arXiv CS corpus ("deciding what a document is").

Reads the Cornell/Kaggle arXiv metadata dump (one JSON object per paper), either streamed from the
Hugging Face mirror or read from the downloaded Kaggle file, and writes a stratified, seeded subset.

Single streaming pass:
  * filter by category and first-submission year,
  * clean title/abstract (whitespace, LaTeX leftovers), drop short/empty abstracts, dedupe by id,
  * keep a uniform reservoir sample per publication year (Algorithm R) and count the population,
then allocate the target size across years (water-filling, quota proportional to count^power) so
every year is represented, and subsample each year's reservoir.

Usage:
    python -m scholar_ir.data_loader                    # source from config.yaml (HF parquet shards, cached)
    python -m scholar_ir.data_loader --source file --path data/raw/arxiv-metadata-oai-snapshot.json
    python -m scholar_ir.data_loader --limit 200000     # quick partial run
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
import time
from collections import Counter
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from .config import load_config, p, seed_everything, setup_logging

log = logging.getLogger("data_loader")

_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.S)
_CITE = re.compile(r"\\(cite|ref|eqref|label|url|footnote)\w*\{[^}]*\}")
_CMD_ARG = re.compile(r"\\(emph|textbf|textit|texttt|textsc|mathrm|mathbf|mathcal|text|it|bf)\s*\{([^}]*)\}")
_CMD = re.compile(r"\\[a-zA-Z]+\*?")
_ACCENT = re.compile(r"\\['`^\"~=.uvHcdbt]\{?([a-zA-Z])\}?")
_BRACES = re.compile(r"[{}]")
_WS = re.compile(r"\s+")
_NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")


def clean_text(text: str | None) -> str:
    """Collapse whitespace and strip LaTeX leftovers ($...$ math, \\cite{}, commands, braces)."""
    if not text:
        return ""
    t = _MATH.sub(" ", text)
    t = _CITE.sub(" ", t)
    t = _CMD_ARG.sub(r"\2", t)
    t = _ACCENT.sub(r"\1", t)
    t = _CMD.sub(" ", t)
    t = _BRACES.sub("", t)
    return _WS.sub(" ", t).strip()


def _parse_versions(versions: Any) -> list[dict]:
    if isinstance(versions, list):
        return versions
    if isinstance(versions, str) and versions.strip():
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(versions)
                if isinstance(v, list):
                    return v
            except (ValueError, SyntaxError):
                continue
    return []


def first_created(record: dict) -> datetime | None:
    """Date of the FIRST submission (versions[0].created), not update_date (matters for age)."""
    versions = _parse_versions(record.get("versions"))
    if versions:
        created = versions[0].get("created") if isinstance(versions[0], dict) else None
        if created:
            try:
                return parsedate_to_datetime(created)
            except (TypeError, ValueError):
                pass
    m = _NEW_ID.match(str(record.get("id", "")))  # fallback: new-style ids encode YYMM
    if m:
        return datetime(2000 + int(m.group(1)), int(m.group(2)), 1)
    return None


def iter_records(source: str, cfg: dict, path: str | None = None) -> Iterator[dict]:
    """Yield raw metadata records from the configured source."""
    if source == "file":
        fpath = Path(path or p(cfg, "raw_dir") / Path(cfg["data"]["kaggle_file"]).name)
        if not fpath.exists():
            fpath = Path(path or cfg["data"]["kaggle_file"])
        log.info("Reading %s", fpath)
        with open(fpath, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    elif source == "hf":
        # Download the mirror's parquet shards once (cached in data/raw/hf, resumable), then read only
        # the columns we need with pyarrow. Much faster than `datasets` streaming, and an interrupted
        # run does not have to download again.
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, hf_hub_download
        name = cfg["data"]["hf_dataset"]
        local = p(cfg, "raw_dir") / "hf"
        files = sorted(f for f in HfApi().list_repo_files(name, repo_type="dataset") if f.endswith(".parquet"))
        log.info("Hugging Face dataset %s: %d parquet shards -> %s", name, len(files), local)
        cols = ["id", "title", "abstract", "categories", "versions", "doi"]
        for f in files:
            path = hf_hub_download(name, f, repo_type="dataset", local_dir=local)
            pf = pq.ParquetFile(path)
            use = [c for c in cols if c in pf.schema_arrow.names]
            log.info("reading %s (%d rows)", f, pf.metadata.num_rows)
            for batch in pf.iter_batches(batch_size=20000, columns=use):
                yield from batch.to_pylist()
    else:
        raise ValueError(f"unknown source {source!r}")


def allocate_quotas(counts: dict[int, int], capacity: dict[int, int], target: int,
                    power: float) -> dict[int, int]:
    """Water-filling allocation: quota_y ∝ counts_y**power, capped at what we hold for year y."""
    quotas = {y: 0 for y in counts}
    remaining = target
    open_years = {y for y in counts if capacity[y] > 0}
    while remaining > 0 and open_years:
        weights = {y: counts[y] ** power for y in open_years}
        total_w = sum(weights.values())
        give = {y: max(1, int(remaining * weights[y] / total_w)) for y in open_years}
        progressed = 0
        for y in sorted(open_years):
            room = capacity[y] - quotas[y]
            g = min(give[y], room, remaining - progressed)
            quotas[y] += g
            progressed += g
            if remaining - progressed <= 0:
                break
        remaining -= progressed
        open_years = {y for y in open_years if quotas[y] < capacity[y]}
        if progressed == 0:
            break
    return quotas


def build_subset(records: Iterable[dict], cfg: dict, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    dcfg = cfg["data"]
    cats = set(dcfg["categories"])
    min_year, max_year = dcfg["min_year"], dcfg.get("max_year")
    min_words = dcfg["min_abstract_words"]
    cap = dcfg["reservoir_per_year"]
    rng = random.Random(cfg["seed"])

    reservoirs: dict[int, list[dict]] = {}
    population: Counter = Counter()
    seen: set[str] = set()
    stats = Counter()
    t0 = time.time()
    for i, rec in enumerate(records):
        if limit and i >= limit:
            break
        if i and i % 250_000 == 0:
            log.info("scanned %d records, kept-population %d (%.0fs)", i, sum(population.values()),
                     time.time() - t0)
        stats["scanned"] += 1
        rec_cats = str(rec.get("categories") or "").split()
        if not cats.intersection(rec_cats):
            continue
        stats["category_match"] += 1
        arxiv_id = str(rec.get("id", "")).strip()
        if not arxiv_id or arxiv_id in seen:
            stats["duplicate_or_no_id"] += 1
            continue
        created = first_created(rec)
        if created is None:
            stats["no_date"] += 1
            continue
        year = created.year
        if year < min_year or (max_year and year > max_year):
            stats["out_of_year_range"] += 1
            continue
        title = clean_text(rec.get("title"))
        abstract = clean_text(rec.get("abstract"))
        if not title or len(abstract.split()) < min_words:
            stats["short_or_empty"] += 1
            continue
        seen.add(arxiv_id)
        population[year] += 1
        item = {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "categories": rec_cats,
            "year": year,
            "first_created_date": created.strftime("%Y-%m-%d"),
            "n_versions": len(_parse_versions(rec.get("versions"))),
            "journal_doi": rec.get("doi") or None,
        }
        res = reservoirs.setdefault(year, [])
        n = population[year]
        if len(res) < cap:            # Algorithm R reservoir sampling
            res.append(item)
        else:
            j = rng.randrange(n)
            if j < cap:
                res[j] = item
    log.info("scan done in %.0fs: %s", time.time() - t0, dict(stats))

    capacity = {y: len(r) for y, r in reservoirs.items()}
    quotas = allocate_quotas(dict(population), capacity, dcfg["target_size"], dcfg["year_alloc_power"])
    rows: list[dict] = []
    for y in sorted(reservoirs):
        res = sorted(reservoirs[y], key=lambda r: r["arxiv_id"])  # order-independent before sampling
        rows.extend(random.Random(cfg["seed"] + y).sample(res, quotas[y]))
    df = pd.DataFrame(rows).sort_values(["year", "arxiv_id"]).reset_index(drop=True)
    df.insert(0, "doc_id", range(len(df)))
    info = {"filter_stats": dict(stats), "population_per_year": dict(sorted(population.items())),
            "quota_per_year": dict(sorted(quotas.items()))}
    return df, info


def save_statistics(df: pd.DataFrame, info: dict, cfg: dict) -> None:
    out = p(cfg, "results_dir") / "data"
    out.mkdir(parents=True, exist_ok=True)
    per_year = pd.DataFrame({
        "year": list(info["population_per_year"].keys()),
        "population": list(info["population_per_year"].values()),
        "sampled": [info["quota_per_year"].get(y, 0) for y in info["population_per_year"]],
    })
    per_year.to_csv(out / "papers_per_year.csv", index=False)
    cat_counts = Counter(c for cats in df["categories"] for c in cats if c in set(cfg["data"]["categories"]))
    pd.DataFrame(sorted(cat_counts.items(), key=lambda kv: -kv[1]), columns=["category", "papers"]) \
        .to_csv(out / "papers_per_category.csv", index=False)
    lengths = df["abstract"].str.split().str.len()
    desc = lengths.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1)
    desc.to_frame("abstract_words").to_csv(out / "abstract_length.csv")
    with open(out / "filter_stats.json", "w") as fh:
        json.dump(info["filter_stats"], fh, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].bar(per_year["year"], per_year["sampled"], color="#3b6ea5")
        axes[0].set_title("Papers per year in the corpus (first-submission year)")
        axes[0].set_xlabel("year"); axes[0].set_ylabel("papers")
        axes[1].hist(lengths, bins=50, color="#3b6ea5")
        axes[1].set_title("Abstract length (words)"); axes[1].set_xlabel("words")
        fig.tight_layout(); fig.savefig(out / "corpus_stats.png", dpi=150); plt.close(fig)
    except Exception as exc:  # plotting is optional
        log.warning("plot skipped: %s", exc)
    log.info("statistics written to %s", out)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--source", choices=["hf", "file"])
    ap.add_argument("--path", help="path to the Kaggle JSON-lines file (source=file)")
    ap.add_argument("--limit", type=int, help="stop after N raw records (testing)")
    ap.add_argument("--out", help="override output parquet path")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    source = args.source or cfg["data"]["source"]
    df, info = build_subset(iter_records(source, cfg, args.path), cfg, limit=args.limit)
    if df.empty:
        raise SystemExit("No papers matched the filters; check the source and config.")
    assert df["title"].str.len().gt(0).all() and df["abstract"].str.len().gt(0).all()
    out = Path(args.out) if args.out else p(cfg, "papers")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    info["source"] = source
    info["n_papers"] = len(df)
    with open(out.with_suffix(".info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    save_statistics(df, info, cfg)
    log.info("wrote %d papers (%d-%d) to %s", len(df), df.year.min(), df.year.max(), out)


if __name__ == "__main__":
    main()
