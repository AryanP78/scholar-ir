"""`make index`: build every index and the authority scores from data/processed/*.parquet.

Writes data/processed/index.pkl, data/processed/authority.parquet and index statistics under
results/index/ (vocabulary and postings sizes, compression, build time, load time).
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from .authority import compute_authority
from .config import load_config, p, seed_everything, setup_logging
from .index import Index, build_index, compressed_size

log = logging.getLogger("build")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    papers = pd.read_parquet(p(cfg, "papers"))
    log.info("building indexes for %d papers", len(papers))
    t0 = time.time()
    idx = build_index(papers, cfg)
    build_s = time.time() - t0
    idx.save(p(cfg, "index"))
    t1 = time.time()
    Index.load(p(cfg, "index"))
    load_s = time.time() - t1

    out = p(cfg, "results_dir") / "index"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, zi in idx.zones.items():
        comp = compressed_size(zi)
        rows.append({"zone": name, "terms": len(zi.terms), "postings": zi.n_postings,
                     "positions": int(len(zi.positions)), "avg_doc_len": round(zi.avgdl, 1),
                     "index_MB": round(zi.nbytes() / 1e6, 1),
                     "docid_raw_MB": round(comp["docid_raw_int32_bytes"] / 1e6, 2),
                     "docid_gap_vb_MB": round(comp["docid_gap_vb_bytes"] / 1e6, 2),
                     "docid_compression_ratio": round(comp["docid_raw_int32_bytes"] / max(1, comp["docid_gap_vb_bytes"]), 2)})
    pd.DataFrame(rows).to_csv(out / "index_stats.csv", index=False)
    champs = {z: len(c) for z, c in idx.champions.items()}
    info = {"N": idx.N, "build_seconds": round(build_s, 1), "load_seconds": round(load_s, 2),
            "index_file_MB": round(p(cfg, "index").stat().st_size / 1e6, 1),
            "terms_with_champion_lists": champs, "champion_r": cfg["index"]["champion_r"],
            "years": {int(y): int(len(v)) for y, v in idx.year_postings.items()},
            "categories_indexed": len(idx.cat_postings)}
    (out / "build_info.json").write_text(json.dumps(info, indent=2))
    log.info("index: %s", json.dumps({k: v for k, v in info.items() if k != "years"}))

    cpath = p(cfg, "citations")
    if cpath.exists():
        cites = pd.read_parquet(cpath)
        auth = compute_authority(papers[["doc_id", "year"]], cites, cfg)
        auth.to_parquet(p(cfg, "processed_dir") / "authority.parquet", index=False)
        (out / "authority_info.json").write_text(json.dumps(auth.attrs, indent=2))
    else:
        log.warning("no %s yet: authority-based systems (S4-S8) unavailable until `make citations`", cpath)


if __name__ == "__main__":
    main()
