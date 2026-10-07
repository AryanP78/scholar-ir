"""Interactive, blind relevance judging in the terminal.

    python -m scholar_ir.judge --annotator alice --queries F01,F02,R01

Shows one pooled (query, paper) pair at a time — title, year and abstract only, never the system
or its rank — and records a grade: 0 = irrelevant, 1 = relevant, 2 = highly relevant.
Keys: 0/1/2 grade, s skip, b back, q save and quit. Progress is saved after every answer to
eval/judgments/<annotator>.csv, so you can stop and resume any time.

Alternative: open eval/pool_to_judge.csv in a spreadsheet, fill the `grade` column, and save it as
eval/judgments/<annotator>.csv (keep the query_id, doc_id and grade columns).
"""
from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

import pandas as pd

from .config import load_config, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--queries", help="comma-separated query ids (default: all)")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pool = pd.read_csv(p(cfg, "eval_dir") / "pool_to_judge.csv")
    if args.queries:
        pool = pool[pool.query_id.isin(args.queries.split(","))]
    out = p(cfg, "eval_dir") / "judgments" / f"{args.annotator}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["query_id", "doc_id", "grade"])
    grades = {(q, int(d)): int(g) for q, d, g in zip(done.query_id, done.doc_id, done.grade)}
    items = [r for _, r in pool.iterrows() if (r.query_id, int(r.doc_id)) not in grades]
    width = min(100, shutil.get_terminal_size((100, 20)).columns)
    print(f"{len(items)} pairs to judge ({len(grades)} already saved in {out})")
    i = 0
    history = []

    def save():
        pd.DataFrame([{"query_id": q, "doc_id": d, "grade": g} for (q, d), g in grades.items()]).to_csv(out, index=False)

    while i < len(items):
        r = items[i]
        print("\n" + "=" * width)
        print(f"[{i + 1}/{len(items)}]  QUERY {r.query_id}: {r.query}")
        print(textwrap.fill(f"Relevant if: {r.description}", width))
        print("-" * width)
        print(textwrap.fill(f"{r.title}  ({r.year})", width))
        print(textwrap.fill(str(r.abstract), width))
        ans = input("grade 0/1/2 (s skip, b back, q quit) > ").strip().lower()
        if ans in ("0", "1", "2"):
            grades[(r.query_id, int(r.doc_id))] = int(ans)
            history.append(i)
            save()
            i += 1
        elif ans == "s":
            i += 1
        elif ans == "b" and history:
            i = history.pop()
            grades.pop((items[i].query_id, int(items[i].doc_id)), None)
            save()
        elif ans == "q":
            break
    save()
    print(f"saved {len(grades)} judgments to {out}")


if __name__ == "__main__":
    main()
