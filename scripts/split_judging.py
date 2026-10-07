"""Split the judging pool between team members, with an overlap for Cohen's kappa.

    python scripts/split_judging.py aryan rahul priya            # 6 overlap queries by default
    python scripts/split_judging.py aryan rahul priya --overlap 8

Queries are assigned greedily so that everyone gets a similar number of pairs and a mix of
foundational and recent queries. The overlap queries are judged by two people (paired round-robin).
Writes eval/to_judge/<name>.csv (spreadsheet route) and prints the `make judge` command per person.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--overlap", type=int, default=6)
args = ap.parse_args()
names = args.names
pool = pd.read_csv(ROOT / "eval" / "pool_to_judge.csv")
sizes = pool.groupby("query_id").size().sort_values(ascending=False)
qtype = {q: q[0] for q in sizes.index}
load = {n: 0 for n in names}
assign = {n: [] for n in names}
for q, n_pairs in sizes.items():  # largest first -> least-loaded person
    who = min(names, key=lambda n: (load[n], sum(qtype[x] == qtype[q] for x in assign[n])))
    assign[who].append(q)
    load[who] += n_pairs
extra = {n: [] for n in names}
if len(names) > 1 and args.overlap:
    # overlap: alternate F/R queries, each judged additionally by the next person in the list
    fq = [q for q in sizes.index if q.startswith("F")]
    rq = [q for q in sizes.index if q.startswith("R")]
    picks = [x for pair in zip(fq, rq) for x in pair][: args.overlap]
    for q in picks:
        owner = next(n for n in names if q in assign[n])
        second = min((n for n in names if n != owner), key=lambda n: load[n])  # least-loaded other person
        extra[second].append(q)
        load[second] += int(sizes[q])
out = ROOT / "eval" / "to_judge"
out.mkdir(parents=True, exist_ok=True)
for n in names:
    qs = sorted(assign[n] + extra[n])
    pool[pool.query_id.isin(qs)].to_csv(out / f"{n}.csv", index=False)
    print(f"\n{n}: {len(qs)} queries, {load[n]} pairs (overlap: {','.join(sorted(extra[n])) or 'none'})")
    print(f"  make judge NAME={n} QUERIES={','.join(qs)}")
print(f"\nSpreadsheet route: fill the grade column of eval/to_judge/<name>.csv and save it as "
      f"eval/judgments/<name>.csv", file=sys.stderr)
