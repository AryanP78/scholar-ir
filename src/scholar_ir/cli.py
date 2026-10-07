"""Command-line interface.

    python -m scholar_ir.cli search "dense retrieval year:>=2021" --system S7 --k 10 --explain
    python -m scholar_ir.cli compare "seminal work on word embeddings" --systems S1,S3,S4,S7
    python -m scholar_ir.cli boolean '"neural information retrieval" AND title:survey'
    python -m scholar_ir.cli postings retrieval --zone title --limit 5
    python -m scholar_ir.cli nl "recent papers on RAG for question answering since 2022"
    python -m scholar_ir.cli answer "How does dense passage retrieval differ from BM25?"
    python -m scholar_ir.cli eval        # same as `make eval`
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import load_config, setup_logging


def _engine(args):
    from .engine import SearchEngine
    return SearchEngine.load(load_config(args.config))


def cmd_search(args):
    from .explain import format_explain
    eng = _engine(args)
    r = eng.search(args.query, system=args.system, k=args.k, explain=args.explain, mode=args.mode)
    if args.explain:
        print(format_explain(r))
    else:
        print(f"parsed: {r.parsed}   system: {args.system}   candidates: {r.n_candidates}   "
              f"{'λ=%.3f ' % r.lam if r.lam is not None else ''}time: {r.timings_ms['total']:.1f} ms")
        with pd.option_context("display.width", 200, "display.max_colwidth", 90):
            print(r.table().to_string(index=False))


def cmd_compare(args):
    eng = _engine(args)
    for s in args.systems.split(","):
        r = eng.search(args.query, system=s, k=args.k)
        extra = f" λ={r.lam:.3f} p_found={r.p_found:.2f}" if r.lam is not None else ""
        print(f"\n== {s}{extra}")
        for x in r.results:
            print(f"  {x.rank:>2}. [{x.arxiv_id}] {x.year} cites={x.citations:<6} {x.title[:85]}")


def cmd_boolean(args):
    from .boolean_search import boolean_search
    from .query_parser import parse
    eng = _engine(args)
    pq = parse(args.query)
    docs, st = boolean_search(eng.index, pq, order_by_df=not args.no_df_order, use_skips=args.skips)
    print(f"parsed: {pq.to_string()}  ->  {len(docs)} docs; postings entries touched: {st.touched}; "
          f"lists (term, df) in processing order: {st.lists}")
    for d in docs[: args.k]:
        print(f"  [{eng.index.arxiv_ids[d]}] {eng.years[d]} {eng.titles[d][:90]}")


def cmd_postings(args):
    eng = _engine(args)
    zi = eng.index.zone(args.zone)
    term = (eng.index.analyzer.terms(args.term) or [args.term])[0]
    print(f"term {args.term!r} -> {term!r} in zone {args.zone}: df={zi.get_df(term)}, N={zi.N}, "
          f"idf(BM25)={zi.idf_ln(term):.3f}, champion list: {'yes' if term in eng.index.champions.get(args.zone, {}) else 'no (df<=r)'}")
    for d, tf, pos in zi.raw_postings(term, args.limit):
        print(f"  doc {d:>6} [{eng.index.arxiv_ids[d]}] tf={tf} positions={pos}")


def cmd_nl(args):
    from .llm_client import LLMClient
    from .llm_query_parser import llm_parse
    eng = _engine(args)
    sq, src = llm_parse(args.request, LLMClient(eng.cfg), int(eng.years.max()))
    print(f"[{src}] {sq.model_dump_json()}\ncompiled query: {sq.compile()}")
    r = eng.search(sq.compile() or args.request, system=args.system, k=args.k, llm_intent=sq.intent)
    print(r.table().to_string(index=False))


def cmd_answer(args):
    from .citation_checker import CitationChecker
    from .llm_client import LLMClient
    from .rag import answer
    eng = _engine(args)
    ck = eng.cfg["checker"]
    a = answer(args.question, eng, LLMClient(eng.cfg), CitationChecker(eng.index, ck["alpha"], ck["tau"]), k=args.k)
    print(a.report())


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(prog="scholar_ir", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--system", default="S7")
    s.add_argument("--k", type=int, default=10); s.add_argument("--explain", action="store_true")
    s.add_argument("--mode", default="exact", choices=["exact", "champion", "eliminate"]); s.set_defaults(fn=cmd_search)
    c = sub.add_parser("compare"); c.add_argument("query"); c.add_argument("--systems", default="S1,S3,S4,S7")
    c.add_argument("--k", type=int, default=5); c.set_defaults(fn=cmd_compare)
    b = sub.add_parser("boolean"); b.add_argument("query"); b.add_argument("--k", type=int, default=10)
    b.add_argument("--no-df-order", action="store_true"); b.add_argument("--skips", action="store_true"); b.set_defaults(fn=cmd_boolean)
    pp = sub.add_parser("postings"); pp.add_argument("term"); pp.add_argument("--zone", default="all")
    pp.add_argument("--limit", type=int, default=10); pp.set_defaults(fn=cmd_postings)
    n = sub.add_parser("nl"); n.add_argument("request"); n.add_argument("--system", default="S7")
    n.add_argument("--k", type=int, default=10); n.set_defaults(fn=cmd_nl)
    a = sub.add_parser("answer"); a.add_argument("question"); a.add_argument("--k", type=int, default=5); a.set_defaults(fn=cmd_answer)
    e = sub.add_parser("eval"); e.set_defaults(fn=lambda args: __import__("scholar_ir.experiments", fromlist=["main"]).main(["all"]))
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
