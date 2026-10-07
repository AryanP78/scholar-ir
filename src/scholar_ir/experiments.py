"""Experiment driver: pooling, judgment merging, DEV-only tuning, evaluation, efficiency, ablations.

    python -m scholar_ir.experiments pool          # blind judging pool for eval/queries.csv
    python -m scholar_ir.experiments merge         # eval/judgments/*.csv -> eval/qrels.csv (+ kappa)
    python -m scholar_ir.experiments tune          # DEV split only -> results/tuning/params.json
    python -m scholar_ir.experiments eval          # everything in results/ (needs eval/qrels.csv)
    python -m scholar_ir.experiments bias          # judgment-free age-bias diagnostics
    python -m scholar_ir.experiments efficiency    # latency, champion lists, index elimination, skips
    python -m scholar_ir.experiments citerec       # automatic "papers this abstract should cite" evaluation

Every number written here comes from running the engine; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation as ev
from .adaptive import FEATURES, fit_logistic
from .boolean_search import Stats, intersect_many
from .config import load_config, p, seed_everything, setup_logging
from .engine import SYSTEMS, SearchEngine, deep_merge
from .index import champion_lists

log = logging.getLogger("experiments")

MAIN_SYSTEMS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
ALL_SYSTEMS = ["RAND", "S1", "S2", "S3", "S3L", "S4", "S5", "S6", "S7", "S8"]
DEPTH = 100


# ---- helpers ------------------------------------------------------------------------------------

def load_queries(cfg) -> pd.DataFrame:
    return pd.read_csv(p(cfg, "eval_dir") / "queries.csv", dtype=str)


def qrels_path(cfg) -> Path:
    return p(cfg, "eval_dir") / "qrels.csv"


def run(engine: SearchEngine, queries: pd.DataFrame, system: str, k: int = DEPTH, **kw) -> tuple[dict, dict]:
    """Run one system over a query set. Returns ({qid: ranked doc ids}, {qid: meta})."""
    ranked, meta = {}, {}
    llm = kw.pop("llm_intents", None)
    for qid, text in zip(queries["query_id"], queries["text"]):
        r = engine.search(text, system=system, k=k, llm_intent=(llm or {}).get(qid), **kw)
        ranked[qid] = r.doc_ids
        meta[qid] = {"lambda": r.lam, "p_found": r.p_found, "ms": r.timings_ms.get("total"),
                     "features": r.features.as_dict() if r.features else None}
    return ranked, meta


def score_runs(ranked: dict, qrels: dict, queries: pd.DataFrame, engine: SearchEngine, cfg: dict,
               system: str) -> pd.DataFrame:
    rows = []
    ref_year = int(engine.years.max())
    window = cfg["evaluation"]["recent_window_years"]
    rg = cfg["evaluation"]["relevant_grade"]
    for qid, typ, split in zip(queries["query_id"], queries["type"], queries["split"]):
        grades = qrels.get(qid, {})
        m = ev.evaluate_run(ranked[qid], grades, rg) if grades else {}
        m.update(ev.bias_diagnostic(ranked[qid], engine.years, ref_year, window))
        rows.append({"system": system, "query_id": qid, "type": typ, "split": split, **m})
    return pd.DataFrame(rows)


def mean_metric(ranked, qrels, queries, metric="nDCG@10", rg=1) -> float:
    vals = [ev.evaluate_run(ranked[q], qrels.get(q, {}), rg)[metric] for q in queries["query_id"] if qrels.get(q)]
    return float(np.mean(vals)) if vals else float("nan")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    return plt


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(out) + "\n"


# ---- pooling & judgments ------------------------------------------------------------------------

def already_judged(cfg) -> dict[str, set[int]]:
    """(query, doc) pairs present in qrels.csv or in any eval/judgments/*.csv file."""
    out: dict[str, set[int]] = {}
    files = [qrels_path(cfg)] + sorted((p(cfg, "eval_dir") / "judgments").glob("*.csv"))
    for f in files:
        if f.exists():
            df = pd.read_csv(f)
            for q, d in zip(df["query_id"], df["doc_id"]):
                out.setdefault(str(q), set()).add(int(d))
    return out


def cmd_pool(cfg, engine, systems: list[str] | None = None) -> None:
    """Pooling. Incremental: pairs already judged (qrels or any judgments file) are not re-exported,
    so the pool can be grown later (e.g. S1-S3 first, S4-S7 once authority is available)."""
    queries = load_queries(cfg)
    depth = cfg["evaluation"]["pool_depth"]
    systems = systems or cfg["evaluation"]["pool_systems"]
    if not engine.g:
        systems = [s for s in systems if s in ("S1", "S2", "S3", "S3L")]
        log.warning("no authority scores yet: pooling only %s (re-run `make pool` after `make index` "
                    "to add the S4-S7 documents; already-judged pairs are skipped)", systems)
    runs = {s: run(engine, queries, s, k=depth)[0] for s in systems}
    pool = ev.make_pool(runs, depth)
    judged = already_judged(cfg)
    papers = engine.papers.set_index("doc_id")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for _, q in queries.iterrows():
        docs = [d for d in pool[q.query_id] if d not in judged.get(q.query_id, set())]
        for d in rng.permutation(docs).tolist():  # shuffled: judges must not infer system or rank
            pp = papers.loc[d]
            rows.append({"query_id": q.query_id, "query": q.text, "description": q.description, "doc_id": d,
                         "arxiv_id": pp.arxiv_id, "year": int(pp.year), "title": pp.title,
                         "abstract": pp.abstract, "grade": ""})
    out = p(cfg, "eval_dir") / "pool_to_judge.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    sizes = pd.Series({q: len(v) for q, v in pool.items()})
    log.info("pool from %s: %d query-doc pairs, %d not yet judged; %.1f docs/query (min %d, max %d) -> %s",
             ",".join(systems), sum(sizes), len(rows), sizes.mean(), sizes.min(), sizes.max(), out)


def cmd_merge(cfg) -> None:
    jdir = p(cfg, "eval_dir") / "judgments"
    frames = []
    for f in sorted(jdir.glob("*.csv")):
        df = pd.read_csv(f)
        df = df[pd.to_numeric(df["grade"], errors="coerce").notna()]
        df["grade"] = df["grade"].astype(int)
        df["annotator"] = f.stem
        frames.append(df[["query_id", "doc_id", "grade", "annotator"]])
    if not frames:
        raise SystemExit(f"no judgment files in {jdir}")
    allj = pd.concat(frames)
    bad = allj[~allj["grade"].isin([0, 1, 2])]
    if len(bad):
        raise SystemExit(f"grades must be 0/1/2; found {bad.grade.unique()}")
    # agreement on pairs judged by two annotators
    out = p(cfg, "results_dir") / "eval"
    out.mkdir(parents=True, exist_ok=True)
    piv = allj.pivot_table(index=["query_id", "doc_id"], columns="annotator", values="grade", aggfunc="first")
    ann = list(piv.columns)
    agree = []
    for i in range(len(ann)):
        for j in range(i + 1, len(ann)):
            both = piv[[ann[i], ann[j]]].dropna()
            if len(both):
                bx, by = both[ann[i]].astype(int), both[ann[j]].astype(int)
                agree.append({"annotator_a": ann[i], "annotator_b": ann[j], "pairs": len(both),
                              "kappa_graded": round(ev.cohens_kappa(bx, by), 3),
                              "kappa_binary": round(ev.cohens_kappa((bx >= 1).astype(int), (by >= 1).astype(int)), 3),
                              "raw_agreement": round(float((bx == by).mean()), 3)})
    pd.DataFrame(agree).to_csv(out / "agreement.csv", index=False)
    final = allj.groupby(["query_id", "doc_id"])["grade"].mean()
    final = np.floor(final + 0.5).astype(int).rename("grade").reset_index()  # mean grade, rounded half up
    final.to_csv(qrels_path(cfg), index=False)
    log.info("qrels: %d judgments over %d queries; agreement: %s", len(final), final.query_id.nunique(), agree)


# ---- tuning (DEV ONLY) --------------------------------------------------------------------------

def cmd_tune(cfg, engine) -> None:
    qrels = ev.load_qrels(qrels_path(cfg))
    queries = load_queries(cfg)
    dev = queries[queries.split == "dev"].reset_index(drop=True)
    rg = cfg["evaluation"]["relevant_grade"]
    out = p(cfg, "results_dir") / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    params: dict = {"ranking": {"bm25": {}, "bm25f": {"w": {}}}, "net_score": {}, "adaptive": {}}
    report: dict = {}

    # 1. BM25 k1 x b grid (S2)
    rc = cfg["ranking"]
    grid = np.zeros((len(rc["bm25"]["grid_k1"]), len(rc["bm25"]["grid_b"])))
    for i, k1 in enumerate(rc["bm25"]["grid_k1"]):
        for j, b in enumerate(rc["bm25"]["grid_b"]):
            grid[i, j] = mean_metric(run(engine, dev, "S2", overrides={"k1": k1, "b": b})[0], qrels, dev, rg=rg)
    i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
    params["ranking"]["bm25"] = {"k1": rc["bm25"]["grid_k1"][i], "b": rc["bm25"]["grid_b"][j]}
    pd.DataFrame(grid, index=rc["bm25"]["grid_k1"], columns=rc["bm25"]["grid_b"]).to_csv(out / "bm25_grid_dev_ndcg10.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(grid.shape[1]), rc["bm25"]["grid_b"]); ax.set_yticks(range(grid.shape[0]), rc["bm25"]["grid_k1"])
    ax.set_xlabel("b"); ax.set_ylabel("k1"); ax.set_title("BM25 (S2) dev nDCG@10"); ax.grid(False)
    for a in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.text(c, a, f"{grid[a, c]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "bm25_grid_dev.png"); plt.close(fig)

    # 2. BM25F title weight
    wres = {w: mean_metric(run(engine, dev, "S3", overrides={"w_title": w})[0], qrels, dev, rg=rg)
            for w in rc["bm25f"]["grid_w_title"]}
    best_w = max(wres, key=wres.get)
    params["ranking"]["bm25f"]["w"] = {"title": best_w}
    engine.cfg = deep_merge(engine.cfg, {"ranking": {"bm25f": {"w": {"title": best_w}}}})
    report["bm25f_w_title_dev"] = wres

    # 3. fixed lambda per authority variant (S4 A0, S5 A1, S6 A2) + the sweep figure
    lams = np.round(np.arange(0.0, 0.85, 0.05), 2)
    sweep = []
    for sysname in ("S4", "S5", "S6"):
        for lam in lams:
            ranked = run(engine, dev, sysname, overrides={"lambda": float(lam)})[0]
            for typ in ("foundational", "recent"):
                sub = dev[dev.type == typ]
                sweep.append({"system": sysname, "lambda": lam, "type": typ,
                              "nDCG@10": mean_metric(ranked, qrels, sub, rg=rg)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(out / "lambda_sweep_dev.csv", index=False)
    by_auth = {}
    for sysname, auth in (("S4", "A0"), ("S5", "A1"), ("S6", "A2")):
        m = sw[sw.system == sysname].groupby("lambda")["nDCG@10"].mean()
        by_auth[auth] = float(m.idxmax())
    params["net_score"]["fixed_lambda_by_authority"] = by_auth
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, typ in zip(axes, ("foundational", "recent")):
        for sysname, lab in (("S4", "A0 raw"), ("S5", "A1 per-year"), ("S6", "A2 cohort")):
            d = sw[(sw.system == sysname) & (sw.type == typ)]
            ax.plot(d["lambda"], d["nDCG@10"], marker="o", ms=3, label=lab)
        ax.set_title(f"{typ} queries (dev)"); ax.set_xlabel("λ (authority weight)")
    axes[0].set_ylabel("nDCG@10"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "lambda_sweep_dev.png"); plt.close(fig)

    # 4. adaptive lambda: logistic weights on DEV query-type labels, then lambda_min/max grid
    feats = []
    _, meta = run(engine, dev, "S3", k=10)
    for qid in dev.query_id:
        feats.append(meta[qid]["features"])
    fdf = pd.DataFrame(feats)
    # recompute raw mean idf and length (features stored standardised with the prior norm)
    prior = cfg["adaptive"]["norm"]
    raw_idf = fdf["idf_z"] * prior["idf_std"] + prior["idf_mean"]
    raw_len = fdf["len_z"] * prior["len_std"] + prior["len_mean"]
    norm = {"idf_mean": float(raw_idf.mean()), "idf_std": float(raw_idf.std() or 1.0),
            "len_mean": float(raw_len.mean()), "len_std": float(raw_len.std() or 1.0)}
    fdf["idf_z"] = (raw_idf - norm["idf_mean"]) / norm["idf_std"]
    fdf["len_z"] = (raw_len - norm["len_mean"]) / norm["len_std"]
    X = np.column_stack([np.ones(len(fdf)), fdf.found_cue, fdf.recent_cue, fdf.idf_z, fdf.len_z, fdf.flatness,
                         np.zeros(len(fdf)), np.zeros(len(fdf))])
    y = (dev.type == "foundational").astype(float).to_numpy()
    w = fit_logistic(X, y, l2=1.0)
    params["adaptive"]["norm"] = norm
    params["adaptive"]["weights"] = [round(float(v), 4) for v in w]
    engine.cfg = deep_merge(engine.cfg, {"adaptive": {"norm": norm, "weights": params["adaptive"]["weights"]}})
    pf = 1 / (1 + np.exp(-(X @ w)))
    report["intent_classifier_dev_accuracy"] = float(((pf > 0.5) == (y == 1)).mean())
    best = (-1.0, None)
    grid_rows = []
    for lo in (0.0, 0.05, 0.1, 0.2):
        for hi in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            if hi <= lo:
                continue
            engine.cfg["adaptive"]["lambda_min"], engine.cfg["adaptive"]["lambda_max"] = lo, hi
            s = mean_metric(run(engine, dev, "S7")[0], qrels, dev, rg=rg)
            grid_rows.append({"lambda_min": lo, "lambda_max": hi, "nDCG@10": s})
            if s > best[0]:
                best = (s, (lo, hi))
    pd.DataFrame(grid_rows).to_csv(out / "adaptive_lambda_grid_dev.csv", index=False)
    params["adaptive"]["lambda_min"], params["adaptive"]["lambda_max"] = best[1]

    # 5. optional: logistic weights including the LLM intent label
    llm = load_llm_intents(cfg)
    if llm:
        Xl = X.copy()
        Xl[:, 6] = [float(llm.get(q) == "foundational") for q in dev.query_id]
        Xl[:, 7] = [float(llm.get(q) == "recent") for q in dev.query_id]
        wl = fit_logistic(Xl, y, l2=1.0)
        params["adaptive"]["weights_with_llm"] = [round(float(v), 4) for v in wl]
    params["ranking"]["bm25f"]["w"]["title"] = best_w
    report.update({"bm25_best": params["ranking"]["bm25"], "fixed_lambda_by_authority": by_auth,
                   "adaptive_lambda_range": best[1], "adaptive_dev_ndcg10": best[0],
                   "features": FEATURES, "weights": params["adaptive"]["weights"]})
    (out / "params.json").write_text(json.dumps({"tuned_on": "dev", "params": params, "report": report}, indent=2))
    log.info("tuned on DEV: %s", json.dumps(report, indent=1))


def load_llm_intents(cfg) -> dict:
    f = p(cfg, "eval_dir") / "llm_intents.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f, dtype=str)
    return dict(zip(df["query_id"], df["intent"]))


# ---- judgment-free diagnostics ------------------------------------------------------------------

def cmd_bias(cfg, engine) -> None:
    """Age-bias diagnostic that needs no relevance judgments: how does the median publication year
    and the share of recent papers in the top-10 move as lambda grows, for each authority source?"""
    queries = load_queries(cfg)
    out = p(cfg, "results_dir") / "bias"
    out.mkdir(parents=True, exist_ok=True)
    ref = int(engine.years.max())
    win = cfg["evaluation"]["recent_window_years"]
    rows = []
    for sysname, auth in (("S4", "A0 raw"), ("S5", "A1 per-year"), ("S6", "A2 cohort"), ("S8", "A3 PageRank cohort")):
        for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            ranked, _ = run(engine, queries, sysname, k=10, overrides={"lambda": lam})
            for typ in ("foundational", "recent"):
                qs = queries[queries.type == typ].query_id
                ys = [engine.years[ranked[q]] for q in qs if ranked[q]]
                cites = [engine.citations[ranked[q]] for q in qs if ranked[q]]
                rows.append({"authority": auth, "lambda": lam, "type": typ,
                             "median_year@10": float(np.median(np.concatenate(ys))),
                             f"frac_last{win}y@10": float(np.mean(np.concatenate(ys) > ref - win)),
                             "median_citations@10": float(np.median(np.concatenate(cites)))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "age_bias_vs_lambda.csv", index=False)
    # S3 vs S4 vs S7 at their configured settings
    rows2 = []
    for s in ("S3", "S4", "S6", "S7"):
        ranked, meta = run(engine, queries, s, k=10)
        for typ in ("foundational", "recent"):
            qs = queries[queries.type == typ].query_id
            ys = np.concatenate([engine.years[ranked[q]] for q in qs if ranked[q]])
            rows2.append({"system": s, "type": typ, "median_year@10": float(np.median(ys)),
                          f"frac_last{win}y@10": round(float(np.mean(ys > ref - win)), 3),
                          "mean_lambda": None if meta[qs.iloc[0]]["lambda"] is None else
                          round(float(np.mean([meta[q]["lambda"] for q in qs])), 3)})
    pd.DataFrame(rows2).to_csv(out / "age_bias_by_system.csv", index=False)
    (out / "age_bias_by_system.md").write_text(to_md(pd.DataFrame(rows2)))
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    for ax, typ in zip(axes, ("foundational", "recent")):
        for auth in df.authority.unique():
            d = df[(df.authority == auth) & (df.type == typ)]
            ax.plot(d["lambda"], d["median_year@10"], marker="o", ms=3, label=auth)
        ax.set_title(f"{typ} queries"); ax.set_xlabel("λ (authority weight)")
    axes[0].set_ylabel("median publication year of top-10"); axes[1].legend(fontsize=8)
    fig.suptitle("Age bias: raw citations pull results back in time; cohort percentiles do not", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "age_bias_vs_lambda.png"); plt.close(fig)
    log.info("bias diagnostics written to %s", out)


def cmd_efficiency(cfg, engine) -> None:
    queries = load_queries(cfg)
    out = p(cfg, "results_dir") / "efficiency"
    out.mkdir(parents=True, exist_ok=True)
    # 1. latency per system (3 repetitions, first run discarded as warm-up)
    lat = []
    for s in ALL_SYSTEMS:
        if s in ("S4", "S5", "S6", "S7", "S8") and not engine.g:
            continue
        run(engine, queries, s, k=10)
        times = []
        for _ in range(3):
            for t in queries.text:
                t0 = time.perf_counter()
                engine.search(t, system=s, k=10)
                times.append((time.perf_counter() - t0) * 1e3)
        lat.append({"system": s, "label": SYSTEMS[s]["label"], "mean_ms": np.mean(times),
                    "median_ms": np.median(times), "p95_ms": np.percentile(times, 95)})
    lat = pd.DataFrame(lat).round(2)
    lat.to_csv(out / "latency.csv", index=False)
    (out / "latency.md").write_text(to_md(lat))

    # 2. champion lists: recall@10 vs the exact S3 top-10, and latency, for several r
    exact = run(engine, queries, "S3", k=10)[0]
    saved = engine.index.champions
    rows = []
    for r in (50, 100, 250, 500, 1000):
        engine.index.champions = {z: champion_lists(engine.index.zone(z), r) for z in ("all", "title", "abstract")}
        times, overlap, cands = [], [], []
        for qid, t in zip(queries.query_id, queries.text):
            t0 = time.perf_counter()
            res = engine.search(t, system="S3", k=10, mode="champion")
            times.append((time.perf_counter() - t0) * 1e3)
            overlap.append(len(set(res.doc_ids) & set(exact[qid])) / max(1, len(exact[qid])))
            cands.append(res.n_candidates)
        rows.append({"r": r, "recall@10_vs_exact": np.mean(overlap), "mean_candidates": np.mean(cands),
                     "median_ms": np.median(times)})
    engine.index.champions = saved
    times, cands = [], []
    for t in queries.text:
        t0 = time.perf_counter()
        res = engine.search(t, system="S3", k=10)
        times.append((time.perf_counter() - t0) * 1e3)
        cands.append(res.n_candidates)
    rows.append({"r": "full postings", "recall@10_vs_exact": 1.0, "mean_candidates": np.mean(cands),
                 "median_ms": np.median(times)})
    # index elimination
    times, overlap, cands = [], [], []
    for qid, t in zip(queries.query_id, queries.text):
        t0 = time.perf_counter()
        res = engine.search(t, system="S3", k=10, mode="eliminate")
        times.append((time.perf_counter() - t0) * 1e3)
        overlap.append(len(set(res.doc_ids) & set(exact[qid])) / max(1, len(exact[qid])))
        cands.append(res.n_candidates)
    rows.append({"r": "index elimination", "recall@10_vs_exact": np.mean(overlap), "mean_candidates": np.mean(cands),
                 "median_ms": np.median(times)})
    ch = pd.DataFrame(rows).round(3)
    ch.to_csv(out / "champion_lists.csv", index=False)
    (out / "champion_lists.md").write_text(to_md(ch))

    # 3. Boolean AND: postings touched with/without df ordering; time with/without skip pointers
    zi = engine.index.zone("all")
    brow = []
    for qid, t in zip(queries.query_id, queries.text):
        terms = [q for q in engine.index.analyzer.terms(t) if zi.get_df(q)]
        if len(terms) < 2:
            continue
        lists = [zi.doc_list(q).tolist() for q in terms]
        res = {}
        for name, order, skips in (("query_order", False, False), ("df_order", True, False), ("df_order+skips", True, True)):
            st = Stats()
            t0 = time.perf_counter()
            for _ in range(5):
                st = Stats()
                docs = intersect_many(lists, st, order_by_df=order, use_skips=skips)
            res[name] = (st.touched, (time.perf_counter() - t0) / 5 * 1e3, len(docs))
        brow.append({"query_id": qid, "n_terms": len(terms), "result_docs": res["df_order"][2],
                     **{f"touched_{k}": v[0] for k, v in res.items()}, **{f"ms_{k}": round(v[1], 3) for k, v in res.items()}})
    bdf = pd.DataFrame(brow)
    bdf.to_csv(out / "boolean_and_per_query.csv", index=False)
    summ = bdf[[c for c in bdf.columns if c.startswith(("touched_", "ms_"))]].agg(["mean", "median"]).round(2)
    summ.to_csv(out / "boolean_and_summary.csv")
    (out / "boolean_and_summary.md").write_text(to_md(summ.reset_index()))
    log.info("efficiency results written to %s\n%s\n%s", out, lat.to_string(index=False), ch.to_string(index=False))


def cmd_vocab(cfg, engine) -> None:
    from .preprocess import Analyzer, vocabulary_size
    out = p(cfg, "results_dir") / "index"
    out.mkdir(parents=True, exist_ok=True)
    texts = (engine.papers["title"] + " " + engine.papers["abstract"]).tolist()
    rows = []
    for stem in (False, True):
        for stop in (False, True):
            v, n = vocabulary_size(texts, Analyzer.from_config(cfg, stem=stem, stopwords=stop))
            rows.append({"stemming": stem, "stopwords_removed": stop, "vocabulary": v, "tokens": n})
    df = pd.DataFrame(rows)
    base = df[(~df.stemming) & (~df.stopwords_removed)].iloc[0]
    df["vocab_vs_raw_%"] = (100 * (df.vocabulary / base.vocabulary - 1)).round(1)
    df["tokens_vs_raw_%"] = (100 * (df.tokens / base.tokens - 1)).round(1)
    df.to_csv(out / "vocabulary_ablation.csv", index=False)
    (out / "vocabulary_ablation.md").write_text(to_md(df))
    log.info("vocabulary ablation:\n%s", df.to_string(index=False))


# ---- the main evaluation ------------------------------------------------------------------------

def alt_engine(cfg, engine, stem: bool, stop: bool) -> SearchEngine:
    """Engine over an index built with a different analyzer (cached on disk)."""
    from .index import Index, build_index
    from .preprocess import Analyzer
    path = p(cfg, "processed_dir") / f"index_stem{int(stem)}_stop{int(stop)}.pkl"
    if path.exists():
        idx = Index.load(path)
    else:
        log.info("building ablation index stem=%s stop=%s", stem, stop)
        idx = build_index(engine.papers, cfg, Analyzer.from_config(cfg, stem=stem, stopwords=stop), with_champions=False)
        idx.save(path)
    return SearchEngine(idx, engine.papers, engine.auth, cfg, engine.tuned)


def cmd_eval(cfg, engine, skip_ablation_indexes: bool = False) -> None:
    if not qrels_path(cfg).exists():
        raise SystemExit("eval/qrels.csv not found: judge eval/pool_to_judge.csv first (see README, 'Judging')")
    qrels = ev.load_qrels(qrels_path(cfg))
    queries = load_queries(cfg)
    out = p(cfg, "results_dir") / "eval"
    out.mkdir(parents=True, exist_ok=True)
    rg = cfg["evaluation"]["relevant_grade"]
    plt = _plt()

    # sanity: judged / relevant counts per query
    cnt = [{"query_id": q, "judged": len(qrels.get(q, {})),
            "relevant(>=1)": sum(1 for g in qrels.get(q, {}).values() if g >= 1),
            "highly(2)": sum(1 for g in qrels.get(q, {}).values() if g == 2)} for q in queries.query_id]
    pd.DataFrame(cnt).merge(queries[["query_id", "type", "split"]]).to_csv(out / "judged_counts.csv", index=False)

    per_q, metas = [], {}
    for s in ALL_SYSTEMS:
        ranked, meta = run(engine, queries, s)
        metas[s] = meta
        per_q.append(score_runs(ranked, qrels, queries, engine, cfg, s))
    pq = pd.concat(per_q, ignore_index=True)
    pq.to_csv(out / "per_query.csv", index=False)
    order = {s: i for i, s in enumerate(ALL_SYSTEMS)}
    for split in ("test", "dev"):
        sub = pq[pq.split == split]
        tab = pd.concat([ev.mean_table(sub, ["system", "type"]),
                         ev.mean_table(sub.assign(type="all"), ["system", "type"])])
        tab = tab.sort_values(["type", "system"], key=lambda c: c.map(order) if c.name == "system" else c)
        tab.to_csv(out / f"summary_{split}.csv", index=False)
        (out / f"summary_{split}.md").write_text(to_md(tab))
    # headline figure (TEST)
    test = pq[pq.split == "test"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    systems = [s for s in ALL_SYSTEMS if s != "RAND"]
    x = np.arange(len(systems))
    for off, typ, col in ((-0.2, "foundational", "#3b6ea5"), (0.2, "recent", "#d9822b")):
        vals = [test[(test.system == s) & (test.type == typ)]["nDCG@10"].mean() for s in systems]
        ax.bar(x + off, vals, width=0.4, label=typ, color=col)
    ax.set_xticks(x, systems); ax.set_ylabel("nDCG@10 (test)"); ax.legend()
    ax.set_title("Test-split nDCG@10 by system and query type")
    fig.tight_layout(); fig.savefig(out / "ndcg_by_system_test.png"); plt.close(fig)

    # per-query win/loss (TEST): S7 vs S3 and S7 vs S4
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, other in zip(axes, ("S3", "S4")):
        a = test[test.system == "S7"].set_index("query_id")["nDCG@10"]
        b = test[test.system == other].set_index("query_id")["nDCG@10"]
        d = (a - b).sort_values()
        cols = ["#3b6ea5" if q.startswith("F") else "#d9822b" for q in d.index]
        ax.bar(range(len(d)), d.values, color=cols)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"S7 − {other}, per test query (blue = foundational)")
        ax.set_xticks(range(len(d)), d.index, rotation=90, fontsize=6)
    axes[0].set_ylabel("Δ nDCG@10")
    fig.tight_layout(); fig.savefig(out / "win_loss_test.png"); plt.close(fig)

    # significance on TEST (all, and by type)
    sig = []
    for a, b in (("S7", "S3"), ("S7", "S4"), ("S7", "S6"), ("S4", "S3"), ("S3", "S1")):
        for typ in ("all", "foundational", "recent"):
            sub = test if typ == "all" else test[test.type == typ]
            for metric in ("nDCG@10", "P@10", "MAP"):
                xa = sub[sub.system == a].sort_values("query_id")[metric].to_numpy()
                xb = sub[sub.system == b].sort_values("query_id")[metric].to_numpy()
                bs = ev.paired_bootstrap(xa, xb, cfg["evaluation"]["bootstrap_samples"], cfg["seed"])
                sig.append({"A": a, "B": b, "type": typ, "metric": metric, "n_queries": len(xa),
                            "mean_A": xa.mean(), "mean_B": xb.mean(), **bs, "p_wilcoxon": ev.wilcoxon(xa, xb)})
    sig = pd.DataFrame(sig).round(4)
    sig.to_csv(out / "significance_test.csv", index=False)
    (out / "significance_test.md").write_text(to_md(sig[sig.metric == "nDCG@10"]))

    # ablations (TEST nDCG@10 / P@10 by type)
    abl = []

    def add(name, ranked):
        sc = score_runs(ranked, qrels, queries, engine, cfg, name)
        sc = sc[sc.split == "test"]
        row = {"variant": name}
        for typ in ("foundational", "recent", "all"):
            s2 = sc if typ == "all" else sc[sc.type == typ]
            row[f"nDCG@10_{typ}"] = s2["nDCG@10"].mean()
        row["P@10_all"] = sc["P@10"].mean()
        row["median_year@10"] = sc["median_year@10"].median()
        abl.append(row)

    for s in ("S1", "S2", "S3", "S3L", "S4", "S5", "S6", "S7", "S8"):
        add(f"{s} {SYSTEMS[s]['label']}", run(engine, queries, s)[0])
    add("S6 multiplicative net score", run(engine, queries, "S6", overrides={"combine": "multiplicative"})[0])
    add("S7 multiplicative net score", run(engine, queries, "S7", overrides={"combine": "multiplicative"})[0])
    add("BM25F + PageRank-cohort, fixed λ", run(engine, queries, "S8", overrides={"adaptive": False})[0])
    llm = load_llm_intents(cfg)
    if llm and "weights_with_llm" in engine.cfg["adaptive"]:
        saved = engine.cfg["adaptive"]["weights"]
        engine.cfg["adaptive"]["weights"] = engine.cfg["adaptive"]["weights_with_llm"]
        engine.cfg["adaptive"]["use_llm_feature"] = True
        add("S7 + LLM intent feature", run(engine, queries, "S7", llm_intents=llm)[0])
        engine.cfg["adaptive"]["weights"], engine.cfg["adaptive"]["use_llm_feature"] = saved, False
    add("S3 champion lists (r=500)", run(engine, queries, "S3", mode="champion")[0])
    add("S3 index elimination", run(engine, queries, "S3", mode="eliminate")[0])
    if not skip_ablation_indexes:
        for stem, stop in ((False, True), (True, False), (False, False)):
            alt = alt_engine(cfg, engine, stem, stop)
            add(f"S3 stem={stem} stopwords={stop}", run(alt, queries, "S3")[0])
    abl = pd.DataFrame(abl).round(4)
    abl.to_csv(out / "ablations_test.csv", index=False)
    (out / "ablations_test.md").write_text(to_md(abl))

    # failure analysis: the 5 test queries where S7 is worst relative to the better of S3 / S4
    s7 = test[test.system == "S7"].set_index("query_id")
    best_other = test[test.system.isin(["S3", "S4"])].groupby("query_id")["nDCG@10"].max()
    gap = (s7["nDCG@10"] - best_other).sort_values().head(5)
    fails = []
    for qid in gap.index:
        q = queries.set_index("query_id").loc[qid]
        rel = [d for d, g in qrels.get(qid, {}).items() if g >= rg]
        top = engine.search(q.text, system="S7", k=3)
        fails.append({"query_id": qid, "query": q.text, "type": q.type, "S7_nDCG@10": s7.loc[qid, "nDCG@10"],
                      "best_of_S3_S4": best_other[qid], "lambda": metas["S7"][qid]["lambda"],
                      "p_found": metas["S7"][qid]["p_found"], "n_relevant_judged": len(rel),
                      "relevant_without_citation_data_%": round(100 * float(np.mean([not engine.has_cite[d] for d in rel])), 1) if rel else None,
                      "S7_top3": " | ".join(f"{r.arxiv_id} ({r.year}, {r.citations} cites) {r.title[:50]}" for r in top.results),
                      "cause (fill in by hand)": ""})
    pd.DataFrame(fails).to_csv(out / "failure_analysis_test.csv", index=False)
    log.info("evaluation written to %s", out)
    log.info("TEST summary:\n%s", pd.read_csv(out / "summary_test.csv").to_string(index=False))


def abstract_query(engine: SearchEngine, doc_id: int, n_terms: int):
    """Top-n tf-idf terms of a paper's title+abstract as a free-text ParsedQuery (each stem is
    represented by one of its surface forms so the query re-analyses to the same stems)."""
    from collections import Counter
    from .preprocess import raw_tokens, stem as porter
    from .query_parser import Filters, ParsedQuery, Seq, Term
    an = engine.index.analyzer
    text = engine.titles[doc_id] + ". " + engine.papers.loc[doc_id, "abstract"]
    tf = Counter(an.terms(text))
    surface = {}
    for tok, _, joined in raw_tokens(text):
        if not joined:
            surface.setdefault(porter(tok) if an.stem else tok, tok)
    zi = engine.index.zone("all")
    w = {t: (1 + np.log10(c)) * zi.idf_log10(t) for t, c in tf.items() if t in surface and zi.get_df(t) > 1}
    top = sorted(w, key=w.get, reverse=True)[:n_terms]
    words = [surface[t] for t in top]
    return ParsedQuery(raw=" ".join(words), root=Seq([Term(x) for x in words]) if words else None,
                       filters=Filters(year_max=int(engine.years[doc_id])))


def cmd_citerec(cfg, engine, n_papers: int = 300, n_terms: int = 30, k: int = 20) -> None:
    """Automatic evaluation (no human judgments): "papers this abstract should cite".
    Query = top-30 tf-idf terms of a held-out paper; relevant = its references that are inside the
    corpus (OpenAlex referenced_works); filter = published no later than the paper; the paper itself
    is excluded. Caveat: the targets' citation counts include the held-out paper's own citation (+1)."""
    cites = pd.read_parquet(p(cfg, "citations"), columns=["doc_id", "referenced_doc_ids"])
    cites["n_in"] = cites["referenced_doc_ids"].str.len()
    min_refs = cfg["evaluation"].get("citerec_min_refs", 5)
    pool = cites[cites.n_in >= min_refs].sample(frac=1.0, random_state=cfg["seed"]).head(n_papers)
    if pool.empty:
        log.warning("citerec: no paper has >= %d in-corpus references; skipped", min_refs)
        return
    out = p(cfg, "results_dir") / "citerec"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    systems = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    for _, r in pool.iterrows():
        d = int(r.doc_id)
        rel = {int(x) for x in r.referenced_doc_ids if int(x) != d}
        pq = abstract_query(engine, d, n_terms)
        for s in systems:
            res = engine.search(pq, system=s, k=k + 1)
            ranked = [x for x in res.doc_ids if x != d][:k]
            rows.append({"doc_id": d, "year": int(engine.years[d]), "system": s, "n_refs_in_corpus": len(rel),
                         f"R@{k}": ev.recall_at(ranked, rel, k), "R@10": ev.recall_at(ranked, rel, 10),
                         "MAP": ev.average_precision(ranked, rel), "MRR": ev.reciprocal_rank(ranked, rel),
                         "median_year@10": float(np.median(engine.years[ranked[:10]])) if ranked else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out / "citerec_per_paper.csv", index=False)
    summ = df.groupby("system")[[f"R@{k}", "R@10", "MAP", "MRR", "median_year@10"]].mean().round(4).reset_index()
    summ.insert(1, "papers", df.groupby("system").size().values)
    summ.to_csv(out / "citerec_summary.csv", index=False)
    (out / "citerec_summary.md").write_text(to_md(summ))
    sig = []
    for a, b in (("S7", "S3"), ("S7", "S4"), ("S4", "S3"), ("S3", "S1")):
        xa = df[df.system == a].sort_values("doc_id")[f"R@{k}"].to_numpy()
        xb = df[df.system == b].sort_values("doc_id")[f"R@{k}"].to_numpy()
        sig.append({"A": a, "B": b, "metric": f"R@{k}", **ev.paired_bootstrap(xa, xb, cfg["evaluation"]["bootstrap_samples"], cfg["seed"]),
                    "p_wilcoxon": ev.wilcoxon(xa, xb)})
    pd.DataFrame(sig).round(4).to_csv(out / "citerec_significance.csv", index=False)
    log.info("citation recommendation (%d held-out papers):\n%s", len(pool), summ.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pool", "merge", "tune", "eval", "bias", "efficiency", "vocab", "citerec", "all"])
    ap.add_argument("--config")
    ap.add_argument("--no-tuned", action="store_true", help="ignore results/tuning/params.json")
    ap.add_argument("--skip-ablation-indexes", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    if args.command == "merge":
        return cmd_merge(cfg)
    engine = SearchEngine.load(cfg, use_tuned=not args.no_tuned and args.command not in ("tune", "pool"))
    if args.command == "pool":
        cmd_pool(cfg, engine)
    elif args.command == "tune":
        cmd_tune(cfg, engine)
    elif args.command == "eval":
        cmd_eval(cfg, engine, args.skip_ablation_indexes)
    elif args.command == "bias":
        cmd_bias(cfg, engine)
    elif args.command == "efficiency":
        cmd_efficiency(cfg, engine)
    elif args.command == "vocab":
        cmd_vocab(cfg, engine)
    elif args.command == "citerec":
        cmd_citerec(cfg, engine)
    elif args.command == "all":
        cmd_vocab(cfg, engine)
        cmd_efficiency(cfg, engine)
        cmd_bias(cfg, engine)
        if engine.g:
            cmd_citerec(cfg, engine)
        if qrels_path(cfg).exists():
            cmd_eval(cfg, engine, args.skip_ablation_indexes)
        else:
            log.warning("no eval/qrels.csv yet: skipped the judged evaluation (pool, judge, merge first)")


if __name__ == "__main__":
    main()
