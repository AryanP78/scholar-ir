"""Experiments for the LLM layer (needs an LLM API key for the generation steps; scoring is offline).

    python -m scholar_ir.llm_experiments parser    # LLM vs rule-based NL->query parser on eval/nl_requests.csv
    python -m scholar_ir.llm_experiments intents   # LLM intent label per eval query -> eval/llm_intents.csv
    python -m scholar_ir.llm_experiments answers   # cited answers for eval/rag_questions.csv
                                                   #   -> eval/answers.jsonl + eval/answer_judgments.csv (to label)
    python -m scholar_ir.llm_experiments checker   # citation-checker evaluation from the labelled sentences

Corrupted sentences for the checker evaluation are injected programmatically (and labelled
'unsupported' automatically, with `injected=1`): (a) citation swap — a sentence's citation replaced
by another retrieved paper that does not support it; (b) claim swap — a sentence from a different
question's answer, keeping this answer's citation; (c) fabricated ID — a citation to an arXiv id that
was not retrieved.
"""
from __future__ import annotations

import argparse
import json
import logging
import random

import numpy as np
import pandas as pd

from .citation_checker import CitationChecker, cited_ids, split_sentences, strip_citations
from .config import load_config, p, setup_logging
from .engine import SearchEngine
from .llm_client import LLMClient
from .llm_query_parser import llm_intent, llm_parse, rule_parse
from .rag import answer

log = logging.getLogger("llm_experiments")


def _stems(engine, words) -> set[str]:
    out = set()
    for w in words:
        out.update(engine.index.analyzer.terms(w))
    return out


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(pred), tp / len(gold)
    return 2 * pr * rc / (pr + rc)


def _int(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else int(float(v))


def cmd_parser(cfg, engine, client):
    df = pd.read_csv(p(cfg, "eval_dir") / "nl_requests.csv", dtype=str).fillna("")
    year = int(engine.years.max())
    rows = []
    for _, r in df.iterrows():
        gold_terms = _stems(engine, r.gold_terms.split(";"))
        gold_cats = set(filter(None, r.gold_categories.split(";")))
        for name in ("rule", "llm"):
            if name == "rule":
                sq, src = rule_parse(r.request, year), "rule"
            else:
                if not client.available():
                    continue
                sq, src = llm_parse(r.request, client, year)
            pred_terms = _stems(engine, sq.terms + sq.phrases)
            rows.append({"request_id": r.request_id, "parser": name, "source": src, "compiled": sq.compile(),
                         "term_f1": _f1(pred_terms, gold_terms),
                         "year_min_ok": sq.year_min == _int(r.gold_year_min),
                         "year_max_ok": sq.year_max == _int(r.gold_year_max),
                         "categories_ok": set(sq.categories) == gold_cats,
                         "zone_ok": sq.zone == r.gold_zone,
                         "intent_ok": sq.intent == r.gold_intent})
    res = pd.DataFrame(rows)
    res["filters_exact"] = res.year_min_ok & res.year_max_ok & res.categories_ok & res.zone_ok
    out = p(cfg, "results_dir") / "llm"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "parser_eval_per_request.csv", index=False)
    summ = res.groupby("parser")[["term_f1", "year_min_ok", "year_max_ok", "categories_ok", "zone_ok",
                                  "filters_exact", "intent_ok"]].mean().round(3).reset_index()
    summ["fallbacks"] = res.groupby("parser")["source"].apply(lambda s: int((s == "rule-fallback").sum())).values
    summ.to_csv(out / "parser_eval.csv", index=False)
    log.info("parser evaluation (%d requests):\n%s", len(df), summ.to_string(index=False))


def cmd_intents(cfg, client):
    q = pd.read_csv(p(cfg, "eval_dir") / "queries.csv", dtype=str)
    if not client.available():
        raise SystemExit("LLM not configured")
    labs = [{"query_id": qid, "intent": llm_intent(t, client)} for qid, t in zip(q.query_id, q.text)]
    pd.DataFrame(labs).to_csv(p(cfg, "eval_dir") / "llm_intents.csv", index=False)
    lab = pd.DataFrame(labs).merge(q[["query_id", "type"]])
    agree = (lab.intent == lab.type).mean()
    log.info("LLM intent labels written; agreement with our query-type labels: %.2f", agree)


def cmd_answers(cfg, engine, client, checker):
    qs = pd.read_csv(p(cfg, "eval_dir") / "rag_questions.csv", dtype=str)
    rng = random.Random(cfg["seed"])
    records = []
    with open(p(cfg, "eval_dir") / "answers.jsonl", "w") as fh:
        for _, q in qs.iterrows():
            a = answer(q.question, engine, client, checker)
            rec = {"question_id": q.question_id, "split": q.split, "question": q.question, "parsed": a.compiled_query,
                   "parser_source": a.parser_source, "retrieved": list(a.context), "answer": a.text}
            fh.write(json.dumps(rec) + "\n")
            records.append((rec, a.context))
    rows = []
    for rec, ctx in records:
        for i, s in enumerate(split_sentences(rec["answer"])):
            rows.append({"question_id": rec["question_id"], "split": rec["split"], "sentence_id": f"{rec['question_id']}-s{i}",
                         "sentence": s, "cited": ";".join(cited_ids(s)), "injected": 0, "corruption": "",
                         "label": ""})
    # inject corrupted sentences (auto-labelled unsupported)
    base = [r for r in rows if r["cited"]]
    inj = []
    kinds = ["citation_swap"] * 6 + ["claim_swap"] * 6 + ["fabricated_id"] * 3
    ctx_by_q = {rec["question_id"]: ctx for rec, ctx in records}
    for n, kind in enumerate(kinds):
        if not base:
            break
        r = rng.choice(base)
        ctx = ctx_by_q[r["question_id"]]
        body = strip_citations(r["sentence"]).strip().rstrip(".")
        cited = r["cited"].split(";")
        if kind == "citation_swap":
            others = [k for k in ctx if k not in cited]
            if not others:
                continue
            new = f"{body} [arXiv:{rng.choice(others)}]."
        elif kind == "claim_swap":
            other = rng.choice([x for x in base if x["question_id"] != r["question_id"]])
            new = f"{strip_citations(other['sentence']).strip().rstrip('.')} [arXiv:{cited[0]}]."
        else:
            new = f"{body} [arXiv:{rng.randint(1500, 2499)}.{rng.randint(10000, 19999)}]."
        inj.append({"question_id": r["question_id"], "split": r["split"], "sentence_id": f"{r['question_id']}-x{n}",
                    "sentence": new, "cited": ";".join(cited_ids(new)), "injected": 1, "corruption": kind,
                    "label": "unsupported"})
    out = pd.DataFrame(rows + inj)
    path = p(cfg, "eval_dir") / "answer_judgments.csv"
    if path.exists():  # keep labels already entered by hand
        old = pd.read_csv(path, dtype=str).fillna("")
        keep = dict(zip(old.sentence, old.label))
        out["label"] = [l or keep.get(s, "") for s, l in zip(out.sentence, out.label)]
    out.to_csv(path, index=False)
    log.info("%d answer sentences (+%d injected) written to %s — label the `label` column "
             "(supported / partial / unsupported)", len(rows), len(inj), path)


def cmd_checker(cfg, engine, checker):
    path = p(cfg, "eval_dir") / "answer_judgments.csv"
    lab = pd.read_csv(path, dtype=str).fillna("")
    lab = lab[lab.label.isin(["supported", "partial", "unsupported"])].copy()
    if lab.empty:
        raise SystemExit("no labelled sentences in eval/answer_judgments.csv")
    answers = {json.loads(l)["question_id"]: json.loads(l) for l in open(p(cfg, "eval_dir") / "answers.jsonl")}
    abstracts = engine.papers.set_index("arxiv_id")
    rows = []
    for _, r in lab.iterrows():
        ret = answers[r.question_id]["retrieved"]
        ctx = {a: f"{abstracts.loc[a, 'title']}. {abstracts.loc[a, 'abstract']}" for a in ret if a in abstracts.index}
        c = checker.check_sentence(r.sentence, ctx, alpha=1.0)
        v = checker.check_sentence(r.sentence, ctx, alpha=0.0)
        rows.append({**r.to_dict(), "cos": c.support, "cov": v.support, "fabricated": bool(c.fabricated),
                     "no_citation": c.no_citation, "numbers_flag": bool(c.unmatched_numbers)})
    df = pd.DataFrame(rows)
    df["y_unsup_lenient"] = (df.label == "unsupported").astype(int)
    df["y_unsup_strict"] = df.label.isin(["unsupported", "partial"]).astype(int)

    def predict(d, alpha, tau):
        s = alpha * d.cos + (1 - alpha) * d.cov
        hard = d.fabricated | d.no_citation
        return ((s < tau) | hard).astype(int), s

    def prf(y, yhat):
        tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    dev, test = df[df.split == "dev"], df[df.split == "test"]
    taus = np.round(np.arange(0.05, 0.80, 0.025), 3)
    results, curves, chosen = [], [], {}
    for name, alpha in (("cosine only", 1.0), ("coverage only", 0.0), ("combined", None)):
        best = (-1, None, None)
        alphas = [alpha] if alpha is not None else [0.25, 0.5, 0.75]
        for a in alphas:
            for t in taus:
                f = prf(dev.y_unsup_lenient, predict(dev, a, t)[0])[2]
                if f > best[0]:
                    best = (f, a, t)
        _, a, t = best
        chosen[name] = {"alpha": a, "tau": float(t)}
        for split_name, d in (("dev", dev), ("test", test)):
            for target in ("y_unsup_lenient", "y_unsup_strict"):
                pr, rc, f1 = prf(d[target], predict(d, a, t)[0])
                results.append({"variant": name, "alpha": a, "tau": t, "split": split_name,
                                "target": "unsupported" if target.endswith("lenient") else "unsupported+partial",
                                "n": len(d), "positives": int(d[target].sum()), "precision": pr, "recall": rc, "F1": f1})
        for tt in taus:
            pr, rc, _ = prf(test.y_unsup_lenient, predict(test, a, tt)[0])
            curves.append({"variant": name, "tau": tt, "precision": pr, "recall": rc})
    out = p(cfg, "results_dir") / "llm"
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(results).round(3)
    res.to_csv(out / "checker_eval.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "checker_pr_curve_test.csv", index=False)
    a, t = chosen["combined"]["alpha"], chosen["combined"]["tau"]
    df["pred_combined"], df["support_combined"] = predict(df, a, t)
    df.to_csv(out / "checker_per_sentence.csv", index=False)
    # detection rate per corruption type and the misses (where it fails)
    inj = df[df.injected == "1"]
    if len(inj):
        inj.groupby("corruption")["pred_combined"].mean().rename("detected_rate").reset_index() \
            .to_csv(out / "checker_injected_detection.csv", index=False)
    fails = df[(df.pred_combined != df.y_unsup_lenient)].sort_values("support_combined")
    fails[["sentence_id", "label", "corruption", "cos", "cov", "support_combined", "sentence"]] \
        .to_csv(out / "checker_errors.csv", index=False)
    (out / "checker_params.json").write_text(json.dumps(chosen, indent=2))
    from .experiments import _plt
    plt = _plt()
    cv = pd.DataFrame(curves)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for name in cv.variant.unique():
        d = cv[cv.variant == name]
        ax.plot(d.recall, d.precision, marker=".", label=name)
    ax.set_xlabel("recall (unsupported)"); ax.set_ylabel("precision"); ax.set_title("Citation checker, test sentences")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "checker_pr_curve_test.png"); plt.close(fig)
    log.info("checker evaluation:\n%s", res[res.split == "test"].to_string(index=False))


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["parser", "intents", "answers", "checker"])
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    engine = SearchEngine.load(cfg)
    client = LLMClient(cfg)
    ck = cfg["checker"]
    checker = CitationChecker(engine.index, ck["alpha"], ck["tau"])
    log.info("LLM: provider=%s model=%s available=%s", client.provider, client.model, client.available())
    if args.command == "parser":
        cmd_parser(cfg, engine, client)
    elif args.command == "intents":
        cmd_intents(cfg, client)
    elif args.command == "answers":
        if not client.available():
            raise SystemExit("LLM not configured: set GEMINI_API_KEY (see README)")
        cmd_answers(cfg, engine, client, checker)
    elif args.command == "checker":
        cmd_checker(cfg, engine, checker)


if __name__ == "__main__":
    main()
