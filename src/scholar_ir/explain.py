"""Per-result score breakdowns ("why is this paper here?").

For every returned document this recomputes, from the postings, each query term's contribution:
  * tf-idf lnc.ltc: query weight w_tq (ltc), document weight w_td (lnc), and their product;
  * BM25: idf, tf, length normalisation, contribution;
  * BM25F: per-zone tf and length-normalised tf, the combined tf', idf and the saturated contribution;
plus the authority signals (citations, year, g raw / per-year / cohort / PageRank), R(q,d), lambda(q)
and the final net score.
"""
from __future__ import annotations

import math

import numpy as np

from .ranking import bm25_idf


def _tf(index, zone: str, term: str, doc: int) -> int:
    return int(len(index.zone(zone).positions_in(term, doc)))


def explain_tfidf(engine, qterms, doc: int) -> dict:
    idx = engine.index
    N = idx.N
    weights = {}
    for q in qterms:
        df = idx.zone(q.zone).get_df(q.term)
        if df:
            weights[q] = (1 + math.log10(q.qtf)) * math.log10(N / df)
    qn = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    rows = []
    for q, w in weights.items():
        tf = _tf(idx, q.zone, q.term, doc)
        wtd = (1 + math.log10(tf)) / idx.lnc_norm[doc] if tf else 0.0
        rows.append({"term": q.term, "zone": q.zone, "df": idx.zone(q.zone).get_df(q.term), "w_tq(ltc)": round(w / qn, 4),
                     "tf": tf, "w_td(lnc)": round(wtd, 4), "product": round(w / qn * wtd, 4)})
    return {"model": "lnc.ltc", "doc_norm": round(float(idx.lnc_norm[doc]), 3), "terms": rows,
            "cosine": round(sum(r["product"] for r in rows), 4)}


def explain_bm25(engine, qterms, doc: int, k1: float, b: float) -> dict:
    idx = engine.index
    zi = idx.zone("all")
    rows = []
    for q in qterms:
        z = idx.zone(q.zone)
        df = z.get_df(q.term)
        if not df:
            continue
        tf = _tf(idx, q.zone, q.term, doc)
        idf = bm25_idf(idx.N, df)
        norm = k1 * (1 - b + b * z.doc_len[doc] / z.avgdl)
        rows.append({"term": q.term, "df": df, "idf": round(idf, 3), "tf": tf,
                     "contribution": round(q.qtf * idf * tf * (k1 + 1) / (tf + norm), 4) if tf else 0.0})
    return {"model": "BM25", "k1": k1, "b": b, "doc_len": int(zi.doc_len[doc]), "avgdl": round(zi.avgdl, 1), "terms": rows}


def explain_bm25f(engine, qterms, doc: int, k1: float, w: dict, b: dict) -> dict:
    idx = engine.index
    rows = []
    for q in qterms:
        df = idx.zone("all").get_df(q.term)
        if not df:
            continue
        idf = bm25_idf(idx.N, df)
        zones = ("title", "abstract") if q.zone == "all" else (q.zone,)
        per_zone, tfp = {}, 0.0
        for z in zones:
            zi = idx.zone(z)
            tf = _tf(idx, z, q.term, doc)
            denom = 1 - b[z] + b[z] * zi.doc_len[doc] / zi.avgdl
            part = w[z] * tf / denom
            tfp += part
            per_zone[z] = {"tf": tf, "len": int(zi.doc_len[doc]), "w_z*tf/B_z": round(part, 3)}
        rows.append({"term": q.term, "idf": round(idf, 3), "zones": per_zone, "tf'": round(tfp, 3),
                     "contribution": round(q.qtf * idf * tfp / (k1 + tfp), 4)})
    return {"model": "BM25F", "k1": k1, "w": w, "b": b, "terms": rows,
            "score": round(sum(r["contribution"] for r in rows), 4)}


def explain_response(engine, resp, spec: dict, ov: dict) -> None:
    rc = engine.cfg["ranking"]
    w = dict(rc["bm25f"]["w"]); w["title"] = ov.get("w_title", w["title"])
    b = dict(rc["bm25f"]["b"])
    k1f = ov.get("k1", rc["bm25f"]["k1"])
    for r in resp.results:
        d = r.doc_id
        e: dict = {}
        base = spec["base"]
        if base == "tfidf":
            e["relevance"] = explain_tfidf(engine, resp.query_terms, d)
        elif base == "bm25":
            e["relevance"] = explain_bm25(engine, resp.query_terms, d, ov.get("k1", rc["bm25"]["k1"]),
                                          ov.get("b", rc["bm25"]["b"]))
        else:
            e["relevance"] = explain_bm25f(engine, resp.query_terms, d, k1f, w, b)
        if engine.auth is not None:
            a = {"citations": int(engine.citations[d]), "year": int(engine.years[d]),
                 "has_citation_data": bool(engine.has_cite[d])}
            for key, arr in engine.g.items():
                a[f"g_{key}"] = round(float(arr[d]), 3)
            e["authority"] = a
        if resp.lam is not None:
            e["net"] = {"R": round(r.relevance, 4), "g": round(r.authority, 4), "lambda": round(resp.lam, 3),
                        "net": round(r.score, 4),
                        "formula": f"(1-{resp.lam:.2f})*{r.relevance:.3f} + {resp.lam:.2f}*{r.authority:.3f}"}
        r.explain = e


def format_explain(resp, max_terms: int = 6) -> str:
    """Plain-text rendering used by the CLI and notebooks."""
    lines = [f"query: {resp.query!r}   parsed: {resp.parsed!r}   system: {resp.system}   candidates: {resp.n_candidates}"]
    if resp.features is not None:
        f = resp.features
        lines.append(f"intent features: found_cue={f.found_cue:.0f} recent_cue={f.recent_cue:.0f} idf_z={f.idf_z:+.2f} "
                     f"len_z={f.len_z:+.2f} flatness={f.flatness:.2f} cues={list(f.matched_cues)}  "
                     f"p_found={resp.p_found:.3f}" + (f"  lambda={resp.lam:.3f}" if resp.lam is not None else ""))
    for r in resp.results:
        lines.append(f"\n#{r.rank} [{r.arxiv_id}] ({r.year}) {r.title[:80]}   score={r.score:.4f}")
        e = r.explain or {}
        rel = e.get("relevance", {})
        for t in rel.get("terms", [])[:max_terms]:
            if rel.get("model") == "BM25F":
                zs = " ".join(f"{z}:tf={v['tf']},len={v['len']}->{v['w_z*tf/B_z']}" for z, v in t["zones"].items())
                tfp = t["tf'"]
                lines.append(f"    {t['term']:<14} idf={t['idf']:<6} {zs}  tf'={tfp}  +{t['contribution']}")
            elif rel.get("model") == "lnc.ltc":
                lines.append(f"    {t['term']:<14} w_tq={t['w_tq(ltc)']:<7} tf={t['tf']:<3} w_td={t['w_td(lnc)']:<7} +{t['product']}")
            else:
                lines.append(f"    {t['term']:<14} idf={t['idf']:<6} tf={t['tf']:<3} +{t['contribution']}")
        if "authority" in e:
            a = e["authority"]
            gs = " ".join(f"{k}={v}" for k, v in a.items() if k.startswith("g_"))
            lines.append(f"    authority: citations={a['citations']} year={a['year']} data={a['has_citation_data']}  {gs}")
        if "net" in e:
            lines.append(f"    net = {e['net']['formula']} = {e['net']['net']}")
    return "\n".join(lines)
