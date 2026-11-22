"""All figures for the final report, drawn from real project data. Writes report/assets/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "report" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.edgecolor": "#444", "axes.linewidth": .7})
NAVY, BLUE, LBLUE, GREY, LGREY, DARK = "#1F3A5F", "#3E6E9E", "#DCE6F1", "#7A7F87", "#EEF0F3", "#222"
ACC = "#14263F"   # dark navy for the new components (restrained palette)


def _box(ax, x, y, w, h, title, body="", fc=LGREY, ec=GREY, lw=1.0, ts=8.4, bs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.6", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=DARK)
    if body:
        ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=bs, color="#333", linespacing=1.3)


def _arr(ax, a, b, style="-|>", color="#555", ls="-", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def architecture():
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66.5); ax.axis("off")
    NEW = dict(fc="#D3DEEB", ec=ACC, lw=1.6)
    ax.text(1, 64, "Offline (build once)", fontsize=8.5, color=NAVY, fontweight="bold")
    _box(ax, 1, 49, 17, 13, "arXiv metadata", "Kaggle/HF snapshot\n80,000 CS papers\n2010-2023, 7 fields")
    _box(ax, 21, 49, 17, 13, "Preprocessing", "case folding, tokens,\nstop words, Porter\nstemming")
    _box(ax, 41, 49, 19, 13, "Indexes", "positional inverted index\ntitle / abstract zones\nyear + category postings")
    _box(ax, 63, 49, 17, 13, "Citation data", "Semantic Scholar counts\nOpenAlex references\n922,768 links", **NEW)
    _box(ax, 83, 49, 16, 13, "NB classifier", "multinomial NB\ntrained on 80k\nabstracts (7 fields)", **NEW)
    _arr(ax, (18.3, 55.5), (20.7, 55.5)); _arr(ax, (38.3, 55.5), (40.7, 55.5))
    ax.text(1, 44.2, "Search path (per query)", fontsize=8.5, color=NAVY, fontweight="bold")
    _box(ax, 1, 26, 15, 15.5, "Query", "structured syntax:\nphrases, AND/OR/NOT,\nNEAR/k, title:, year:,\ncat:")
    _box(ax, 19, 26, 16, 15.5, "Candidate\nretrieval", "\nBoolean filters +\nBM25F (title x4)\n-> top-200 root set", fc=LBLUE, ec=BLUE)
    _box(ax, 38, 26, 19, 15.5, "Topic-local HITS", "base set: root + cited\n+ citing papers; votes\nweighted by citer's\nBM25F relevance", **NEW)
    _box(ax, 60, 26, 18, 15.5, "Net score", "g = 0.5 HITS pct\n  + 0.5 global (A2)\nnet = (1-λ)R + λg\nλ from query intent", **NEW)
    _box(ax, 81, 26, 18, 15.5, "Results page", "ranked papers, score\nbars, why-this-paper,\npredicted field, hubs,\nold-vs-new compare")
    for a, b in (((16.3, 33.7), (18.7, 33.7)), ((35.3, 33.7), (37.7, 33.7)), ((57.3, 33.7), (59.7, 33.7)), ((78.3, 33.7), (80.7, 33.7))):
        _arr(ax, a, b)
    _arr(ax, (50.5, 48.7), (27, 41.8), rad=0.0)
    _arr(ax, (71.5, 48.7), (48, 41.8))
    _arr(ax, (91, 48.7), (91, 41.8), ls=(0, (3, 2)))
    ax.text(92, 45.2, "hint only", fontsize=6.6, color=GREY)
    ax.text(1, 21, "Ask path (optional, needs an LLM API key)", fontsize=8.5, color=NAVY, fontweight="bold")
    _box(ax, 1, 3, 19, 15.5, "Plain-English\nquestion", "\n\"How does dense passage\nretrieval differ from\nBM25?\"")
    _box(ax, 23, 3, 19, 15.5, "Gemini parser\n+ filter guard", "\nterms, phrases, intent\nfrom the LLM; filters\nonly if user named them", **NEW)
    _box(ax, 45, 3, 17, 15.5, "Search path", "same retrieval +\nranking (S10),\ntop-5 papers")
    _box(ax, 65, 3, 15, 15.5, "Gemini answer", "uses only the 5\nabstracts; cites\n[arXiv:ID] per\nsentence")
    _box(ax, 83, 3, 16, 15.5, "Citation checker", "tf-idf cosine +\nterm coverage per\nsentence; flags\nunsupported claims", fc=LBLUE, ec=BLUE)
    for a, b in (((20.3, 10.7), (22.7, 10.7)), ((42.3, 10.7), (44.7, 10.7)), ((62.3, 10.7), (64.7, 10.7)), ((80.3, 10.7), (82.7, 10.7))):
        _arr(ax, a, b)
    _arr(ax, (53.5, 18.8), (53.5, 25.7), ls=(0, (3, 2)))
    ax.text(54.3, 21.6, "reuses", fontsize=6.6, color=GREY)
    ax.add_patch(FancyBboxPatch((70.5, 63.4), 3, 1.8, boxstyle="round,pad=0.1", fc="#D3DEEB", ec=ACC, lw=1.4))
    ax.text(74.5, 64.3, "= new in the final project", fontsize=7.2, va="center", color="#333")
    fig.tight_layout(); fig.savefig(OUT / "fig1_architecture.png", dpi=220); plt.close(fig)


def _wrap(t, w=38):
    import textwrap
    lines = textwrap.wrap(t, w)
    return "\n".join(lines[:2]) + ("…" if len(lines) > 2 else "")


def citation_network():
    """Real subgraph for the query 'sequence to sequence learning for machine translation'."""
    from scholar_ir.engine import SearchEngine
    from scholar_ir.query_parser import parse
    from scholar_ir.ranking import query_terms
    e = SearchEngine.load()
    q = "sequence to sequence learning for machine translation"
    pq = parse(q)
    qt = [t for t in query_terms(e.index, pq) if t.term not in e.meta_stems]
    acc, _ = e._base_scores("bm25f", qt, None, {})
    r = e.search(q, system="S3", k=200)
    root = np.array(r.doc_ids)
    G = e.graph
    H = G.hits(root, weight=acc)
    auth = pd.Series(H.authority, index=root).sort_values(ascending=False)
    hub = pd.Series(H.hub, index=root).sort_values(ascending=False)
    A = auth.index[:5].tolist()
    bart = int(e.papers.index[e.papers["arxiv_id"] == "1910.13461"][0]) if "arxiv_id" in e.papers else None
    if bart is not None and bart in root and bart not in A:
        A.append(bart)
    Hs = [d for d in hub.index[:5] if d not in A][:5]
    out = G.out
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    ys_h = np.linspace(50, 8, len(Hs)); ys_a = np.linspace(52, 6, len(A))
    pos = {}
    for d, y in zip(Hs, ys_h):
        pos[d] = (22, y)
    for d, y in zip(A, ys_a):
        pos[d] = (72, y)
    for u in Hs:
        cited = set(out.indices[out.indptr[u]:out.indptr[u + 1]].tolist())
        for v in A:
            if v in cited:
                _arr(ax, (pos[u][0] + 1.5, pos[u][1]), (pos[v][0] - 1.5, pos[v][1]), color="#8FA9C6", lw=0.9)
    amax = max(auth.max(), 1e-9)
    for d in Hs:
        x, y = pos[d]
        ax.scatter([x], [y], s=60 + 500 * hub[d] / max(hub.max(), 1e-9), color=BLUE, zorder=3, edgecolor="white")
        ax.text(x - 2.5, y, f"{_wrap(e.titles[d])}\n({e.years[d]}, hub {hub[d]:.2f})", ha="right", va="center", fontsize=6.6)
    for d in A:
        x, y = pos[d]
        isb = d == bart
        ax.scatter([x], [y], s=60 + 500 * auth[d] / amax, color=(GREY if isb else ACC), zorder=3, edgecolor="white")
        lab = f"{_wrap(e.titles[d])}\n({e.years[d]}, {e.citations[d]:,} cites, authority {auth[d]:.2f})"
        ax.text(x + 2.5, y, lab, ha="left", va="center", fontsize=6.6, color=("#555" if isb else DARK))
    ax.text(22, 57, "Hubs (cite many authorities)", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    ax.text(72, 57, "Authorities in the root set", ha="center", fontsize=8.5, fontweight="bold", color=ACC)
    ax.text(50, 1.0, f"Base set: {H.base_size:,} papers, {H.edges:,} weighted citation links, {H.iterations} iterations. "
                     "Node size ~ score; grey = BART (13,310 global citations, low topic authority).",
            ha="center", fontsize=7, color="#444")
    fig.tight_layout(); fig.savefig(OUT / "fig2_citation_network.png", dpi=220); plt.close(fig)
    return {"base": H.base_size, "edges": H.edges, "iter": H.iterations,
            "bart_auth": float(auth.get(bart, np.nan)) if bart is not None else None,
            "bart_rank": int(list(auth.index).index(bart) + 1) if bart in auth.index else None,
            "top_auth": [(e.titles[d], e.years[d], float(auth[d])) for d in auth.index[:5]],
            "top_hub": [(e.titles[d], e.years[d], float(hub[d])) for d in hub.index[:5]],
            "cited_in_topic": int((H.authority > 0).sum())}


def scoring_flow():
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    NEW = dict(fc="#D3DEEB", ec=ACC, lw=1.4)
    _box(ax, 1, 12, 15, 11, "BM25F", "score each\ncandidate", fc=LBLUE, ec=BLUE)
    _box(ax, 20, 22, 16, 10, "R(q,d)", "min-max over\ntop-200")
    _box(ax, 20, 1, 16, 10, "HITS a(d)", "on topic base set\n-> percentile", **NEW)
    _box(ax, 40, 1, 16, 10, "A2(d)", "global citation\npercentile in year")
    _box(ax, 60, 1, 16, 10, "g(d)", "0.5 pct(a) + 0.5 A2", **NEW)
    _box(ax, 60, 22, 16, 10, "λ(q)", "logistic on intent\nfeatures, [0.05, 0.7]")
    _box(ax, 81, 12, 18, 11, "net(q,d)", "(1-λ)·R + λ·g\nsort -> top-k", fc=LBLUE, ec=BLUE)
    _arr(ax, (16.3, 18.5), (19.7, 26.5)); _arr(ax, (16.3, 16.5), (19.7, 6.5))
    _arr(ax, (36.3, 6), (59.7, 7.5), rad=-0.18); _arr(ax, (56.3, 6), (59.7, 6))
    _arr(ax, (36.3, 27), (80.7, 20)); _arr(ax, (76.3, 27), (80.7, 21.5)); _arr(ax, (76.3, 6), (80.7, 15))
    fig.tight_layout(); fig.savefig(OUT / "fig3_scoring.png", dpi=220); plt.close(fig)


def classifier_flow():
    fig, ax = plt.subplots(figsize=(9.6, 2.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 24); ax.axis("off")
    _box(ax, 1, 4, 16, 16, "80,000 abstracts", "label = primary\narXiv category\n(7 classes)")
    _box(ax, 21, 4, 17, 16, "Term vectors", "doc x term tf matrix\nfrom the index\n(stemmed, stopped)")
    _box(ax, 42, 4, 15, 16, "Split", "stratified\n80 / 20\n64k / 16k")
    _box(ax, 61, 4, 17, 16, "Classifiers", "NB (± MI features)\nRocchio\nkNN (k=10)", fc="#D3DEEB", ec=ACC, lw=1.4)
    _box(ax, 82, 4, 17, 16, "Evaluation", "accuracy, macro\nP / R / F1,\nconfusion matrix")
    for a, b in (((17.3, 12), (20.7, 12)), ((38.3, 12), (41.7, 12)), ((57.3, 12), (60.7, 12)), ((78.3, 12), (81.7, 12))):
        _arr(ax, a, b)
    fig.tight_layout(); fig.savefig(OUT / "fig4_classifier_flow.png", dpi=220); plt.close(fig)


def ref_coverage():
    P = pd.read_parquet(ROOT / "data/processed/papers.parquet", columns=["doc_id", "year"]).set_index("doc_id")
    rf = pd.read_parquet(ROOT / "data/processed/refs_full.parquet").set_index("doc_id")
    has = pd.Series(False, index=P.index)
    has.loc[rf.index] = rf["refs"].str.len() > 0
    cov = has.groupby(P["year"]).mean()
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.bar(cov.index, cov.values * 100, color=[BLUE if y < 2021 else GREY for y in cov.index])
    for x, v in zip(cov.index, cov.values):
        ax.text(x, v * 100 + 1.5, f"{v * 100:.0f}", ha="center", fontsize=7)
    ax.set_ylabel("papers with a reference list (%)"); ax.set_ylim(0, 80)
    ax.set_xticks(cov.index, [str(y) for y in cov.index], fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig10_ref_coverage.png", dpi=220); plt.close(fig)
    return cov.round(3).to_dict()


def results_charts():
    s = pd.read_csv(ROOT / "results/eval/summary_test.csv")
    keep = ["S1", "S3", "S4", "S7", "S9", "S10"]
    lab = {"S1": "S1\ntf-idf", "S3": "S3\nBM25F", "S4": "S4\n+raw cites", "S7": "S7\nmid-term", "S9": "S9\n+topic HITS", "S10": "S10\nHITS + λ(q)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
    for ax, m in zip(axes, ("nDCG@10", "MAP")):
        x = np.arange(len(keep))
        for i, (t, c) in enumerate((("foundational", NAVY), ("recent", "#9DB4CF"))):
            v = [s[(s.system == k) & (s.type == t)][m].iloc[0] for k in keep]
            ax.bar(x + (i - .5) * .38, v, .38, color=c, label=t)
        ax.set_xticks(x, [lab[k] for k in keep], fontsize=7.2); ax.set_ylim(.5, .95)
        ax.set_ylabel(f"{m} (20 test queries)"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig6_judged_results.png", dpi=220); plt.close(fig)

    c = pd.read_csv(ROOT / "results/citerec/citerec_summary.csv").set_index("system")
    keep = ["S1", "S3", "S4", "S6", "S7", "S8", "S9", "S10"]
    lab = {"S1": "tf-idf", "S3": "BM25F", "S4": "+raw\ncites", "S6": "+cohort\npct", "S7": "mid-term\nS7", "S8": "+Page-\nRank", "S9": "+topic\nHITS S9", "S10": "S10"}
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    v = c.loc[keep, "R@20"].to_numpy()
    cols = [NAVY if k in ("S9", "S10") else "#9DB4CF" if k in ("S4", "S6", "S7", "S8") else "#C3C7CD" for k in keep]
    ax.bar(range(len(keep)), v, color=cols)
    for i, val in enumerate(v):
        ax.text(i, val + .005, f"{val:.3f}", ha="center", fontsize=7.2)
    ax.set_xticks(range(len(keep)), [lab[k] for k in keep], fontsize=7.2)
    ax.set_ylabel("Recall@20"); ax.set_ylim(0, .32); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig7_citerec.png", dpi=220); plt.close(fig)

    r = pd.read_csv(ROOT / "results/classify/classifier_summary.csv")
    r = r[r.classifier != "Majority class (baseline)"]
    short = {"Naive Bayes (all terms)": "NB\nall terms", "Naive Bayes, top-1000 MI terms": "NB\nMI 1k",
             "Naive Bayes, top-5000 MI terms": "NB\nMI 5k", "Naive Bayes, top-20000 MI terms": "NB\nMI 20k",
             "Rocchio (centroids)": "Rocchio", "kNN (k=10, cosine)": "kNN\nk=10"}
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    x = np.arange(len(r))
    for i, (m, c, l) in enumerate((("accuracy", NAVY, "accuracy"), ("macro_F1", "#9DB4CF", "macro F1"), ("macro_recall", GREY, "macro recall"))):
        ax.bar(x + (i - 1) * .27, r[m], .27, color=c, label=l)
    ax.axhline(0.328, color=DARK, lw=1, ls="--"); ax.text(len(r) - .5, .34, "majority-class accuracy 0.328", ha="right", fontsize=7, color=DARK)
    ax.set_xticks(x, [short[k] for k in r.classifier], fontsize=7.4); ax.set_ylim(0, 1)
    ax.set_ylabel("score (16,000 test papers)"); ax.legend(fontsize=7.2, frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig8_classifiers.png", dpi=220); plt.close(fig)

    cm = pd.read_csv(ROOT / "results/classify/confusion_best.csv", index_col=0)
    M = cm.to_numpy() / cm.to_numpy().sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), cm.columns, rotation=45, fontsize=7.5); ax.set_yticks(range(7), cm.index, fontsize=7.5)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(OUT / "fig9_confusion.png", dpi=220); plt.close(fig)

    g = pd.read_csv(ROOT / "results/tuning/hits_grid_dev.csv")
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    for b, c, l in ((0.5, NAVY, "β = 0.5 (HITS + global)"), (1.0, GREY, "β = 1.0 (HITS only)")):
        d = g[g.blend == b]; ax.plot(d["lambda"], d["nDCG@10"], marker="o", color=c, label=l)
    ax.axhline(0.699, color=DARK, ls=":", lw=1); ax.text(0.6, 0.703, "BM25F alone", ha="right", fontsize=7, color=DARK)
    ax.set_xlabel("λ"); ax.set_ylabel("nDCG@10 (20 dev queries)"); ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig5_hits_grid.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    architecture(); scoring_flow(); classifier_flow(); results_charts()
    print("coverage", ref_coverage())
    print("network", citation_network())
