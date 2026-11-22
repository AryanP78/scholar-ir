"""Figures for the final report (reads results/, writes report/figures/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
BLUE, ORANGE, GREY, GREEN = "#2F6DB5", "#E07B24", "#8A8F98", "#2E8B57"


def pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, title, body, color, new=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color + ("30" if new else "14"), ec=color, lw=2.0 if new else 1.1))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=8.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=7.0, color="#333", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color="#555", lw=1.1))

    # offline row
    ax.text(1, 48.5, "OFFLINE", fontsize=8, color=GREY, fontweight="bold")
    box(1, 33, 17, 13, "arXiv metadata", "80,000 CS papers\n7 categories, 2010-2023", GREY)
    box(22, 33, 18, 13, "Preprocessing", "case fold, tokens, stop\nwords, Porter stems", GREY)
    box(44, 33, 18, 13, "Indexes", "positional + zone\n(title/abstract) +\nyear/category", GREY)
    box(66, 33, 15, 13, "Citations", "Semantic Scholar +\nOpenAlex: 922,768\nreference links", GREY)
    box(84, 33, 15, 13, "NB classifier", "trained on 80k\nabstracts\n(7 fields)", GREEN, new=True)
    arrow(18.5, 39.5, 21.5, 39.5); arrow(40.5, 39.5, 43.5, 39.5)
    # online row
    ax.text(1, 27.5, "PER QUERY", fontsize=8, color=GREY, fontweight="bold")
    box(1, 9, 15, 16, "Query", "plain text, Boolean,\nphrase, NEAR, filters\n(+ guarded LLM\nparser)", BLUE)
    box(19.5, 9, 16, 16, "BM25F ranking", "title x4, abstract x1\n(tuned on dev)\n→ top-200 root set", BLUE)
    box(39, 9, 21, 16, "Topic-local HITS", "base set = root +\ncited + citing papers\nedges weighted by\nrelevance of citer\n→ authority, hubs", ORANGE, new=True)
    box(63.5, 9, 17, 16, "Net score", "(1-λ)·R + λ·g\ng = ½ HITS + ½ age-\nfair global; λ(q)\nfrom query intent", ORANGE, new=True)
    box(84, 9, 15, 16, "Results", "top-10 + why-this-\npaper, predicted\nfield, key hubs;\ncited RAG answer", BLUE)
    for x1, x2 in ((16.5, 19), (36, 38.5), (60.5, 63), (81, 83.5)):
        arrow(x1, 17, x2, 17)
    arrow(53, 32.5, 47, 25.5)       # index -> ranking/HITS
    arrow(73.5, 32.5, 52, 25.5)     # citations -> HITS
    arrow(91.5, 32.5, 91.5, 25.5)   # classifier -> results
    arrow(49, 32.5, 29, 25.5)
    ax.text(50, 3.2, "Orange/green, thick border = added in the final project; grey/blue = mid-term system (with fixes)",
            ha="center", fontsize=7.4, color="#555")
    fig.tight_layout(); fig.savefig(OUT / "fig1_pipeline.png", dpi=200); plt.close(fig)


def main_results():
    s = pd.read_csv(ROOT / "results/eval/summary_test.csv")
    keep = ["S1", "S3", "S4", "S7", "S9", "S10"]
    lab = {"S1": "S1 tf-idf", "S3": "S3 BM25F", "S4": "S4 +raw cites", "S7": "S7 mid-term",
           "S9": "S9 +topic HITS", "S10": "S10 HITS+λ(q)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, metric in zip(axes, ("nDCG@10", "MAP")):
        x = np.arange(len(keep))
        for i, (typ, col) in enumerate((("foundational", BLUE), ("recent", ORANGE))):
            v = [s[(s.system == k) & (s.type == typ)][metric].iloc[0] for k in keep]
            ax.bar(x + (i - .5) * 0.38, v, 0.38, color=col, label=typ)
        ax.set_xticks(x, [lab[k] for k in keep], rotation=22, ha="right", fontsize=7.5)
        ax.set_ylim(0.5, 0.95); ax.set_ylabel(f"{metric} (test, 20 queries)"); ax.grid(axis="y", alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig2_judged_results.png", dpi=200); plt.close(fig)


def citerec():
    c = pd.read_csv(ROOT / "results/citerec/citerec_summary.csv").set_index("system")
    keep = ["S1", "S3", "S4", "S6", "S7", "S8", "S9", "S10"]
    lab = {"S1": "tf-idf", "S3": "BM25F", "S4": "+raw\ncitations", "S6": "+cohort\npercentile", "S7": "mid-term\n(S7)",
           "S8": "+PageRank", "S9": "+topic\nHITS (S9)", "S10": "HITS +\nλ(q) (S10)"}
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    v = c.loc[keep, "R@20"].to_numpy()
    cols = [ORANGE if k in ("S9", "S10") else BLUE if k in ("S4", "S6", "S7", "S8") else GREY for k in keep]
    ax.bar(range(len(keep)), v, color=cols)
    for i, val in enumerate(v):
        ax.text(i, val + 0.004, f"{val:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(keep)), [lab[k] for k in keep], fontsize=7.5)
    ax.set_ylabel("Recall@20"); ax.set_ylim(0, 0.32); ax.grid(axis="y", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Citation recommendation: find a held-out paper's references from its abstract (300 papers)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig3_citerec.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    pipeline(); main_results(); citerec()
    print("figures written to", OUT)
