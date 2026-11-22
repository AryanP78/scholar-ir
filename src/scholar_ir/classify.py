"""Text classification of papers into arXiv categories (course: Naive Bayes; vector space classification).

Three classifiers written from scratch over our own inverted index (zone "all": title + abstract,
the same stemmed, stop-worded terms the search engine uses):

  NB       multinomial Naive Bayes with add-one (Laplace) smoothing
           c* = argmax_c  log P(c) + Σ_t tf(t,d) · log P(t|c),  P(t|c) = (T_ct + 1) / (Σ_t' T_ct' + |V|)
  NB-MI    the same, using only the k terms with the highest mutual information I(U_t; C_c) for some class
  Rocchio  centroid of the length-normalised tf-idf vectors of each class; assign to the nearest centroid
           (cosine)
  kNN      k nearest training documents by tf-idf cosine; vote weighted by similarity

Label = the paper's primary arXiv category among the 7 we collected (cs.LG, cs.CV, cs.CL, cs.AI, cs.SI,
cs.IR, cs.DB). Split: stratified 80/20, fixed seed. Metrics: accuracy, macro precision/recall/F1,
per-class P/R/F1 and the confusion matrix; also "lenient accuracy" (prediction is ANY of the paper's
categories), since arXiv papers often carry several.

Used in the engine to show the predicted field of a query, and to keep the LLM query parser from adding
category filters the user never asked for.

    python -m scholar_ir.classify eval        # writes results/classify/
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .config import load_config, p, setup_logging
from .index import Index

log = logging.getLogger("classify")
CLASSES = ["cs.LG", "cs.CV", "cs.CL", "cs.AI", "cs.SI", "cs.IR", "cs.DB"]


def labels(papers: pd.DataFrame) -> tuple[np.ndarray, list[set[str]]]:
    """Primary label (first listed category that is one of CLASSES) and the full label set."""
    prim, allsets = [], []
    for cats in papers["categories"]:
        cl = list(cats) if not isinstance(cats, str) else cats.split()
        m = [c for c in cl if c in CLASSES]
        prim.append(CLASSES.index(m[0]) if m else -1)
        allsets.append(set(m))
    return np.array(prim), allsets


def doc_term_matrix(index: Index, zone: str = "all") -> sparse.csr_matrix:
    """N x |V| term-frequency matrix built directly from the postings lists."""
    z = index.zone(zone)
    T = len(z.df)
    cols = np.repeat(np.arange(T, dtype=np.int64), np.diff(z.ptr))
    X = sparse.csr_matrix((z.tf.astype(np.float32), (z.doc_ids.astype(np.int64), cols)), shape=(index.N, T))
    return X


def stratified_split(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_te = int(round(len(idx) * test_frac))
        te.append(idx[:n_te])
        tr.append(idx[n_te:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


# ---- classifiers ----------------------------------------------------------------------------------

@dataclass
class NaiveBayes:
    alpha: float = 1.0
    features: np.ndarray | None = None   # term ids to use (None = all)

    def fit(self, X: sparse.csr_matrix, y: np.ndarray, n_classes: int) -> "NaiveBayes":
        if self.features is not None:
            X = X[:, self.features]
        C = n_classes
        self.log_prior = np.log(np.bincount(y, minlength=C) / len(y))
        Y = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(C, len(y)))
        Tct = np.asarray((Y @ X).todense())                       # C x V term counts per class
        self.log_cond = np.log((Tct + self.alpha) / (Tct.sum(1, keepdims=True) + self.alpha * Tct.shape[1]))
        return self

    def scores(self, X: sparse.csr_matrix) -> np.ndarray:
        if self.features is not None:
            X = X[:, self.features]
        return np.asarray(X @ self.log_cond.T) + self.log_prior

    def predict(self, X):
        return self.scores(X).argmax(1)


def mutual_information(X: sparse.csr_matrix, y: np.ndarray, n_classes: int) -> np.ndarray:
    """I(U_t; C_c) for every term t and class c (binary term occurrence), IIR eq. 13.17. Returns C x V."""
    B = (X > 0).astype(np.float64).tocsc()
    N = X.shape[0]
    n_t = np.asarray(B.sum(0)).ravel()                            # docs containing t
    out = np.zeros((n_classes, X.shape[1]))
    for c in range(n_classes):
        in_c = (y == c)
        N_c = in_c.sum()
        N11 = np.asarray(B[in_c].sum(0)).ravel()                  # t present, class c
        N10 = n_t - N11                                           # t present, not c
        N01 = N_c - N11                                           # t absent, class c
        N00 = N - N_c - N10                                       # t absent, not c
        N1_, N0_, N_1, N_0 = N11 + N10, N01 + N00, N11 + N01, N10 + N00
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = [N11 / N * np.log2(N * N11 / (N1_ * N_1)), N01 / N * np.log2(N * N01 / (N0_ * N_1)),
                     N10 / N * np.log2(N * N10 / (N1_ * N_0)), N00 / N * np.log2(N * N00 / (N0_ * N_0))]
        out[c] = np.nansum(np.where(np.isfinite(terms), terms, 0.0), axis=0)
    return out


def tfidf_rows(X: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    """(1 + log tf) * idf, each row L2-normalised (lnc-style document vectors with idf)."""
    W = X.copy().astype(np.float64)
    W.data = 1.0 + np.log(W.data)
    W = W @ sparse.diags(idf)
    norms = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ W


@dataclass
class Rocchio:
    def fit(self, W: sparse.csr_matrix, y: np.ndarray, n_classes: int) -> "Rocchio":
        Y = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(n_classes, len(y)))
        cnt = np.bincount(y, minlength=n_classes)
        cent = np.asarray((Y @ W).todense()) / cnt[:, None]
        self.centroids = cent / np.linalg.norm(cent, axis=1, keepdims=True)
        return self

    def scores(self, W):
        return np.asarray(W @ self.centroids.T)

    def predict(self, W):
        return self.scores(W).argmax(1)


@dataclass
class KNN:
    k: int = 10
    batch: int = 1000

    def fit(self, W: sparse.csr_matrix, y: np.ndarray, n_classes: int) -> "KNN":
        self.WT = W.T.tocsr()
        self.y = y
        self.C = n_classes
        return self

    def predict(self, W):
        out = np.empty(W.shape[0], dtype=np.int64)
        for s in range(0, W.shape[0], self.batch):
            S = (W[s:s + self.batch] @ self.WT).toarray()       # cosine similarities (rows normalised)
            nn = np.argpartition(-S, self.k, axis=1)[:, : self.k]
            sims = np.take_along_axis(S, nn, 1)
            votes = np.zeros((len(S), self.C))
            np.add.at(votes, (np.repeat(np.arange(len(S)), self.k), self.y[nn].ravel()), sims.ravel())
            out[s:s + len(S)] = votes.argmax(1)
        return out


# ---- metrics --------------------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray, label_sets: list[set[str]], n_classes: int) -> tuple[dict, pd.DataFrame, np.ndarray]:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n_classes), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n_classes), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n_classes), where=(prec + rec) > 0)
    lenient = np.mean([CLASSES[p_] in s for p_, s in zip(y_pred, label_sets)])
    summary = {"accuracy": float(tp.sum() / cm.sum()), "macro_precision": float(prec.mean()),
               "macro_recall": float(rec.mean()), "macro_F1": float(f1.mean()),
               "weighted_F1": float((f1 * cm.sum(1)).sum() / cm.sum()), "lenient_accuracy": float(lenient)}
    per = pd.DataFrame({"class": CLASSES[:n_classes], "support": cm.sum(1), "precision": prec, "recall": rec, "F1": f1})
    return summary, per, cm


# ---- query-time helper used by the engine / GUI --------------------------------------------------

class TopicClassifier:
    """NB trained on the whole corpus; predicts the field of a free-text query."""

    def __init__(self, index: Index, papers: pd.DataFrame, alpha: float = 1.0):
        self.index = index
        y, _ = labels(papers)
        ok = y >= 0
        X = doc_term_matrix(index)
        self.nb = NaiveBayes(alpha).fit(X[ok], y[ok], len(CLASSES))
        self.vocab = index.zone("all").vocab

    def predict(self, text: str, top: int = 2) -> list[tuple[str, float]]:
        ids = [self.vocab[t] for t in self.index.analyzer.terms(text) if t in self.vocab]
        if not ids:
            return []
        x = sparse.csr_matrix((np.ones(len(ids)), (np.zeros(len(ids), dtype=int), ids)), shape=(1, len(self.vocab)))
        s = self.nb.scores(x)[0]
        pr = np.exp(s - s.max())
        pr /= pr.sum()
        o = np.argsort(-pr)[:top]
        return [(CLASSES[i], float(pr[i])) for i in o]


# ---- experiment ---------------------------------------------------------------------------------

def cmd_eval(cfg: dict) -> None:
    import time
    out = p(cfg, "results_dir") / "classify"
    out.mkdir(parents=True, exist_ok=True)
    index = Index.load(p(cfg, "index"))
    papers = pd.read_parquet(p(cfg, "papers"), columns=["doc_id", "categories"]).sort_values("doc_id")
    y, sets = labels(papers)
    keep = np.flatnonzero(y >= 0)
    X = doc_term_matrix(index)[keep]
    y, sets = y[keep], [sets[i] for i in keep]
    tr, te = stratified_split(y, 0.2, cfg["seed"])
    C = len(CLASSES)
    df_tr = np.asarray((X[tr] > 0).sum(0)).ravel()
    idf = np.log(len(tr) / np.maximum(df_tr, 1))
    W = tfidf_rows(X, idf)
    rows, perclass, cms = [], {}, {}

    def record(name, pred, secs):
        s, per, cm = metrics(y[te], pred, [sets[i] for i in te], C)
        rows.append({"classifier": name, **s, "seconds": round(secs, 1)})
        perclass[name], cms[name] = per, cm
        log.info("%-22s acc=%.3f macroF1=%.3f (%.1fs)", name, s["accuracy"], s["macro_F1"], secs)

    t = time.time()
    nb = NaiveBayes().fit(X[tr], y[tr], C)
    record("Naive Bayes (all terms)", nb.predict(X[te]), time.time() - t)
    t = time.time()
    mi = mutual_information(X[tr], y[tr], C)
    for k in (1000, 5000, 20000):
        feats = np.argsort(-mi.max(0))[:k]
        t = time.time()
        nbk = NaiveBayes(features=np.sort(feats)).fit(X[tr], y[tr], C)
        record(f"Naive Bayes, top-{k} MI terms", nbk.predict(X[te]), time.time() - t)
    t = time.time()
    ro = Rocchio().fit(W[tr], y[tr], C)
    record("Rocchio (centroids)", ro.predict(W[te]), time.time() - t)
    t = time.time()
    kn = KNN(k=cfg.get("classify", {}).get("knn_k", 10)).fit(W[tr], y[tr], C)
    record(f"kNN (k={kn.k}, cosine)", kn.predict(W[te]), time.time() - t)
    # majority-class baseline
    record("Majority class (baseline)", np.full(len(te), np.bincount(y[tr]).argmax()), 0.0)
    # cross-check our NB against scikit-learn's MultinomialNB on the same features (if installed)
    try:
        from sklearn.naive_bayes import MultinomialNB
        sk = MultinomialNB(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
        agree = float((sk == nb.predict(X[te])).mean())
        log.info("cross-check: our NB agrees with sklearn MultinomialNB on %.4f of test docs", agree)
    except ImportError:
        agree = None
    res = pd.DataFrame(rows).round(4)
    res.to_csv(out / "classifier_summary.csv", index=False)
    pd.concat([d.assign(classifier=n) for n, d in perclass.items()]).round(4).to_csv(out / "per_class.csv", index=False)
    best = res.iloc[res["macro_F1"].idxmax()]["classifier"]
    pd.DataFrame(cms[best], index=CLASSES, columns=CLASSES).to_csv(out / "confusion_best.csv")
    top_terms = {CLASSES[c]: [index.zone("all").terms[i] for i in np.argsort(-mi[c])[:12]] for c in range(C)}
    (out / "info.json").write_text(json.dumps({"n_train": int(len(tr)), "n_test": int(len(te)),
                                               "class_counts_test": np.bincount(y[te], minlength=C).tolist(),
                                               "sklearn_nb_agreement": agree, "best_by_macro_F1": best,
                                               "top_MI_terms": top_terms}, indent=2))
    _plots(out, res, cms[best], best)
    log.info("written to %s", out)


def _plots(out, res, cm, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res[res.classifier != "Majority class (baseline)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(r))
    for i, (m, lab) in enumerate((("accuracy", "accuracy"), ("macro_F1", "macro F1"), ("lenient_accuracy", "lenient accuracy"))):
        ax.bar(x + (i - 1) * 0.27, r[m], 0.27, label=lab)
    ax.set_xticks(x, r["classifier"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("score (test, 20%)"); ax.legend(fontsize=8)
    ax.set_title("arXiv category classification (7 classes)")
    fig.tight_layout(); fig.savefig(out / "classifier_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, fontsize=8); ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion (row-normalised): {best}", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "confusion_best.png", dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["eval"])
    args = ap.parse_args(argv)
    cfg = load_config()
    setup_logging()
    if args.command == "eval":
        cmd_eval(cfg)


if __name__ == "__main__":
    main()
