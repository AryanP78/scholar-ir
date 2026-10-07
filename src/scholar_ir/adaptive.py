"""Query-adaptive authority weight lambda(q) (the project's novelty, part 2).

lambda(q) = lambda_min + (lambda_max - lambda_min) * p_found(q)

p_found(q) = sigmoid(w . x(q)) estimates how much the user wants foundational / highly-regarded
work rather than the newest work. x(q) is a vector of cheap, inspectable IR features:

  found_cue   1 if the query contains a foundational cue word ("survey", "seminal", "tutorial", ...)
  recent_cue  1 if it contains a recency cue ("recent", "latest", "2024", "LLM"-era terms, ...)
  idf_z       standardised mean BM25 idf of the query terms   (low idf = broad query)
  len_z       standardised number of query terms
  flatness    score(top-10) / score(top-1) of the BM25F ranking (flat = broad / ambiguous query)
  llm_found, llm_recent   optional one-hot LLM intent label (0 when the LLM layer is off)

The logistic weights are fitted on the DEV queries only (query type = foundational is the target);
lambda_min / lambda_max are then grid-searched on DEV nDCG@10. The defaults in config.yaml are
hand-set priors used before tuning.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

import numpy as np

FEATURES = ["bias", "found_cue", "recent_cue", "idf_z", "len_z", "flatness", "llm_found", "llm_recent"]
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class IntentFeatures:
    found_cue: float
    recent_cue: float
    idf_z: float
    len_z: float
    flatness: float
    llm_found: float = 0.0
    llm_recent: float = 0.0
    matched_cues: tuple = ()

    def vector(self) -> np.ndarray:
        return np.array([1.0, self.found_cue, self.recent_cue, self.idf_z, self.len_z, self.flatness,
                         self.llm_found, self.llm_recent])

    def as_dict(self) -> dict:
        d = asdict(self)
        d["matched_cues"] = list(self.matched_cues)
        return d


def _has_cue(text: str, cue: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(cue.lower()) + r"(?![a-z0-9])", text) is not None


def cue_features(raw_query: str, acfg: dict) -> tuple[float, float, list[str]]:
    text = raw_query.lower()
    found = [c for c in acfg["foundational_cues"] if _has_cue(text, c)]
    recent = [c for c in acfg["recent_cues"] + acfg["era_terms"] if _has_cue(text, c)]
    recent += [m.group(0) for m in _YEAR.finditer(text) if int(m.group(0)) >= acfg["recent_year_cue_from"]]
    return float(bool(found)), float(bool(recent)), found + recent


def score_flatness(top_scores: list[float]) -> float:
    """score at rank 10 / score at rank 1 (1.0 = perfectly flat). Fewer than 10 hits -> last/first."""
    if not top_scores or top_scores[0] <= 0:
        return 0.0
    s = top_scores[min(9, len(top_scores) - 1)]
    return float(s / top_scores[0])


def compute_features(raw_query: str, term_idfs: list[float], top_scores: list[float], acfg: dict,
                     norm: dict, llm_intent: str | None = None) -> IntentFeatures:
    f_cue, r_cue, cues = cue_features(raw_query, acfg)
    mean_idf = float(np.mean(term_idfs)) if term_idfs else 0.0
    idf_z = (mean_idf - norm["idf_mean"]) / (norm["idf_std"] or 1.0)
    len_z = (len(term_idfs) - norm["len_mean"]) / (norm["len_std"] or 1.0)
    return IntentFeatures(found_cue=f_cue, recent_cue=r_cue, idf_z=idf_z, len_z=len_z,
                          flatness=score_flatness(top_scores),
                          llm_found=float(llm_intent == "foundational"),
                          llm_recent=float(llm_intent == "recent"), matched_cues=tuple(cues))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def p_foundational(feats: IntentFeatures, weights: list[float], use_llm: bool) -> float:
    x = feats.vector()
    w = np.asarray(weights, dtype=np.float64)
    if not use_llm:
        x = x.copy()
        x[6:] = 0.0
    return sigmoid(float(x @ w))


def adaptive_lambda(p_found: float, lam_min: float, lam_max: float) -> float:
    lam = lam_min + (lam_max - lam_min) * p_found
    return float(min(1.0, max(0.0, lam)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, lr: float = 0.1, iters: int = 5000) -> np.ndarray:
    """L2-regularised logistic regression by gradient descent (bias not penalised)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) / len(y)
        reg = l2 * w / len(y)
        reg[0] = 0.0
        w -= lr * (grad + reg)
    return w
