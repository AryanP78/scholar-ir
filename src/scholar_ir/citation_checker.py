"""IR-powered citation verifier for generated answers (tf-idf / cosine, reusing the engine's own
preprocessing and idf statistics — no separate model).

For an answer sentence s and each paper p it cites:
  cos_sup(s,p) = cosine(tfidf(s), tfidf(title_p + abstract_p))   (ltc weights on both sides)
  cov_sup(s,p) = sum idf(t) over content terms t of s that occur in p / sum idf(t) over content terms of s
  support(s)   = max over cited p of  alpha * cos_sup + (1 - alpha) * cov_sup
Flags: no citation; fabricated citation (ID not among the retrieved papers); numbers in s that do
not occur in any cited abstract; unsupported if support(s) < tau.
Known blind spot (measured in the evaluation): a correct paraphrase shares few words with the
abstract and gets a low score.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .index import Index

CITE = re.compile(r"\[(?:arXiv:)?\s*([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})(?:v\d+)?\s*\]", re.I)
CITE_GROUP = re.compile(r"\[[^\]]*arXiv:[^\]]*\]", re.I)
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")


def split_sentences(text: str) -> list[str]:
    """Split after sentence-final punctuation that follows a citation group or a word."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])|(?<=\])\s+(?=[A-Z])", text)
    return [s.strip() for s in parts if s.strip()]


def cited_ids(sentence: str) -> list[str]:
    ids = []
    for grp in CITE_GROUP.findall(sentence):
        ids += re.findall(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})", grp, re.I)
    ids += CITE.findall(sentence)
    out = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


def strip_citations(sentence: str) -> str:
    return CITE.sub(" ", CITE_GROUP.sub(" ", sentence))


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[str]
    support: float
    best_paper: str | None
    cos: float
    cov: float
    no_citation: bool
    fabricated: list[str] = field(default_factory=list)
    unmatched_numbers: list[str] = field(default_factory=list)
    unsupported: bool = False

    @property
    def flags(self) -> list[str]:
        f = []
        if self.no_citation:
            f.append("NO-CITATION")
        if self.fabricated:
            f.append("FABRICATED:" + ",".join(self.fabricated))
        if self.unmatched_numbers:
            f.append("NUMBER-NOT-IN-SOURCE:" + ",".join(self.unmatched_numbers))
        if self.unsupported:
            f.append("UNSUPPORTED")
        return f


class CitationChecker:
    def __init__(self, index: Index, alpha: float = 0.5, tau: float = 0.35):
        self.index = index
        self.alpha = alpha
        self.tau = tau
        self.zi = index.zone("all")
        self.max_idf = math.log10(index.N)

    def idf(self, term: str) -> float:
        df = self.zi.get_df(term)
        return math.log10(self.index.N / df) if df else self.max_idf

    def vector(self, text: str) -> dict[str, float]:
        tf = Counter(self.index.analyzer.terms(text))
        v = {t: (1 + math.log10(c)) * self.idf(t) for t, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / n for t, x in v.items()}

    def cos_sup(self, sentence: str, paper_text: str) -> float:
        a, b = self.vector(sentence), self.vector(paper_text)
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def cov_sup(self, sentence: str, paper_text: str) -> float:
        s_terms = set(self.index.analyzer.terms(sentence))
        if not s_terms:
            return 0.0
        p_terms = set(self.index.analyzer.terms(paper_text))
        tot = sum(self.idf(t) for t in s_terms)
        return sum(self.idf(t) for t in s_terms & p_terms) / tot if tot else 0.0

    def check_sentence(self, sentence: str, papers: dict[str, str], alpha: float | None = None,
                       cited_override: list[str] | None = None) -> SentenceCheck:
        """papers: arXiv id -> 'title. abstract' of the RETRIEVED papers (the only legal citations)."""
        alpha = self.alpha if alpha is None else alpha
        cited = cited_override if cited_override is not None else cited_ids(sentence)
        body = strip_citations(sentence)
        fabricated = [c for c in cited if c not in papers]
        legal = [c for c in cited if c in papers]
        best, best_s, best_cos, best_cov = None, 0.0, 0.0, 0.0
        for c in legal:
            cs, cv = self.cos_sup(body, papers[c]), self.cov_sup(body, papers[c])
            s = alpha * cs + (1 - alpha) * cv
            if s > best_s or best is None:
                best, best_s, best_cos, best_cov = c, s, cs, cv
        nums = [n for n in NUM.findall(body) if not any(n.rstrip("%") in papers[c] for c in legal)]
        nums = [n for n in nums if not re.fullmatch(r"(19|20)\d{2}", n)]  # years are usually context
        chk = SentenceCheck(sentence=sentence, cited=cited, support=round(best_s, 4), best_paper=best,
                            cos=round(best_cos, 4), cov=round(best_cov, 4), no_citation=not cited,
                            fabricated=fabricated, unmatched_numbers=nums)
        chk.unsupported = (not legal) or best_s < self.tau
        return chk

    def check_answer(self, answer: str, papers: dict[str, str]) -> list[SentenceCheck]:
        return [self.check_sentence(s, papers) for s in split_sentences(answer)]
