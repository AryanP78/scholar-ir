"""Linguistic preprocessing (Term vocabulary and postings lecture).

One pipeline, applied identically to documents and queries:
  case folding -> regex tokenization (hyphenated words emit their parts AND the joined form)
  -> stop-word removal (NLTK English list + configurable domain stop words) -> Porter stemming.

Positions are assigned BEFORE stop words are removed, so a removed stop word leaves a gap. Phrase
queries use the same gaps on the query side ("learning to rank" -> learn@0, rank@2), which lets the
positional index answer phrases containing stop words without a separate no-stoplist index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from nltk.stem import PorterStemmer

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STEMMER = PorterStemmer()


def load_stopwords() -> frozenset[str]:
    path = Path(__file__).with_name("resources") / "stopwords_en.txt"
    return frozenset(w.strip() for w in path.read_text().splitlines() if w.strip() and not w.startswith("#"))


NLTK_STOPWORDS = load_stopwords()


@lru_cache(maxsize=500_000)
def stem(word: str) -> str:
    """Porter stemmer (NLTK implementation), memoised: the vocabulary is far smaller than the token stream."""
    return _STEMMER.stem(word, to_lowercase=False)


def raw_tokens(text: str) -> list[tuple[str, int, bool]]:
    """Case-fold and tokenize. Returns (token, position, is_joined_form) triples.

    'state-of-the-art' -> state@p, of@p+1, the@p+2, art@p+3 and the joined form 'stateoftheart'@p.
    Apostrophes are dropped ("bert's" -> "berts" is avoided: we split on the apostrophe and keep the stem part).
    """
    out: list[tuple[str, int, bool]] = []
    pos = 0
    for m in _TOKEN.finditer(text.lower()):
        tok = m.group(0)
        if "'" in tok:
            tok = tok.split("'")[0]
        if "-" in tok:
            parts = [x for x in tok.split("-") if x]
            if len(parts) > 1:
                out.append(("".join(parts), pos, True))
            for part in parts:
                out.append((part, pos, False))
                pos += 1
        elif tok:
            out.append((tok, pos, False))
            pos += 1
    return out


@dataclass(frozen=True)
class Analyzer:
    """Configurable analysis chain (switches used by the stemming / stop-word ablation)."""
    stem: bool = True
    stopwords: bool = True
    domain_stopwords: frozenset[str] = field(default_factory=frozenset)
    min_token_len: int = 1

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "Analyzer":
        pc = cfg["preprocess"]
        kw = dict(stem=pc["stem"], stopwords=pc["stopwords"],
                  domain_stopwords=frozenset(pc.get("domain_stopwords", [])), min_token_len=pc["min_token_len"])
        kw.update(overrides)
        return cls(**kw)

    @property
    def stoplist(self) -> frozenset[str]:
        return (NLTK_STOPWORDS | self.domain_stopwords) if self.stopwords else frozenset()

    def analyze(self, text: str) -> list[tuple[str, int]]:
        """Text -> [(term, position)] after stop-word removal and stemming."""
        stop = self.stoplist
        out = []
        for tok, pos, _joined in raw_tokens(text):
            if len(tok) < self.min_token_len or tok in stop:
                continue
            out.append((stem(tok) if self.stem else tok, pos))
        return out

    def terms(self, text: str) -> list[str]:
        return [t for t, _ in self.analyze(text)]

    def phrase_terms(self, text: str) -> list[tuple[str, int]]:
        """Phrase analysis: like analyze() but skips joined hyphen forms so offsets stay contiguous,
        and returns positions relative to the first kept term."""
        stop = self.stoplist
        out = []
        for tok, pos, joined in raw_tokens(text):
            if joined or len(tok) < self.min_token_len or tok in stop:
                continue
            out.append((stem(tok) if self.stem else tok, pos))
        if not out:
            return []
        base = out[0][1]
        return [(t, p - base) for t, p in out]


def vocabulary_size(texts: Iterable[str], analyzer: Analyzer) -> tuple[int, int]:
    """(distinct terms, total tokens) for a text collection under an analyzer setting."""
    vocab: set[str] = set()
    n = 0
    for t in texts:
        toks = analyzer.terms(t)
        n += len(toks)
        vocab.update(toks)
    return len(vocab), n
