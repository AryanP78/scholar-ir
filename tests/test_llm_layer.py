"""LLM layer: parser validation/fallback, citation checker invariants (no network: a fake client)."""
import json

import pytest

from scholar_ir.citation_checker import CitationChecker, cited_ids, split_sentences
from scholar_ir.llm_query_parser import StructuredQuery, llm_parse, rule_parse


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def available(self):
        return True

    def complete(self, system, user, as_json=False, max_tokens=0):
        self.calls += 1
        return self.replies.pop(0)


def test_structured_query_compiles():
    sq = StructuredQuery(terms=["retrieval"], phrases=["dense passage"], year_min=2021, categories=["cs.IR", "bogus"],
                         intent="recent")
    assert sq.categories == ["cs.IR"]
    assert sq.compile() == 'retrieval "dense passage" year:>=2021 cat:cs.IR'


def test_llm_parse_valid_then_retry_then_fallback():
    good = json.dumps({"terms": ["bert"], "phrases": [], "zone": "all", "year_min": None, "year_max": 2020,
                       "categories": [], "intent": "specific"})
    sq, src = llm_parse("bert before 2021", FakeClient([good]), 2026)
    assert src == "llm" and sq.year_max == 2020
    sq, src = llm_parse("bert before 2021", FakeClient(["not json", good]), 2026)
    assert src == "llm-retry"
    sq, src = llm_parse("bert papers before 2021", FakeClient(["nope", "{\"zone\": \"everywhere\"}"]), 2026)
    assert src == "rule-fallback" and sq.year_max == 2020 and "bert" in sq.terms


def test_rule_parser_filters():
    sq = rule_parse("Recent NLP work on instruction tuning since 2022", 2026)
    assert sq.year_min == 2022 and sq.categories == ["cs.CL"] and sq.intent == "recent"
    assert set(sq.terms) == {"instruction", "tuning"}
    sq = rule_parse("Surveys of graph neural networks published before 2020", 2026)
    assert sq.year_max == 2019 and sq.intent == "foundational"


def test_citation_parsing_and_split():
    s = "Dense retrievers beat BM25 on QA [arXiv:2004.04906]. RAG grounds generation [arXiv:2005.11401, arXiv:2004.04906]."
    parts = split_sentences(s)
    assert len(parts) == 2
    assert cited_ids(parts[1]) == ["2005.11401", "2004.04906"]


def test_checker_invariants(index, papers):
    ck = CitationChecker(index, alpha=0.5, tau=0.35)
    doc = papers.iloc[-2]  # hand-written "Dense passage retrieval" doc
    pid = "2004.04906"  # a real-format arXiv id for the hand-written doc
    ctx = {pid: f"{doc.title}. {doc.abstract}", "1512.03385": "Image classification with convolutional networks."}
    same = ck.check_sentence(f"{doc.abstract} [arXiv:{pid}]", ctx)
    assert same.support > 0.9 and not same.unsupported
    unrelated = ck.check_sentence(f"Quantum chromodynamics predicts diphoton cross sections [arXiv:{pid}]", ctx)
    assert unrelated.support < 0.1 and unrelated.unsupported
    fab = ck.check_sentence("Dense retrieval works [arXiv:9999.99999].", ctx)
    assert fab.fabricated == ["9999.99999"] and fab.unsupported
    nocite = ck.check_sentence("Dense retrieval works.", ctx)
    assert nocite.no_citation
    num = ck.check_sentence(f"Dense passage retrieval improves accuracy by 42% [arXiv:{pid}].", ctx)
    assert "42%" in num.unmatched_numbers


def test_client_fails_over_and_remembers_the_working_model(cfg, tmp_path, monkeypatch):
    import copy
    from scholar_ir.llm_client import LLMClient, LLMError
    c = copy.deepcopy(cfg)
    c["llm"]["cache_dir"] = str(tmp_path)
    c["llm"]["model"]["gemini"] = ["busy-model", "good-model"]
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(LLMClient, "last_good_global", None)
    client = LLMClient(c)
    tried = []

    def fake(self, system, user, as_json, max_tokens):
        tried.append(self.model)
        if self.model == "busy-model":
            raise LLMError("Gemini HTTP 503: high demand")
        return "ok", {}
    monkeypatch.setattr(LLMClient, "_gemini", fake)
    assert client.complete("s", "u1") == "ok" and tried == ["busy-model", "good-model"]
    tried.clear()
    assert client.complete("s", "u2") == "ok" and tried == ["good-model"]  # sticky: busy model skipped
    tried.clear()
    assert client.complete("s", "u1") == "ok" and tried == []              # cached
