"""Ranking, authority and adaptive-lambda invariants."""
import math
import random

import numpy as np
import pandas as pd
import pytest

from scholar_ir.adaptive import adaptive_lambda, fit_logistic, p_foundational, IntentFeatures
from scholar_ir.authority import cohort_percentile, midrank_percentile, pagerank, raw_log
from scholar_ir.engine import SearchEngine
from scholar_ir.ranking import QueryTerm, bm25, heap_topk, query_terms, tfidf_lnc_ltc
from scholar_ir.query_parser import parse


def test_heap_topk_equals_full_sort():
    rng = random.Random(0)
    for _ in range(30):
        n = rng.randint(1, 500)
        docs = list(range(n))
        scores = [round(rng.random(), 2) for _ in docs]  # many ties
        k = rng.randint(1, 20)
        full = sorted(zip(docs, scores), key=lambda x: (-x[1], x[0]))[:k]
        assert heap_topk(docs, scores, k) == full


def test_lnc_ltc_matches_manual_cosine(index):
    pq = parse("neural retrieval")
    qt = query_terms(index, pq)
    acc, touched = tfidf_lnc_ltc(index, qt)
    zi = index.zone("all")
    N = index.N
    for d in np.flatnonzero(touched)[:10]:
        # manual: full doc vector over all terms in d
        dvec = {}
        for t in zi.terms:
            pos = zi.positions_in(t, int(d))
            if len(pos):
                dvec[t] = 1 + math.log10(len(pos))
        dn = math.sqrt(sum(v * v for v in dvec.values()))
        qvec = {q.term: (1 + math.log10(q.qtf)) * math.log10(N / zi.get_df(q.term)) for q in qt}
        qn = math.sqrt(sum(v * v for v in qvec.values()))
        cos = sum(qvec[t] / qn * dvec.get(t, 0) / dn for t in qvec)
        assert acc[d] == pytest.approx(cos, rel=1e-9)


def test_bm25_matches_rank_bm25(index, papers):
    """Cross-check against rank_bm25 (used ONLY as a reference). rank_bm25 uses a different idf
    (ln((N-df+0.5)/(df+0.5)) with an epsilon floor), so we plug its per-term idf into our scorer:
    everything else (tf saturation, length normalisation) must then agree exactly."""
    rank_bm25 = pytest.importorskip("rank_bm25")
    an = index.analyzer
    corpus = [[t for t, _ in (an.analyze(t1) + an.analyze(a))] for t1, a in zip(papers.title, papers.abstract)]
    ref = rank_bm25.BM25Okapi(corpus, k1=1.2, b=0.75)
    rng = random.Random(3)
    vocab = [t for t in index.zone("all").terms if index.zone("all").get_df(t) >= 2]
    for _ in range(20):
        q = rng.sample(vocab, 3)
        acc = np.zeros(index.N)
        for t in q:
            a, _ = bm25(index, [QueryTerm(t, "all", 1)], 1.2, 0.75, idf_fn=lambda N, df, t=t: ref.idf[t])
            acc += a
        np.testing.assert_allclose(acc, ref.get_scores(q), rtol=1e-9, atol=1e-9)


def test_midrank_and_cohort_percentile():
    assert midrank_percentile(np.array([1, 2, 2, 3])).tolist() == [0.125, 0.5, 0.5, 0.875]
    years = np.array([2010] * 4 + [2020] * 4)
    c = np.array([0, 10, 100, 1000, 0, 1, 2, 3])
    g = cohort_percentile(c, years, np.ones(8, bool), min_size=4)
    # the most-cited paper of each year gets the same percentile, regardless of absolute counts
    assert g[3] == g[7] == 0.875 and g[0] == g[4] == 0.125
    g2 = cohort_percentile(c, years, np.array([1, 1, 1, 0, 1, 1, 1, 1], bool), min_size=3, missing_value=0.5)
    assert g2[3] == 0.5


def test_raw_log_bounds():
    g = raw_log(np.array([0, 5, 1000]))
    assert g.min() == 0 and g.max() == 1


def test_pagerank_sums_to_one_and_ranks_hub():
    src = np.array([0, 1, 2, 3, 4])
    dst = np.array([4, 4, 4, 4, 0])
    pr, it = pagerank(6, src, dst, tol=1e-8, max_iter=1000)
    assert pr.sum() == pytest.approx(1.0) and pr.argmax() == 4 and it < 1000


def test_lambda_bounds():
    f = IntentFeatures(1, 0, -2, 0, 1)
    for w in ([0] * 8, [5] * 8, [-50] * 8):
        lam = adaptive_lambda(p_foundational(f, w, False), 0.0, 0.6)
        assert 0.0 <= lam <= 0.6


def test_logistic_fit_separates():
    X = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]], float)
    y = np.array([1, 1, 0, 0], float)
    w = fit_logistic(X, y, l2=0.01, iters=3000)
    p = 1 / (1 + np.exp(-(X @ w)))
    assert (p[:2] > 0.8).all() and (p[2:] < 0.2).all()


@pytest.fixture(scope="module")
def engine(index, papers, cfg):
    from scholar_ir.authority import compute_authority
    import copy
    c = copy.deepcopy(cfg)
    c["authority"]["min_cohort_size"] = 5
    rng = np.random.default_rng(0)
    cites = pd.DataFrame({"doc_id": papers.doc_id, "cited_by_count": rng.integers(0, 300, len(papers)),
                          "has_citation_data": True, "referenced_doc_ids": [[] for _ in range(len(papers))]})
    return SearchEngine(index, papers, compute_authority(papers[["doc_id", "year"]], cites, c), c)


@pytest.mark.parametrize("system", ["S1", "S2", "S3", "S3L", "S4", "S5", "S6", "S7", "S8", "RAND"])
def test_every_system_runs(engine, system):
    r = engine.search("neural information retrieval", system=system, k=5, explain=True)
    assert 0 < len(r.results) <= 5
    assert all(a.score >= b.score for a, b in zip(r.results, r.results[1:]))


def test_authority_only_reranks_candidates(engine):
    base = set(engine.search("retrieval", system="S3", k=200).doc_ids)
    for s in ("S4", "S6", "S7"):
        assert set(engine.search("retrieval", system=s, k=10).doc_ids) <= base


def test_filters_respected(engine, papers):
    r = engine.search("retrieval year:>=2020", system="S7", k=10)
    assert r.results and all(x.year >= 2020 for x in r.results)


def test_lambda_zero_equals_bm25f(engine):
    a = engine.search("neural retrieval", system="S6", k=5, overrides={"lambda": 0.0}).doc_ids
    b = engine.search("neural retrieval", system="S3", k=5).doc_ids
    assert a == b
