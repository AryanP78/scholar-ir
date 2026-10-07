"""Gate 2: preprocessing, postings, positional/phrase/proximity, Boolean optimisation."""
import random

import numpy as np

from scholar_ir.boolean_search import (Stats, boolean_search, difference, intersect, intersect_many,
                                       intersect_with_skips, smallest_window, union)
from scholar_ir.index import vb_decode, vb_encode
from scholar_ir.preprocess import Analyzer, raw_tokens, stem
from scholar_ir.query_parser import parse


def test_tokenizer_hyphen_and_case():
    toks = raw_tokens("State-of-the-Art BERT's results")
    assert ("stateoftheart", 0, True) in toks
    assert [t for t, _, j in toks if not j] == ["state", "of", "the", "art", "bert", "results"]


def test_stemming_and_stopwords():
    a = Analyzer(stem=True, stopwords=True)
    assert a.terms("The retrieval of documents") == ["retriev", "document"]
    assert Analyzer(stem=False, stopwords=False).terms("The retrieval") == ["the", "retrieval"]
    assert stem("ranking") == stem("ranked") == "rank"


def test_stopword_gap_preserved_in_positions():
    a = Analyzer()
    assert a.phrase_terms("learning to rank") == [("learn", 0), ("rank", 2)]


def _brute_postings(papers, analyzer, term, zone="all"):
    out = []
    for d, (t, a) in enumerate(zip(papers.title, papers.abstract)):
        text_terms = analyzer.terms(t) + analyzer.terms(a) if zone == "all" else analyzer.terms(t if zone == "title" else a)
        tf = text_terms.count(term)
        if tf:
            out.append((d, tf))
    return out


def test_postings_match_brute_force(index, papers):
    for term in ["neural", "retriev", "graph", "algorithm", "quantum"]:
        for zone in ("all", "title", "abstract"):
            d, tf = index.zone(zone).postings(term)
            assert list(zip(d.tolist(), tf.tolist())) == _brute_postings(papers, index.analyzer, term, zone), (term, zone)


def test_df_and_N(index, papers):
    zi = index.zone("all")
    assert zi.N == len(papers)
    for t in zi.terms[:200]:
        assert zi.get_df(t) == len(zi.doc_list(t))
    assert np.all(np.diff(zi.doc_ids[zi.ptr[0]:zi.ptr[1]]) > 0)  # postings sorted


def test_phrase_only_adjacent(index, papers):
    docs, _ = boolean_search(index, parse('"neural information retrieval"'))
    ids = [papers.arxiv_id[d] for d in docs]
    assert "hand.0" in ids and "hand.1" in ids
    for d in docs:
        text = (papers.title[d] + " " + papers.abstract[d]).lower()
        assert "neural information retrieval" in text


def test_phrase_with_stopword(index, papers):
    docs, _ = boolean_search(index, parse('"learning to rank"'))
    assert [papers.arxiv_id[d] for d in docs] == ["hand.3"]


def test_proximity(index, papers):
    near, _ = boolean_search(index, parse("query NEAR/2 expansion"))
    assert "hand.2" in [papers.arxiv_id[d] for d in near]
    far, _ = boolean_search(index, parse("title:information NEAR/1 title:style"))
    assert "hand.1" not in [papers.arxiv_id[d] for d in far]          # distance 3 in the title
    close, _ = boolean_search(index, parse("title:information NEAR/3 title:style"))
    assert "hand.1" in [papers.arxiv_id[d] for d in close]


def test_boolean_and_not_zone_filters(index, papers):
    d, _ = boolean_search(index, parse("title:retrieval NOT dense"))
    ids = {papers.arxiv_id[x] for x in d}
    assert "hand.4" not in ids and {"hand.0", "hand.1"} <= ids
    d, _ = boolean_search(index, parse("retrieval year:>=2020 cat:cs.IR"))
    assert all(papers.year[x] >= 2020 for x in d) and d


def test_merge_algorithms_and_skips():
    rng = random.Random(0)
    for _ in range(50):
        a = sorted(rng.sample(range(5000), rng.randint(0, 800)))
        b = sorted(rng.sample(range(5000), rng.randint(0, 800)))
        expect = sorted(set(a) & set(b))
        assert intersect(a, b) == expect
        assert intersect_with_skips(a, b) == expect
        assert union(a, b) == sorted(set(a) | set(b))
        assert difference(a, b) == sorted(set(a) - set(b))


def test_df_ordering_touches_fewer_postings():
    rng = random.Random(1)
    short = sorted(rng.sample(range(100000), 20))
    long1 = sorted(rng.sample(range(100000), 30000))
    long2 = sorted(rng.sample(range(100000), 30000))
    s1, s2 = Stats(), Stats()
    r1 = intersect_many([long1, long2, short], s1, order_by_df=False)
    r2 = intersect_many([long1, long2, short], s2, order_by_df=True)
    assert r1 == r2 and s2.touched < s1.touched


def test_smallest_window():
    assert smallest_window([[1, 10], [4, 20], [6]]) == 5
    assert smallest_window([[1], []]) is None


def test_vbyte_roundtrip():
    nums = [0, 1, 127, 128, 824, 5, 214577, 2 ** 20]
    assert vb_decode(vb_encode(nums)) == nums
    assert vb_encode([824]) == bytes([6, 184])  # textbook example (IIR Fig. 5.8)


def test_parametric_index(index, papers):
    for y, docs in index.year_postings.items():
        assert set(docs.tolist()) == set(np.flatnonzero(papers.year.to_numpy() == y).tolist())
    assert set(index.category_docs(["cs.IR"]).tolist()) == {d for d, c in enumerate(papers.categories) if "cs.IR" in c}
