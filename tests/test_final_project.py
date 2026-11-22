"""Unit tests for the final-project additions: topic-local HITS and the text classifiers."""
import numpy as np
import pandas as pd
from scipy import sparse

from scholar_ir.classify import KNN, NaiveBayes, Rocchio, metrics, mutual_information, tfidf_rows
from scholar_ir.hits import CitationGraph, percentile_score


def _graph():
    # corpus docs 0..4; doc 0 and 1 are surveys citing 2 and 3; doc 4 cites 2 and an outside paper (OA 99)
    refs = pd.DataFrame({"doc_id": [0, 1, 2, 3, 4], "openalex_id": [10, 11, 12, 13, 14],
                         "refs": [[12, 13], [12, 13], [], [12], [12, 99]]})
    return CitationGraph(5, refs, max_citers=100)


def test_graph_maps_inside_and_outside_references():
    g = _graph()
    assert g.n_nodes == 6            # 5 corpus docs + 1 outside paper
    assert g.internal_edges == 6     # 0->2, 0->3, 1->2, 1->3, 3->2, 4->2
    assert g.n_edges == 7


def test_hits_authority_and_hubs():
    g = _graph()
    r = g.hits(np.array([0, 1, 2, 3, 4]))
    a, h = r.authority, r.hub
    assert a.argmax() == 2                     # most-cited paper is the top authority
    assert a[0] == 0 and a[1] == 0             # nobody cites the surveys
    assert h[0] == h[1] and h[0] > h[4]        # the surveys are the best hubs
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-6) or np.linalg.norm(a) <= 1.0


def test_relevance_weighting_silences_off_topic_voters():
    g = _graph()
    w = np.array([1.0, 0.0, 1.0, 1.0, 1.0])   # doc 1 is off topic: its citations should not count
    r_w = g.hits(np.array([0, 1, 2, 3, 4]), weight=w)
    assert r_w.hub[1] == 0


def test_exclude_removes_a_document_from_the_graph():
    g = _graph()
    full = g.hits(np.array([2, 3]))
    ex = g.hits(np.array([2, 3]), exclude=0)
    assert ex.base_size == full.base_size - 1


def test_percentile_score_zero_stays_zero():
    s = percentile_score(np.array([0.0, 0.5, 0.1, 0.0]))
    assert s[0] == 0 and s[3] == 0 and s[1] == 1.0 and 0 < s[2] < 1


def _toy_text():
    # 2 classes; vocabulary: 0 'ball', 1 'goal', 2 'vote', 3 'party'
    X = sparse.csr_matrix(np.array([[3, 2, 0, 0], [2, 3, 0, 1], [0, 0, 3, 2], [0, 1, 2, 3]], dtype=float))
    y = np.array([0, 0, 1, 1])
    return X, y


def test_naive_bayes_matches_hand_computation():
    X, y = _toy_text()
    nb = NaiveBayes(alpha=1.0).fit(X, y, 2)
    # class 0 term counts: ball 5, goal 5, vote 0, party 1 -> total 11; |V| = 4
    assert np.allclose(np.exp(nb.log_cond[0]), np.array([6, 6, 1, 2]) / 15)
    assert np.allclose(np.exp(nb.log_prior), [0.5, 0.5])
    q = sparse.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float))
    assert nb.predict(q).tolist() == [0, 1]


def test_rocchio_and_knn_on_separable_data():
    X, y = _toy_text()
    W = tfidf_rows(X, np.ones(4))
    assert Rocchio().fit(W, y, 2).predict(W).tolist() == [0, 0, 1, 1]
    assert KNN(k=1).fit(W, y, 2).predict(W).tolist() == [0, 0, 1, 1]


def test_mutual_information_prefers_discriminative_terms():
    X, y = _toy_text()
    mi = mutual_information(X, y, 2)
    assert mi[0, 0] > mi[0, 3]     # 'ball' (only class 0) beats 'party' (both classes)


def test_metrics():
    s, per, cm = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), [{"cs.LG"}] * 4, 2)
    assert s["accuracy"] == 0.75 and cm.tolist() == [[1, 1], [0, 2]]
    assert np.isclose(per.loc[1, "precision"], 2 / 3) and per.loc[0, "recall"] == 0.5
