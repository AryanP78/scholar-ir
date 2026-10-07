import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_ir.config import load_config  # noqa: E402
from scholar_ir.data_loader import clean_text, first_created  # noqa: E402


def fixture_papers() -> pd.DataFrame:
    """The two 100-record samples of the real arXiv dump (all categories, 2007) + 6 hand-written
    documents with known phrase/proximity structure used by the positional tests."""
    rows = []
    for f in ("fixtures_sample_cs_100.jsonl", "fixtures_sample_100.jsonl"):
        for line in open(ROOT / "tests" / f):
            r = json.loads(line)
            rows.append({"arxiv_id": r["id"], "title": clean_text(r["title"]), "abstract": clean_text(r["abstract"]),
                         "categories": r["categories"].split(), "year": first_created(r).year})
    hand = [
        ("Neural information retrieval with transformers", "We study neural information retrieval models. Neural models for retrieval of information are compared.", 2019),
        ("Information retrieval, neural style", "Retrieval of neural information is not neural information retrieval at all.", 2021),
        ("Query expansion for search", "Query term expansion improves recall. We expand each query with related terms for expansion.", 2015),
        ("Learning to rank survey", "A survey of learning to rank methods for web search and ranking.", 2012),
        ("Dense passage retrieval", "Dense passage retrieval uses dual encoders; dense retrieval of passages for open-domain QA.", 2020),
        ("State-of-the-art image classification", "We report state-of-the-art results on image classification benchmarks.", 2022),
    ]
    for i, (t, a, y) in enumerate(hand):
        rows.append({"arxiv_id": f"hand.{i}", "title": t, "abstract": a, "categories": ["cs.IR"], "year": y})
    df = pd.DataFrame(rows).drop_duplicates("arxiv_id").reset_index(drop=True)
    df.insert(0, "doc_id", range(len(df)))
    return df


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def papers():
    return fixture_papers()


@pytest.fixture(scope="session")
def index(papers, cfg):
    from scholar_ir.index import build_index
    import copy
    c = copy.deepcopy(cfg)
    c["index"]["champion_r"] = 5
    return build_index(papers, c)
