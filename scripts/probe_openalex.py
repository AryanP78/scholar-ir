"""Probe which OpenAlex lookup strategies find arXiv papers whose arXiv DOI was merged away.

Run on a machine with internet:  python3 scripts/probe_openalex.py
Prints, per strategy, whether each paper was found, its cited_by_count and the request cost.
"""
import json
import os
import sys
import time
import urllib.parse

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scholar_ir.config import load_dotenv  # noqa: E402

load_dotenv()
KEY = os.environ.get("OPENALEX_API_KEY")
BASE = "https://api.openalex.org"
PAPERS = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering",
    "1301.3781": "Efficient Estimation of Word Representations in Vector Space",
}
SEL = "id,doi,title,cited_by_count,publication_year"


def get(url, params=None):
    params = dict(params or {})
    if KEY:
        params["api_key"] = KEY
    r = requests.get(url, params=params, timeout=60)
    time.sleep(0.3)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"text": r.text[:200]}


def show(label, status, data):
    if "results" in data:
        res = data["results"]
        top = res[0] if res else None
        cost = (data.get("meta") or {}).get("cost_usd")
        print(f"   {label:<34} HTTP {status}  n={len(res)}  cost={cost}  "
              + (f"-> {top.get('title','')[:60]!r} cited_by={top.get('cited_by_count')} doi={top.get('doi')}" if top else ""))
    elif "id" in data:
        print(f"   {label:<34} HTTP {status}  -> {data.get('title','')[:60]!r} cited_by={data.get('cited_by_count')} doi={data.get('doi')}")
    else:
        print(f"   {label:<34} HTTP {status}  {json.dumps(data)[:150]}")


for aid, title in PAPERS.items():
    print(f"\n== {aid}  {title}")
    s, d = get(f"{BASE}/works", {"filter": f"doi:10.48550/arxiv.{aid}", "select": SEL})
    show("1 filter doi (current method)", s, d)
    s, d = get(f"{BASE}/works/doi:10.48550/arxiv.{aid}", {"select": SEL})
    show("2 singleton /works/doi:", s, d)
    for scheme in ("https", "http"):
        s, d = get(f"{BASE}/works", {"filter": f"locations.landing_page_url:{scheme}://arxiv.org/abs/{aid}", "select": SEL})
        show(f"3 landing_page_url {scheme}", s, d)
    s, d = get(f"{BASE}/works", {"filter": f"title.search:{title}", "select": SEL, "per-page": 3})
    show("4 filter title.search", s, d)
    s, d = get(f"{BASE}/works", {"filter": f"display_name.search:{title}", "select": SEL, "per-page": 3})
    show("5 filter display_name.search", s, d)
