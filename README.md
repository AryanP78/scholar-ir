# Scholar-IR

A structure-aware vertical search engine over arXiv computer-science papers (CSD358 IR hackathon,
track **T6: vertical search for science**), with an LLM layer that sits *on top of* the engine.

* **Engine (from scratch):** positional inverted indexes per zone (title, abstract, flat), parametric
  indexes (year, category), Boolean / phrase / `NEAR/k` / zone / filter queries, tf-idf cosine
  (SMART `lnc.ltc`), BM25, BM25F, heap top-K, champion lists, index elimination, skip pointers,
  variable-byte compression (measured).
* **Authority:** OpenAlex citation counts as static quality g(d), combined with relevance in a net score.
* **Novelty:** raw citations are biased toward old papers. We (1) age-normalise authority with
  **same-year cohort percentiles** and (2) make the authority weight **λ query-adaptive** (high for
  foundational requests, low for recent/specific ones), and test both claims on judged queries.
* **LLM layer (optional):** plain English → structured query (validated JSON), cited answers from
  the top-5 papers, and an **IR-based citation checker** (tf-idf cosine + idf-weighted coverage)
  that flags unsupported or fabricated citations.

No web server or frontend: the interface is the CLI and the Jupyter notebooks.

---

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # makes `python -m scholar_ir...` work from anywhere
make test                             # unit tests (index, phrase/proximity, rankers, authority, checker)
```

Python ≥ 3.10. API keys go in a git-ignored `.env` file in the repo root, never in code:

```bash
export GEMINI_API_KEY=...             # LLM layer (optional; or ANTHROPIC_API_KEY / OPENAI_API_KEY + llm.provider)
export OPENALEX_API_KEY=...           # needed for a full run: the keyless daily budget ($0.10) is too small (free key: openalex.org/settings/api)
export S2_API_KEY=...                 # optional: Semantic Scholar key (faster; the shared keyless pool is heavily rate-limited)
```

## 2. Reproduce everything

| Step | Command | Needs internet | Time (approx.) | Output |
|---|---|---|---|---|
| Corpus | `make data` | yes (Hugging Face) | 10–20 min | `data/processed/papers.parquet`, `results/data/` |
| Citations | `make citations-test` then `make citations` | yes (Semantic Scholar + OpenAlex) | 30–90 min (rate limits) | `data/processed/citations.parquet`, coverage report |
| Indexes + authority | `make index` | no | a few min | `data/processed/index.pkl`, `authority.parquet`, `results/index/` |
| Judging pool | `make pool` | no | <1 min | `eval/pool_to_judge.csv` |
| Judge | `make judge NAME=you QUERIES=F01,F02` | no | — | `eval/judgments/you.csv` |
| Merge judgments | `make merge` | no | seconds | `eval/qrels.csv`, `results/eval/agreement.csv` |
| Tune on **dev** | `make tune` | no | a few min | `results/tuning/params.json` + figures |
| Evaluate | `make eval` | no | 5–15 min | every table/figure in `results/` |
| LLM experiments | `python -m scholar_ir.llm_experiments parser\|intents\|answers\|checker` | yes (LLM API) | minutes | `results/llm/` |
| Notebooks | `make notebooks` | only for the LLM cells | minutes | executed `notebooks/*.ipynb` |

Everything is seeded (`seed: 42` in `config.yaml`); every tunable number lives in `config.yaml`.
Raw API responses (OpenAlex, LLM) are cached under `data/cache/`, so re-runs are free and identical.
`make eval` regenerates all tables and figures from `eval/qrels.csv`; the parts that need no
judgments (age-bias diagnostics, efficiency, vocabulary ablation) run even before judging.

### Data sources (credit)

* **arXiv metadata** — Cornell University, *arXiv Dataset* (Kaggle, `arxiv-metadata-oai-snapshot.json`,
  CC0), streamed from its Hugging Face mirror `CCRss/arXiv_dataset`. To use the Kaggle file instead,
  download it to `data/raw/` and run `python -m scholar_ir.data_loader --source file`.
  Subset: categories cs.IR, cs.CL, cs.LG, cs.AI, cs.CV, cs.DB, cs.SI; first submitted 2010 or later;
  abstracts ≥ 30 words; deduplicated; **80,000 papers sampled per year** (single-pass reservoir
  sampling, quota ∝ √(papers that year), seed 42), so every year is represented.
  Publication year = year of the **first** version (`versions[0].created`), not `update_date`.
* **Citation counts** — [Semantic Scholar Academic Graph API](https://www.semanticscholar.org/product/api)
  (batch lookup by arXiv id, title-checked). Its canonical paper record includes the citations of
  all versions; OpenAlex's arXiv records often hold only the preprint's share (see §6).
* **Citation graph** — [OpenAlex](https://openalex.org) (CC0) `referenced_works`, matched in three
  title-checked passes: arXiv DOI → arXiv landing page → title search; also the fallback count
  when Semantic Scholar has no record. Unmatched papers get `has_citation_data = False` and the
  cohort-median authority (0.5); coverage per year is in `results/data/citation_coverage.csv`.
* **Stop words** — NLTK English list (bundled in `src/scholar_ir/resources/`). **Stemmer** — NLTK Porter.

## 3. Using it

**Point-and-click app:** `make app` opens `notebooks/07_app.ipynb`; run its one cell. Tab *Ask a question*:
plain-English question → LLM-parsed query → top papers → cited answer with every sentence checked
(green = supported, red = flagged). Tab *Search papers*: the query language below with any ranker
and a per-result "why this paper?" score breakdown. (It is a Jupyter widget app, not a web server.)


```bash
python -m scholar_ir.cli search "dense retrieval year:>=2021 cat:cs.IR" --system S7 --explain
python -m scholar_ir.cli compare "seminal work on word embeddings" --systems S1,S3,S4,S7
python -m scholar_ir.cli boolean '"neural information retrieval" AND title:survey'
python -m scholar_ir.cli postings retrieval --zone title
python -m scholar_ir.cli nl "recent papers on RAG for question answering since 2022"
python -m scholar_ir.cli answer "How does dense passage retrieval differ from BM25?"
```

Query language: free-text words (ranked); `AND` / `OR` / `NOT` / parentheses; `"phrases"`;
`a NEAR/5 b` and `"a b" NEAR/5`; zones `title:` / `abstract:`; filters `year:2020`, `year:>=2020`,
`year:2018..2022`, `cat:cs.IR`. Bare words only rank; phrases, NEAR clauses and Boolean groups are
required; `NOT` excludes.

Systems: `S1` tf-idf lnc.ltc · `S2` BM25 · `S3` BM25F · `S3L` linear zone mix · `S4` S3 + raw citations ·
`S5` S3 + per-year citations · `S6` S3 + cohort percentile · **`S7` S6 + adaptive λ (proposed)** ·
`S8` S7 with PageRank-cohort authority · `RAND` random order (sanity check).

## 4. Judging (the part only humans can do)

1. `make pool` writes `eval/pool_to_judge.csv`: for each of the 40 queries in `eval/queries.csv`, the
   union of S1–S7's top-15, **shuffled and without system names or ranks**.
2. Each member judges some queries: `make judge NAME=alice QUERIES=F01,F02,...` (0 = irrelevant,
   1 = relevant, 2 = highly relevant; read the query's *description* first). Or fill the `grade`
   column of the CSV in a spreadsheet and save it as `eval/judgments/<name>.csv`.
3. For Cohen's κ, two members judge the same ~10 queries.
4. `make merge` → `eval/qrels.csv` (+ κ); `make tune` (dev split only) → `make eval`.

Recall is computed against the pooled judged set; documents never judged count as non-relevant.

## 5. Layout

```
config.yaml  Makefile  requirements.txt  pyproject.toml
src/scholar_ir/   data_loader  citations  preprocess  index  query_parser  boolean_search
                  ranking  authority  adaptive  net_score  champion  engine  explain
                  evaluation  experiments  judge  llm_client  llm_query_parser  rag
                  citation_checker  llm_experiments  build  cli
notebooks/        01_data 02_index 03_ranking 04_evaluation 05_llm_layer 06_demo
eval/             queries.csv  qrels.csv  nl_requests.csv  rag_questions.csv  answer_judgments.csv
results/          every table and figure (generated)
report/           report_draft.md  video_script.md  ai_use_declaration.md
tests/            pytest suite (uses two 100-record samples of the real arXiv dump + 6 hand-made docs)
scripts/          make_notebooks.py  run_notebooks.py  probe_openalex.py
```

## 6. Status

See `report/report_draft.md` §6 for limitations. Works end to end: data → citations → indexes →
all rankers → evaluation harness → LLM layer → notebooks/CLI. Needs the team: relevance judgments
(`eval/judgments/`), answer-sentence labels (`eval/answer_judgments.csv`), the failure-analysis
“cause” column, work division and the final pass over the report.

**Known data caveats (found while building):** OpenAlex merges arXiv preprints into their published
versions, so arXiv-DOI lookup alone misses many famous papers; some records carry wrong titles or
are mis-merged (e.g. the landing-page lookup for BERT returns an unrelated paper). The matcher
title-checks every non-DOI match for this reason; see `src/scholar_ir/citations.py`.

## 7. Libraries and AI assistance

numpy, pandas, pyarrow, scipy, matplotlib, PyYAML, NLTK (Porter stemmer), requests, pydantic,
datasets (download only), pytest, rank_bm25 (test-time cross-check of our BM25 only), Jupyter.
Search engines/IR libraries (Elasticsearch, Lucene, Whoosh, Pyserini) are **not** used.
AI assistance is declared in `report/ai_use_declaration.md`.
