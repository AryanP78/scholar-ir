"""Generate the six notebooks (source of truth for notebook content). Run: python scripts/make_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 90)
from scholar_ir.config import load_config, p
cfg = load_config()  # config.yaml at the repo root (or $SCHOLAR_IR_CONFIG)
R = p(cfg, "results_dir")"""

ENGINE = """from scholar_ir.engine import SearchEngine
engine = SearchEngine.load(cfg)
index, papers = engine.index, engine.papers
print(f"{index.N:,} papers, years {papers.year.min()}-{papers.year.max()}, tuned params loaded: {bool(engine.tuned)}")"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


write("01_data.ipynb", [
    md("# 01 · Data: what is a document?\nA document = one arXiv paper with two text **zones** (title, abstract) and "
       "**metadata** (first-submission year, categories). Citations come from OpenAlex. All numbers below are read "
       "from files written by `make data` / `make citations`."),
    code(SETUP),
    code("papers = pd.read_parquet(p(cfg, 'papers'))\nprint(papers.shape)\npapers.head(3)"),
    code("pd.read_csv(R / 'data/papers_per_year.csv')"),
    code("from IPython.display import Image\nImage(str(R / 'data/corpus_stats.png'))"),
    code("pd.read_csv(R / 'data/papers_per_category.csv')"),
    code("pd.read_csv(R / 'data/abstract_length.csv')"),
    md("## Citation coverage (OpenAlex)\nMatched in three passes (arXiv DOI → arXiv landing page → title search), "
       "each title-checked; see `citations.py` for the OpenAlex metadata problems this guards against."),
    code("pd.read_csv(R / 'data/citation_coverage.csv')"),
    code("pd.read_csv(R / 'data/citation_match_methods.csv')"),
    code("cites = pd.read_parquet(p(cfg, 'citations'))\n"
         "m = cites.merge(papers[['doc_id','year','title']], on='doc_id')\n"
         "m.sort_values('cited_by_count', ascending=False)[['year','cited_by_count','match_method','title']].head(10)"),
])

write("02_index.ipynb", [
    md("# 02 · Indexes: dictionary, postings, positions, zones, parameters"),
    code(SETUP), code(ENGINE),
    code("pd.read_csv(R / 'index/index_stats.csv')"),
    code("json.loads((R / 'index/build_info.json').read_text())"),
    md("## Dictionary entry and postings of one term, per zone\nPostings are `(doc_id, tf, [positions])`; the "
       "dictionary stores df and a pointer into the postings arrays."),
    code("term = index.analyzer.terms('retrieval')[0]\n"
         "for z in ('title', 'abstract', 'all'):\n"
         "    zi = index.zone(z); t = zi.term_id(term)\n"
         "    print(f'{z:8} term_id={t} df={zi.df[t]} pointer=[{zi.ptr[t]}, {zi.ptr[t+1]})  first postings: {zi.raw_postings(term, 3)}')"),
    md("## Boolean, phrase and proximity queries (postings touched are counted)"),
    code("from scholar_ir.boolean_search import boolean_search\nfrom scholar_ir.query_parser import parse\n"
         "for q in ['\"neural information retrieval\"', 'retrieval AND augmented AND generation',\n"
         "          '\"query expansion\" NEAR/5', 'title:survey AND recommender year:2015..2020', 'transformer NOT vision cat:cs.IR']:\n"
         "    pq = parse(q); docs, st = boolean_search(index, pq)\n"
         "    print(f'{q:55} -> {len(docs):5} docs, touched={st.touched:7}, lists={st.lists[:4]}')"),
    md("## Query optimisation: process AND terms in increasing df"),
    code("pd.read_csv(R / 'efficiency/boolean_and_summary.csv')"),
    md("## Effect of stemming and stop words on the vocabulary"),
    code("pd.read_csv(R / 'index/vocabulary_ablation.csv')"),
    md("## Variable-byte gap compression of the doc-id postings"),
    code("from scholar_ir.index import vb_encode, vb_decode\n"
         "d = index.zone('all').doc_list(term)[:8].tolist(); gaps = [d[0]] + [b - a for a, b in zip(d, d[1:])]\n"
         "enc = vb_encode(gaps); print('doc ids', d, '\\ngaps   ', gaps, '\\nbytes  ', list(enc), '\\ndecoded', vb_decode(enc))"),
])

write("03_ranking.ipynb", [
    md("# 03 · Ranking: tf-idf lnc.ltc, BM25, BM25F, authority g(d), adaptive λ"),
    code(SETUP), code(ENGINE),
    code("from scholar_ir.explain import format_explain\nq = 'dense passage retrieval for question answering'\n"
         "print(format_explain(engine.search(q, system='S1', k=3, explain=True)))"),
    code("print(format_explain(engine.search(q, system='S3', k=3, explain=True)))"),
    md("## Heap-based top-K equals a full sort"),
    code("from scholar_ir.ranking import heap_topk, query_terms, bm25f\nfrom scholar_ir.query_parser import parse\nimport time\n"
         "pq = parse(q)\n"
         "rc = engine.cfg['ranking']['bm25f']\nacc, touched = bm25f(index, query_terms(index, pq), rc['k1'], rc['w'], rc['b'])\n"
         "docs = np.flatnonzero(touched); t0 = time.perf_counter(); h = heap_topk(docs.tolist(), acc[docs].tolist(), 10); t1 = time.perf_counter()\n"
         "s = sorted(zip(docs.tolist(), acc[docs].tolist()), key=lambda x: (-x[1], x[0]))[:10]; t2 = time.perf_counter()\n"
         "print(f'J={len(docs)} scored docs; heap == sort: {h == s}; heap {1e3*(t1-t0):.1f} ms vs sort {1e3*(t2-t1):.1f} ms')"),
    md("## Static quality g(d): raw citations are age-biased, cohort percentiles are not"),
    code("a = engine.auth.merge(papers[['doc_id']], on='doc_id')\n"
         "by = a[a.has_citation_data].groupby('year')[['cited_by_count','g_raw','g_per_year','g_cohort','g_pagerank_cohort']].median()\n"
         "by.round(3)"),
    code("import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(7,3.2))\n"
         "for c, lab in [('g_raw','A0 raw'), ('g_per_year','A1 per-year'), ('g_cohort','A2 cohort percentile')]:\n"
         "    ax.plot(by.index, by[c], marker='o', ms=3, label=lab)\n"
         "ax.set_ylabel('median g(d)'); ax.set_xlabel('first-submission year'); ax.legend(); ax.set_title('Authority by publication year')\nplt.show()"),
    md("## Query-adaptive λ: features and λ for a few queries"),
    code("rows = []\nfor t in ['seminal work on word embeddings', 'survey of recommender systems', 'LLM agents that use tools',\n"
         "          'latest work on privacy attacks in federated learning', 'knowledge graph embedding']:\n"
         "    r = engine.search(t, system='S7', k=10); f = r.features\n"
         "    rows.append({'query': t, 'found_cue': f.found_cue, 'recent_cue': f.recent_cue, 'idf_z': round(f.idf_z,2),\n"
         "                 'len_z': round(f.len_z,2), 'flatness': round(f.flatness,2), 'p_found': round(r.p_found,3), 'lambda': round(r.lam,3)})\n"
         "pd.DataFrame(rows)"),
])

write("04_evaluation.ipynb", [
    md("# 04 · Evaluation\nAll tables are produced by `make eval` (`scholar_ir.experiments`). Tuning used the **dev** "
       "split only; **test** numbers are the headline. Recall is relative to the pooled judged set."),
    code(SETUP),
    code("q = pd.read_csv(p(cfg, 'eval_dir') / 'queries.csv'); print(q.groupby(['type','split']).size()); q.head()"),
    code("pd.read_csv(R / 'eval/judged_counts.csv').describe()"),
    code("pd.read_csv(R / 'eval/agreement.csv') if (R / 'eval/agreement.csv').exists() else 'single annotator per query'"),
    md("## Headline: test split"),
    code("pd.read_csv(R / 'eval/summary_test.csv')"),
    code("from IPython.display import Image\nImage(str(R / 'eval/ndcg_by_system_test.png'))"),
    code("Image(str(R / 'eval/win_loss_test.png'))"),
    code("s = pd.read_csv(R / 'eval/significance_test.csv'); s[s.metric == 'nDCG@10']"),
    md("## Age bias (no judgments needed)"),
    code("pd.read_csv(R / 'bias/age_bias_by_system.csv')"),
    code("Image(str(R / 'bias/age_bias_vs_lambda.png'))"),
    md("## Tuning on dev"),
    code("Image(str(R / 'tuning/lambda_sweep_dev.png'))"),
    code("Image(str(R / 'tuning/bm25_grid_dev.png'))"),
    code("json.loads((R / 'tuning/params.json').read_text())['report']"),
    md("## Ablations (test) and efficiency"),
    code("pd.read_csv(R / 'eval/ablations_test.csv')"),
    code("pd.read_csv(R / 'efficiency/latency.csv')"),
    code("pd.read_csv(R / 'efficiency/champion_lists.csv')"),
    md("## Failure analysis (test)"),
    code("pd.read_csv(R / 'eval/failure_analysis_test.csv')"),
    code("pd.read_csv(R / 'eval/summary_dev.csv')"),
])

write("05_llm_layer.ipynb", [
    md("# 05 · LLM layer on top of the engine\nThe LLM (a) translates plain English into our query language, "
       "(b) writes an answer from the top-5 retrieved abstracts, and (c) every answer sentence is checked by our "
       "tf-idf citation checker. Without an API key the parser falls back to rules and no answer is generated."),
    code(SETUP), code(ENGINE),
    code("from scholar_ir.llm_client import LLMClient\nfrom scholar_ir.llm_query_parser import llm_parse, rule_parse\n"
         "client = LLMClient(cfg); print(client.provider, client.model, 'available:', client.available())\n"
         "req = 'recent papers on retrieval-augmented generation for question answering since 2022'\n"
         "sq, src = llm_parse(req, client, int(engine.years.max())); print(src, sq.model_dump()); print('compiled:', sq.compile())\n"
         "print('rule parser:', rule_parse(req, int(engine.years.max())).compile())"),
    code("from scholar_ir.citation_checker import CitationChecker\nfrom scholar_ir.rag import answer\n"
         "ck = engine.cfg['checker']; checker = CitationChecker(index, ck['alpha'], ck['tau'])\n"
         "a = answer('How does dense passage retrieval differ from sparse retrieval such as BM25?', engine, client, checker)\nprint(a.report())"),
    md("## The checker on a deliberately weak sentence and a fabricated citation"),
    code("ctx = a.context; first = list(ctx)[0]\n"
         "for s in [f'Dense retrieval models were first proposed for protein folding in 1995 [arXiv:{first}].',\n"
         "          'Sparse retrieval is always better than dense retrieval [arXiv:2101.00001].']:\n"
         "    c = checker.check_sentence(s, ctx); print(f'{s}\\n   support={c.support} flags={c.flags}\\n')"),
    md("## Measured results (from `llm_experiments`)"),
    code("pd.read_csv(R / 'llm/parser_eval.csv') if (R / 'llm/parser_eval.csv').exists() else 'run: python -m scholar_ir.llm_experiments parser'"),
    code("c = R / 'llm/checker_eval.csv'\npd.read_csv(c) if c.exists() else 'run: python -m scholar_ir.llm_experiments checker (after labelling)'"),
    code("e = R / 'llm/checker_errors.csv'\npd.read_csv(e).head(10) if e.exists() else None"),
])

write("06_demo.ipynb", [
    md("# Scholar-IR — live demo (track T6: vertical search for science)\n"
       "**Problem.** Scholarly search should use the structure of papers (title/abstract zones, year, category) and "
       "their authority (citations). But raw citation counts grow with age, so mixing them into the ranking buries "
       "new, relevant papers. Scholar-IR (1) normalises authority within each publication-year cohort and (2) sets "
       "the authority weight λ per query, high for foundational requests and low for recent/specific ones."),
    code(SETUP), code(ENGINE),
    md("## 1 · Postings in the title and abstract zone indexes"),
    code("term = index.analyzer.terms('transformer')[0]\nfor z in ('title', 'abstract'):\n"
         "    zi = index.zone(z); print(f'{z}: df={zi.get_df(term)}  postings (doc, tf, positions):', zi.raw_postings(term, 4))"),
    md("## 2 · A Boolean + phrase + filter query, with postings touched"),
    code("from scholar_ir.boolean_search import boolean_search\nfrom scholar_ir.query_parser import parse\n"
         "pq = parse('\"language model\" AND (retrieval OR search) year:>=2022 cat:cs.IR')\n"
         "docs, st = boolean_search(index, pq)\nprint(pq.to_string(), '->', len(docs), 'docs; postings entries touched:', st.touched, '; lists:', st.lists)\n"
         "papers.set_index('doc_id').loc[docs[:5], ['arxiv_id', 'year', 'title']]"),
    md("## 3 · Same query, four rankers, with score breakdowns"),
    code("from scholar_ir.explain import format_explain\nQ = 'seminal work on word embeddings'\n"
         "for s in ('S1', 'S3', 'S4', 'S7'):\n    print('=' * 30, s); print(format_explain(engine.search(Q, system=s, k=3, explain=True), max_terms=4))"),
    md("## 4 · The bias demo: a recent query where raw authority (S4) buries new work and S7 recovers it"),
    code("def show(q):\n    for s in ('S3', 'S4', 'S7'):\n        r = engine.search(q, system=s, k=5)\n"
         "        lam = f' λ={r.lam:.2f}' if r.lam is not None else ''\n        print(f'--- {s}{lam}')\n"
         "        for x in r.results: print(f'   {x.year}  cites={x.citations:<6} {x.title[:80]}')\n"
         "show('retrieval-augmented generation for large language models')"),
    code("show('survey of recommender systems')"),
    code("pd.read_csv(R / 'bias/age_bias_by_system.csv')"),
    md("## 5 · Natural language → structured query → results → cited answer → citation check"),
    code("from scholar_ir.llm_client import LLMClient\nfrom scholar_ir.citation_checker import CitationChecker\nfrom scholar_ir.rag import answer\n"
         "client = LLMClient(cfg); ck = engine.cfg['checker']; checker = CitationChecker(index, ck['alpha'], ck['tau'])\n"
         "a = answer('What privacy attacks threaten federated learning? Recent work since 2021 please.', engine, client, checker)\nprint(a.report())"),
    code("ctx = a.context; pid = list(ctx)[0]\n"
         "for s in [f'Federated learning was invented to train image classifiers on satellites [arXiv:{pid}].',\n"
         "          'Gradient inversion recovers training images [arXiv:1999.00001].']:\n"
         "    c = checker.check_sentence(s, ctx); print(s, '->', c.flags, f'support={c.support}')"),
    md("## 6 · A limitation, live: missing citation data / paraphrase blind spot"),
    code("miss = engine.auth[~engine.auth.has_citation_data]\n"
         "print(f'{len(miss)} of {len(engine.auth)} papers ({100*len(miss)/len(engine.auth):.1f}%) have no OpenAlex match; they get the cohort median g = 0.5')\n"
         "para = f'A system that finds documents by comparing learned vector encodings of questions and texts [arXiv:{pid}].'\n"
         "c = checker.check_sentence(para, ctx); print('paraphrase ->', c.flags, f'support={c.support}')"),
    md("## 7 · Evaluation against the baseline (test split)"),
    code("pd.read_csv(R / 'eval/summary_test.csv')"),
    code("from IPython.display import Image\nImage(str(R / 'eval/ndcg_by_system_test.png'))"),
])

write("07_app.ipynb", [
    md("# Scholar-IR app\nRun the cell below (Shift+Enter). Type a question or pick an example, then press **Ask**. "
       "The *Search papers* tab takes the structured query language (phrases, `NEAR/5`, `title:`, `year:>=2021`, `cat:cs.IR`)."),
    code("import sys\nfrom pathlib import Path\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT / 'src'))\nfrom scholar_ir.gui import launch\napp = launch()"),
])
