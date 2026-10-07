"""Point-and-click interface for Scholar-IR inside Jupyter (ipywidgets; no web server).

    from scholar_ir.gui import launch
    launch()

Two tabs:
  * Ask   — a question in plain English -> LLM-parsed query -> top-K papers -> cited answer, with
            every sentence checked by the citation checker (green = supported, red = flagged).
  * Search — the structured query language (phrases, NEAR, title:, year:, cat:) with any ranker
            and an optional per-result score breakdown.
The HTML builders (`answer_html`, `search_html`) are plain functions, so they are testable.
"""
from __future__ import annotations

import html
import re

from .citation_checker import CitationChecker
from .engine import SYSTEMS, SearchEngine, SearchResponse
from .llm_client import LLMClient

EXAMPLE_QUESTIONS = [
    "How does dense passage retrieval differ from sparse retrieval such as BM25?",
    "What privacy attacks threaten federated learning?",
    "How do graph neural networks aggregate information from neighbouring nodes?",
    "How is knowledge distillation used to compress neural networks?",
    "What is chain-of-thought prompting and why does it improve reasoning?",
    "How does pseudo-relevance feedback expand queries in information retrieval?",
    "Recent work since 2022 on hallucination detection in large language models?",
]
EXAMPLE_SEARCHES = [
    "seminal work on word embeddings",
    "LLM agents that use tools",
    '"retrieval-augmented generation" year:>=2022',
    "title:survey recommender systems",
    "(dense OR neural) AND retrieval NOT image cat:cs.IR",
    '"query expansion" NEAR/5',
]

CSS = """
<style>
.sir{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#1f2328;max-width:980px}
.sir h3{font-size:15px;margin:18px 0 8px;color:#57606a;text-transform:uppercase;letter-spacing:.04em}
.sir .card{border:1px solid #d0d7de;border-radius:10px;padding:12px 14px;margin:8px 0;background:#fff}
.sir .muted{color:#57606a;font-size:12.5px}
.sir code{background:#f3f4f6;padding:2px 6px;border-radius:5px;font-size:12.5px}
.sir .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;margin-right:4px}
.sir .ok{background:#dafbe1;color:#116329}.sir .bad{background:#ffebe9;color:#a40e26}.sir .info{background:#ddf4ff;color:#0a3069}
.sir .sent{border-left:4px solid #2da44e;padding:8px 12px;margin:8px 0;background:#f6fef8;border-radius:0 8px 8px 0}
.sir .sent.flag{border-left-color:#cf222e;background:#fff6f6}
.sir .bar{height:6px;background:#eaeef2;border-radius:3px;width:180px;display:inline-block;vertical-align:middle;margin:0 6px}
.sir .bar{position:relative}
.sir .bar>span{display:block;height:6px;border-radius:3px;background:#2da44e}
.sir .bar>i{position:absolute;top:-3px;width:2px;height:12px;background:#57606a}
.sir .sent.flag .bar>span{background:#cf222e}
.sir .paper{display:flex;gap:12px;align-items:flex-start}
.sir .rank{font-weight:700;color:#8c959f;min-width:22px}
.sir a{color:#0969da;text-decoration:none}.sir a:hover{text-decoration:underline}
.sir table{border-collapse:collapse;font-size:12.5px;margin-top:6px}
.sir td,.sir th{border-bottom:1px solid #eaeef2;padding:3px 8px;text-align:left}
.sir details summary{cursor:pointer;color:#57606a;font-size:12.5px}
</style>
"""


def _arxiv_link(aid: str) -> str:
    return f'<a href="https://arxiv.org/abs/{html.escape(aid)}" target="_blank">arXiv:{html.escape(aid)}</a>'


def _linkify_citations(text: str) -> str:
    esc = html.escape(text)
    return re.sub(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5})", lambda m: _arxiv_link(m.group(1)), esc)


def _intent_html(resp: SearchResponse) -> str:
    if resp.features is None:
        return ""
    f = resp.features
    lam = f"<b>λ = {resp.lam:.2f}</b> (authority weight) · " if resp.lam is not None else ""
    cues = ", ".join(f.matched_cues) or "none"
    return (f'<div class="muted">{lam}p(foundational) = {resp.p_found:.2f} · intent cues: {html.escape(cues)} · '
            f"mean-idf z = {f.idf_z:+.2f} · score flatness = {f.flatness:.2f}</div>")


def papers_html(engine: SearchEngine, resp: SearchResponse, explain: bool = False) -> str:
    abstracts = engine.papers.set_index("doc_id")["abstract"]
    out = []
    for r in resp.results:
        extra = ""
        if r.relevance is not None:
            extra = f" · relevance R = {r.relevance:.2f} · authority g = {r.authority:.2f}"
        breakdown = ""
        if explain and r.explain:
            rel = r.explain.get("relevance", {})
            rows = []
            for t in rel.get("terms", []):
                if rel.get("model") == "BM25F":
                    z = " ".join(f"{k}: tf={v['tf']}" for k, v in t["zones"].items())
                    rows.append(f"<tr><td>{html.escape(t['term'])}</td><td>{t['idf']}</td><td>{z}</td><td>{t['contribution']}</td></tr>")
                elif rel.get("model") == "lnc.ltc":
                    rows.append(f"<tr><td>{html.escape(t['term'])}</td><td>{t['w_tq(ltc)']}</td><td>tf={t['tf']} w_td={t['w_td(lnc)']}</td><td>{t['product']}</td></tr>")
                else:
                    rows.append(f"<tr><td>{html.escape(t['term'])}</td><td>{t['idf']}</td><td>tf={t['tf']}</td><td>{t['contribution']}</td></tr>")
            head = "<tr><th>term</th><th>idf / w_q</th><th>matches</th><th>contribution</th></tr>"
            net = r.explain.get("net", {})
            net_line = f"<div class='muted'>net score = {html.escape(net['formula'])} = {net['net']}</div>" if net else ""
            breakdown = f"<details><summary>why this paper? ({html.escape(rel.get('model', ''))})</summary><table>{head}{''.join(rows)}</table>{net_line}</details>"
        abstract = html.escape(str(abstracts[r.doc_id]))
        out.append(
            f'<div class="card paper"><div class="rank">{r.rank}</div><div>'
            f'<div><b>{html.escape(r.title)}</b></div>'
            f'<div class="muted">{_arxiv_link(r.arxiv_id)} · {r.year} · {r.citations:,} citations · score {r.score:.3f}{extra}</div>'
            f'<details><summary>abstract</summary><div class="muted" style="margin-top:4px">{abstract}</div></details>'
            f"{breakdown}</div></div>")
    return "".join(out) or '<div class="card muted">No papers matched.</div>'


def search_html(engine: SearchEngine, resp: SearchResponse, explain: bool = False) -> str:
    head = (f'<div class="card"><div><span class="pill info">{html.escape(resp.system)}</span>'
            f"{html.escape(SYSTEMS[resp.system]['label'])}</div>"
            f'<div class="muted" style="margin-top:6px">parsed query: <code>{html.escape(resp.parsed)}</code> · '
            f"{resp.n_candidates:,} candidate papers · {resp.timings_ms.get('total', 0):.0f} ms</div>{_intent_html(resp)}</div>")
    return f'<div class="sir">{CSS}{head}<h3>Results</h3>{papers_html(engine, resp, explain)}</div>'


def checker_tau(engine: SearchEngine) -> float:
    return float(engine.cfg["checker"]["tau"])


def answer_html(engine: SearchEngine, ans) -> str:
    sq = ans.structured
    parsed = (f'<div class="muted">LLM translation ({html.escape(ans.parser_source)}): '
              f"<code>{html.escape(sq.model_dump_json(exclude_defaults=True) if sq else '-')}</code></div>")
    head = (f'<div class="card"><div><b>Q:</b> {html.escape(ans.question)}</div>{parsed}'
            f'<div class="muted">compiled query: <code>{html.escape(ans.compiled_query)}</code> · ranked by S7</div>'
            f"{_intent_html(ans.retrieval)}</div>")
    if not getattr(ans, "llm_ok", True):
        note = (f'<div class="card" style="border-color:#d4a72c;background:#fff8c5">{html.escape(ans.text)} '
                f"Try again in a minute — cached answers come back instantly.</div>")
        return (f'<div class="sir">{CSS}{head}<h3>Answer</h3>{note}'
                f"<h3>Retrieved papers</h3>{papers_html(engine, ans.retrieval)}</div>")
    n_flag = sum(1 for c in ans.checks if c.flags)
    summary = (f'<span class="pill ok">{len(ans.checks) - n_flag} supported</span>'
               + (f'<span class="pill bad">{n_flag} flagged</span>' if n_flag else ""))
    sents = []
    for c in ans.checks:
        flags = "".join(f'<span class="pill bad">{html.escape(f.split(":")[0])}</span>' for f in c.flags) \
            or '<span class="pill ok">SUPPORTED</span>'
        width = max(2, min(100, int(round(100 * c.support))))
        best = f" · best match {_arxiv_link(c.best_paper)}" if c.best_paper else ""
        sents.append(
            f'<div class="sent{" flag" if c.flags else ""}"><div>{_linkify_citations(c.sentence)}</div>'
            f'<div class="muted" style="margin-top:4px">{flags} support {c.support:.2f}'
            f'<span class="bar" title="tick = threshold τ"><span style="width:{width}%"></span>'
            f'<i style="left:{int(100 * checker_tau(engine))}%"></i></span>'
            f"cosine {c.cos:.2f} · term coverage {c.cov:.2f}{best}</div></div>")
    return (f'<div class="sir">{CSS}{head}<h3>Answer {summary}</h3>{"".join(sents)}'
            f"<h3>Papers the answer may cite</h3>{papers_html(engine, ans.retrieval)}</div>")


def launch(engine: SearchEngine | None = None):
    """Build and display the two-tab app. Returns the widget."""
    import ipywidgets as w
    from IPython.display import HTML, display

    from .rag import answer

    import logging
    for name in ("llm", "llm_parser"):  # progress goes to the status line, not raw log lines
        logging.getLogger(name).setLevel(logging.ERROR)
    engine = engine or SearchEngine.load()
    client = LLMClient(engine.cfg)
    ck = engine.cfg["checker"]
    checker = CitationChecker(engine.index, ck["alpha"], ck["tau"])
    lay = w.Layout(width="760px")

    # --- Ask tab ---------------------------------------------------------------------------------
    q_box = w.Textarea(placeholder="Ask a question about CS research (papers 2010-2023)…", layout=w.Layout(width="760px", height="64px"))
    q_examples = w.Dropdown(options=["— example questions —"] + EXAMPLE_QUESTIONS, layout=lay)
    q_k = w.IntSlider(value=engine.cfg["rag"]["k"], min=3, max=8, description="papers")
    q_btn = w.Button(description="Ask", button_style="primary", icon="search")
    q_out = w.Output()
    q_status = w.HTML()
    client.on_status = lambda msg: setattr(q_status, "value",
                                           f"<span style='color:#57606a;font-size:12px'>⏳ {html.escape(msg)}…</span>")
    llm_note = w.HTML(f"<span style='color:#57606a;font-size:12px'>LLM: {client.provider} "
                      f"({', '.join(client.models) if client.models else 'none'}) · "
                      f"{'ready' if client.available() else 'NO API KEY — answers disabled, retrieval still works'}</span>")

    def pick_q(change):
        if change["new"] in EXAMPLE_QUESTIONS:
            q_box.value = change["new"]
    q_examples.observe(pick_q, names="value")

    def run_q(_):
        if not q_box.value.strip():
            return
        q_btn.disabled = True
        with q_out:
            q_out.clear_output()
            display(HTML("<div class='sir' style='color:#57606a'>Searching 80,000 papers and asking the LLM… "
                         "(if Gemini is busy this can take a minute)</div>"))
            try:
                ans = answer(q_box.value.strip(), engine, client, checker, k=q_k.value, use_llm_parser=q_parse.value)
                q_out.clear_output()
                display(HTML(answer_html(engine, ans)))
                q_status.value = (f"<span style='color:#57606a;font-size:12px'>answered by {html.escape(client.model or '-')}"
                                  f" · cached answers return instantly</span>")
            except Exception as exc:  # show the error in the app instead of a traceback
                q_out.clear_output()
                display(HTML(f"<div class='sir card' style='color:#a40e26'>Error: {html.escape(str(exc))[:500]}</div>"))
            finally:
                q_btn.disabled = False
    q_btn.on_click(run_q)
    q_parse = w.Checkbox(value=True, description="LLM parses the question", indent=False)
    ask_tab = w.VBox([q_examples, q_box, w.HBox([q_btn, q_k, q_parse]), llm_note, q_status, q_out])

    # --- Search tab ------------------------------------------------------------------------------
    s_box = w.Text(placeholder='e.g.  "dense retrieval" year:>=2021 cat:cs.IR', layout=lay)
    s_examples = w.Dropdown(options=["— example searches —"] + EXAMPLE_SEARCHES, layout=lay)
    s_sys = w.Dropdown(options=[(f"{k} · {v['label']}", k) for k, v in SYSTEMS.items() if k != "RAND"], value="S7",
                       layout=w.Layout(width="360px"))
    s_k = w.IntSlider(value=10, min=5, max=30, description="results")
    s_exp = w.Checkbox(value=True, description="explain scores")
    s_btn = w.Button(description="Search", button_style="primary", icon="search")
    s_out = w.Output()

    def pick_s(change):
        if change["new"] in EXAMPLE_SEARCHES:
            s_box.value = change["new"]
    s_examples.observe(pick_s, names="value")

    def run_s(_):
        if not s_box.value.strip():
            return
        with s_out:
            s_out.clear_output()
            try:
                resp = engine.search(s_box.value.strip(), system=s_sys.value, k=s_k.value, explain=s_exp.value)
                display(HTML(search_html(engine, resp, s_exp.value)))
            except Exception as exc:
                display(HTML(f"<div class='sir card' style='color:#a40e26'>Error: {html.escape(str(exc))[:500]}</div>"))
    s_btn.on_click(run_s)
    search_tab = w.VBox([s_examples, s_box, w.HBox([s_sys, s_k, s_exp]), s_btn, s_out])

    tabs = w.Tab(children=[ask_tab, search_tab])
    tabs.set_title(0, "Ask a question")
    tabs.set_title(1, "Search papers")
    title = w.HTML("<div style='font:600 20px -apple-system,sans-serif;margin:4px 0 2px'>Scholar-IR</div>"
                   f"<div style='color:#57606a;font:13px -apple-system,sans-serif;margin-bottom:8px'>"
                   f"{engine.index.N:,} arXiv CS papers · zone-aware BM25F · age-fair, query-adaptive citation authority</div>")
    app = w.VBox([title, tabs])
    display(app)
    return app
