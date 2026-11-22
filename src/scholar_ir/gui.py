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
    "sequence to sequence learning for machine translation",
    "LLM-based recommender systems",
    '"dense retrieval" year:>=2021',
    "seminal work on word embeddings",
    "LLM agents that use tools",
    '"retrieval-augmented generation" year:>=2022',
    "title:survey recommender systems",
    "(dense OR neural) AND retrieval NOT image cat:cs.IR",
    '"query expansion" NEAR/5 feedback NOT image',
]

CSS = """
<style>
.sir{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#1e2330;background:#f5f6fa;
     border-radius:14px;padding:14px 16px;max-width:1060px;line-height:1.45}
.sir h3{font-size:12px;margin:16px 0 8px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.sir .card{border:1px solid #e3e6ee;border-radius:12px;padding:12px 16px;margin:8px 0;background:#fff;
           box-shadow:0 1px 2px rgba(16,24,40,.05)}
.sir .muted{color:#6b7280;font-size:12.5px}
.sir code{background:#eef0f6;color:#1e2330;padding:1px 6px;border-radius:5px;font-size:12px}
.sir .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:600;margin-right:4px}
.sir .ok{background:#dcfce7;color:#166534}.sir .bad{background:#fee2e2;color:#991b1b}.sir .info{background:#e0e7ff;color:#3730a3}
.sir .warn{background:#fef3c7;color:#92400e}.sir .grey{background:#eef0f6;color:#374151}
.sir .sent{border-left:4px solid #16a34a;padding:8px 12px;margin:8px 0;background:#f0fdf4;border-radius:0 8px 8px 0}
.sir .sent.flag{border-left-color:#dc2626;background:#fef2f2}
.sir .bar{height:6px;background:#e5e7eb;border-radius:3px;width:160px;display:inline-block;vertical-align:middle;margin:0 6px;position:relative}
.sir .bar>span{display:block;height:6px;border-radius:3px;background:#16a34a}
.sir .bar>i{position:absolute;top:-3px;width:2px;height:12px;background:#6b7280}
.sir .sent.flag .bar>span{background:#dc2626}
.sir a{color:#4338ca;text-decoration:none}.sir a:hover{text-decoration:underline}
.sir table{border-collapse:collapse;font-size:12px;margin-top:6px}
.sir td,.sir th{border-bottom:1px solid #eef0f6;padding:3px 8px;text-align:left}
.sir details summary{cursor:pointer;color:#6b7280;font-size:12px;margin-top:4px}
/* search summary */
.sir .sys{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.sir .sysname{font-weight:700;font-size:15px}
.sir .tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:10px}
.sir .tile{background:#f8f9fc;border:1px solid #eceff5;border-radius:10px;padding:9px 11px}
.sir .tile .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:#6b7280;font-weight:700}
.sir .tile .v{font-size:18px;font-weight:700;margin-top:2px}
.sir .tile .s{font-size:11.5px;color:#6b7280}
.sir .meter{height:6px;background:#e5e7eb;border-radius:3px;margin-top:6px;overflow:hidden}
.sir .meter>span{display:block;height:6px;background:#4f46e5}
.sir .note{margin-top:10px;font-size:12px;color:#92400e;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:6px 10px}
.sir .hubs{margin-top:10px;font-size:12px}
.sir .hub{display:inline-block;background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;border-radius:8px;padding:3px 8px;margin:3px 4px 0 0}
/* result cards */
.sir .paper{display:flex;gap:14px;align-items:flex-start}
.sir .rank{flex:0 0 30px;height:30px;border-radius:50%;background:#eef2ff;color:#4338ca;font-weight:700;
           display:flex;align-items:center;justify-content:center;font-size:13px}
.sir .ptitle{font-weight:650;font-size:14.5px}
.sir .meta{margin-top:3px}
.sir .scores{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:11.5px;color:#4b5563}
.sir .sc{display:flex;align-items:center;gap:6px}
.sir .sc .mb{width:74px;height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden}
.sir .sc .mb>span{display:block;height:6px}
.sir .final{margin-left:auto;font-weight:700;color:#1e2330;font-size:13px}
/* compare */
.sir .cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.sir .crow{display:flex;gap:8px;align-items:flex-start;padding:6px 0;border-bottom:1px solid #f0f1f5;font-size:12.5px}
.sir .crow:last-child{border-bottom:none}
.sir .crank{flex:0 0 20px;color:#9ca3af;font-weight:700}
.sir .mv{flex:0 0 52px;text-align:right;font-size:11px;font-weight:700}
.sir .up{color:#15803d}.sir .down{color:#b91c1c}.sir .new{color:#b45309}.sir .gone{color:#9ca3af}
@media (max-width:760px){.sir .tiles{grid-template-columns:repeat(2,minmax(0,1fr))}.sir .cmp{grid-template-columns:1fr}}
</style>
"""

HEADER = """
<div style="font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1060px;border-radius:14px;
  padding:16px 20px;margin:2px 0 10px;background:linear-gradient(120deg,#312e81 0%,#4f46e5 55%,#7c3aed 100%);color:#fff">
  <div style="font-size:22px;font-weight:800;letter-spacing:-.01em">Scholar-IR</div>
  <div style="opacity:.9;font-size:13px;margin-top:2px">Search 80,000 arXiv computer-science papers · BM25F ranking ·
    topic-local citation authority (HITS) · query-adaptive weighting · cited answers</div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;font-size:11.5px">
    {chips}
  </div>
</div>
"""

def _arxiv_link(aid: str) -> str:
    return f'<a href="https://arxiv.org/abs/{html.escape(aid)}" target="_blank">arXiv:{html.escape(aid)}</a>'


def _linkify_citations(text: str) -> str:
    esc = html.escape(text)
    return re.sub(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5})", lambda m: _arxiv_link(m.group(1)), esc)


def _intent_html(resp: SearchResponse) -> str:
    """One-line intent summary (used under answers)."""
    if resp.features is None:
        return ""
    f = resp.features
    lam = f"<b>λ = {resp.lam:.2f}</b> (authority weight) · " if resp.lam is not None else ""
    cues = ", ".join(f.matched_cues) or "none"
    return (f'<div class="muted">{lam}p(foundational) = {resp.p_found:.2f} · intent cues: {html.escape(cues)} · '
            f"mean-idf z = {f.idf_z:+.2f} · score flatness = {f.flatness:.2f}</div>")


def _mini(label: str, value: float | None, color: str) -> str:
    if value is None or value != value:  # None or NaN
        return ""
    w = max(2, min(100, int(round(100 * float(value)))))
    return (f'<div class="sc">{html.escape(label)}<div class="mb"><span style="width:{w}%;background:{color}"></span>'
            f"</div><b>{float(value):.2f}</b></div>")


def _category(engine: SearchEngine, doc_id: int) -> str:
    try:
        cats = engine.papers.iloc[doc_id]["categories"]
        cats = list(cats) if not isinstance(cats, str) else cats.split()
        return cats[0] if cats else ""
    except Exception:
        return ""


def papers_html(engine: SearchEngine, resp: SearchResponse, explain: bool = False) -> str:
    abstracts = engine.papers.set_index("doc_id")["abstract"]
    out = []
    for r in resp.results:
        scores = ""
        if r.relevance is not None:
            bars = [_mini("relevance R", r.relevance, "#4f46e5")]
            if resp.hits_info:
                bars.append(_mini("topic HITS", resp.hits_info["hits_authority"].get(r.doc_id, 0.0), "#d97706"))
                if "A2" in engine.g:
                    bars.append(_mini("global", float(engine.g["A2"][r.doc_id]), "#0d9488"))
            else:
                bars.append(_mini("authority g", r.authority, "#0d9488"))
            scores = f'<div class="scores">{"".join(bars)}<div class="final">score {r.score:.3f}</div></div>'
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
            breakdown = (f"<details><summary>why this paper? ({html.escape(rel.get('model', ''))})</summary>"
                         f"<table>{head}{''.join(rows)}</table>{net_line}</details>")
        abstract = html.escape(str(abstracts[r.doc_id]))
        cat = _category(engine, r.doc_id)
        cat_pill = f'<span class="pill grey">{html.escape(cat)}</span>' if cat else ""
        url = f"https://arxiv.org/abs/{html.escape(r.arxiv_id)}"
        out.append(
            f'<div class="card paper"><div class="rank">{r.rank}</div><div style="flex:1;min-width:0">'
            f'<div class="ptitle"><a href="{url}" target="_blank" style="color:#1e2330">{html.escape(r.title)}</a></div>'
            f'<div class="meta muted"><span class="pill info">{r.year}</span><span class="pill warn">{r.citations:,} citations</span>'
            f"{cat_pill} {_arxiv_link(r.arxiv_id)}</div>{scores}"
            f'<details><summary>abstract</summary><div class="muted" style="margin-top:4px">{abstract}</div></details>'
            f"{breakdown}</div></div>")
    return "".join(out) or '<div class="card muted">No papers matched.</div>'


def _summary_html(engine: SearchEngine, resp: SearchResponse) -> str:
    """System name + four tiles (candidates, λ, predicted field, topic graph) + hubs."""
    t_ms = resp.timings_ms.get("total", 0)
    tiles = [f'<div class="tile"><div class="k">Candidates</div><div class="v">{resp.n_candidates:,}</div>'
             f'<div class="s">{t_ms:.0f} ms</div></div>']
    if resp.lam is not None:
        cues = ", ".join(resp.features.matched_cues) if resp.features and resp.features.matched_cues else "no cue words"
        tiles.append(f'<div class="tile"><div class="k">Authority weight λ</div><div class="v">{resp.lam:.2f}</div>'
                     f'<div class="meter"><span style="width:{int(100 * resp.lam)}%"></span></div>'
                     f'<div class="s">p(foundational) {resp.p_found:.2f} · {html.escape(cues)}</div></div>')
    try:
        pred = engine.topic_classifier.predict(resp.query, top=2)
    except Exception:
        pred = []
    if pred:
        tiles.append(f'<div class="tile"><div class="k">Predicted field · Naive Bayes</div><div class="v">{pred[0][0]}</div>'
                     f'<div class="s">{pred[0][1]:.0%} · then {pred[1][0]} {pred[1][1]:.0%} · add <code>cat:{pred[0][0]}</code> to filter</div></div>'
                     if len(pred) > 1 else
                     f'<div class="tile"><div class="k">Predicted field</div><div class="v">{pred[0][0]}</div></div>')
    hi = resp.hits_info
    extra = ""
    if hi:
        tiles.append(f'<div class="tile"><div class="k">Topic citation graph · HITS</div>'
                     f'<div class="v">{hi["edges"]:,} <span style="font-size:12px;font-weight:600">links</span></div>'
                     f'<div class="s">{hi["base_size"]:,} papers · {hi["iterations"]} iterations · '
                     f'{hi["cited_in_topic"]} candidates cited in-topic</div></div>')
        if hi["edges"] == 0:
            extra += ('<div class="note">No citation links among these papers (most 2021+ papers have no reference '
                      "list in our OpenAlex data), so this ranking falls back on text relevance + global authority.</div>")
        hubs = [(d, v) for d, v in sorted(hi["hub"].items(), key=lambda kv: -kv[1]) if v > 0][:3]
        if hubs:
            extra += ('<div class="hubs"><span class="muted"><b>Key surveys / hubs for this topic:</b></span><br>'
                      + "".join(f'<span class="hub">{html.escape(engine.titles[d][:72])} ({engine.years[d]})</span>' for d, _ in hubs)
                      + "</div>")
    return (f'<div class="card"><div class="sys"><span class="pill info">{html.escape(resp.system)}</span>'
            f'<span class="sysname">{html.escape(SYSTEMS[resp.system]["label"])}</span>'
            f'<span class="muted">parsed: <code>{html.escape(resp.parsed)}</code></span></div>'
            f'<div class="tiles">{"".join(tiles)}</div>{extra}</div>')


def _topic_html(engine: SearchEngine, resp: SearchResponse) -> str:  # kept for compatibility
    return _summary_html(engine, resp)


def search_html(engine: SearchEngine, resp: SearchResponse, explain: bool = False) -> str:
    return f'<div class="sir">{CSS}{_summary_html(engine, resp)}<h3>Results</h3>{papers_html(engine, resp, explain)}</div>'


def compare_html(engine: SearchEngine, old: SearchResponse, new: SearchResponse) -> str:
    """Side-by-side: the old ranking (left) vs the selected one (right), with rank movements."""
    old_rank = {d: i for i, d in enumerate(old.doc_ids, 1)}
    new_rank = {d: i for i, d in enumerate(new.doc_ids, 1)}

    def col(resp, other, is_new):
        rows = []
        for r in resp.results:
            o = other.get(r.doc_id)
            if is_new:
                mv = ('<span class="mv new">NEW</span>' if o is None else
                      f'<span class="mv up">▲ {o - r.rank}</span>' if o > r.rank else
                      f'<span class="mv down">▼ {r.rank - o}</span>' if o < r.rank else '<span class="mv gone">=</span>')
            else:
                mv = '<span class="mv gone">dropped</span>' if o is None else '<span class="mv"></span>'
            rows.append(f'<div class="crow"><span class="crank">{r.rank}</span><div style="flex:1;min-width:0">'
                        f'<a href="https://arxiv.org/abs/{html.escape(r.arxiv_id)}" target="_blank" style="color:#1e2330">'
                        f"{html.escape(r.title[:90])}</a><div class='muted'>{r.year} · {r.citations:,} citations</div></div>{mv}</div>")
        lam = f" · λ {resp.lam:.2f}" if resp.lam is not None else ""
        return (f'<div class="card"><div class="sys"><span class="pill {"warn" if is_new else "grey"}">{resp.system}</span>'
                f'<span class="sysname" style="font-size:13.5px">{html.escape(SYSTEMS[resp.system]["label"])}</span>'
                f'<span class="muted">{lam}</span></div>{"".join(rows)}</div>')
    moved_in = sum(1 for d in new.doc_ids if d not in old_rank)
    note = (f'<div class="muted" style="margin:2px 0 6px">{moved_in} of the top {len(new.doc_ids)} are new compared '
            f"with {old.system}; arrows show how far each paper moved.</div>")
    return (f'<div class="sir">{CSS}<h3>Old vs new ranking</h3>{note}<div class="cmp">{col(old, new_rank, False)}'
            f"{col(new, old_rank, True)}</div></div>")


def checker_tau(engine: SearchEngine) -> float:
    return float(engine.cfg["checker"]["tau"])


def answer_html(engine: SearchEngine, ans) -> str:
    sq = ans.structured
    parsed = (f'<div class="muted">LLM translation ({html.escape(ans.parser_source)}): '
              f"<code>{html.escape(sq.model_dump_json(exclude_defaults=True) if sq else '-')}</code></div>")
    head = (f'<div class="card"><div><b>Q:</b> {html.escape(ans.question)}</div>{parsed}'
            f'<div class="muted">compiled query: <code>{html.escape(ans.compiled_query)}</code> · ranked by {html.escape(ans.retrieval.system)}</div>'
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
    lay = w.Layout(width="620px")

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
    s_box = w.Text(placeholder='Search papers…  e.g.  "dense retrieval" year:>=2021 cat:cs.IR', layout=lay)
    s_examples = w.Dropdown(options=["— example searches —"] + EXAMPLE_SEARCHES, layout=lay)
    s_sys = w.Dropdown(options=[(f"{k} · {v['label']}", k) for k, v in SYSTEMS.items() if k != "RAND"], value="S10",
                       layout=w.Layout(width="340px"))
    s_k = w.IntSlider(value=10, min=5, max=30, description="results", layout=w.Layout(width="260px"))
    s_exp = w.Checkbox(value=True, description="explain scores", indent=False, layout=w.Layout(width="130px"))
    s_cmp = w.Checkbox(value=False, description="compare with S4 (old ranking)", indent=False, layout=w.Layout(width="230px"))
    s_btn = w.Button(description="Search", button_style="primary", icon="search", layout=w.Layout(width="140px"))
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
                q = s_box.value.strip()
                resp = engine.search(q, system=s_sys.value, k=s_k.value, explain=s_exp.value)
                if s_cmp.value and s_sys.value != "S4":
                    display(HTML(compare_html(engine, engine.search(q, system="S4", k=s_k.value), resp)))
                display(HTML(search_html(engine, resp, s_exp.value)))
            except Exception as exc:
                display(HTML(f"<div class='sir card' style='color:#a40e26'>Error: {html.escape(str(exc))[:500]}</div>"))
    s_btn.on_click(run_s)
    search_tab = w.VBox([s_examples, w.HBox([s_box, s_btn]), w.HBox([s_sys, s_k, s_exp, s_cmp]), s_out])

    tabs = w.Tab(children=[ask_tab, search_tab])
    tabs.set_title(0, "Ask a question")
    tabs.set_title(1, "Search papers")
    try:
        n_links = f"{engine.graph.n_edges:,} citation links"
    except Exception:
        n_links = "citation graph not built"
    chip = "<span style='background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:3px 10px'>{}</span>"
    chips = "".join(chip.format(c) for c in (f"{engine.index.N:,} papers", "2010–2023", n_links, "7 CS fields",
                                             "default: S10"))
    title = w.HTML(HEADER.replace("{chips}", chips))
    app = w.VBox([title, tabs])
    display(app)
    return app
