"""Cited answer generation on top of the engine (retrieval stays the IR system's job).

1. retrieve the top-K papers with the full system (S7);
2. build the context block "[arXiv:ID] Title. Abstract." for each;
3. ask the LLM with the fixed system prompt (config rag.system_prompt) to answer ONLY from them,
   citing [arXiv:ID] after every sentence;
4. split the answer into sentences and verify each with the citation checker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .citation_checker import CitationChecker, SentenceCheck
from .engine import SearchEngine, SearchResponse
from .llm_client import LLMClient, LLMError
from .llm_query_parser import StructuredQuery, llm_parse


@dataclass
class Answer:
    question: str
    structured: StructuredQuery | None
    parser_source: str
    compiled_query: str
    retrieval: SearchResponse
    context: dict[str, str]
    text: str
    checks: list[SentenceCheck] = field(default_factory=list)
    llm_ok: bool = True

    def report(self) -> str:
        lines = [f"Q: {self.question}", f"parsed ({self.parser_source}): {self.compiled_query}",
                 "retrieved: " + ", ".join(f"[arXiv:{r.arxiv_id}]" for r in self.retrieval.results), "", "ANSWER:"]
        for c in self.checks:
            mark = "OK " if not c.flags else "!! "
            lines.append(f"{mark}{c.sentence}")
            lines.append(f"     support={c.support:.3f} (cos={c.cos:.3f}, cov={c.cov:.3f}, best={c.best_paper}) "
                         f"{' '.join(c.flags)}")
        return "\n".join(lines)


def build_context(engine: SearchEngine, resp: SearchResponse) -> dict[str, str]:
    abstracts = engine.papers.set_index("doc_id")["abstract"]
    return {r.arxiv_id: f"{r.title}. {abstracts[r.doc_id]}" for r in resp.results}


def context_block(ctx: dict[str, str]) -> str:
    return "\n\n".join(f"[arXiv:{k}] {v}" for k, v in ctx.items())


def answer(question: str, engine: SearchEngine, client: LLMClient, checker: CitationChecker,
           k: int | None = None, system: str = "S7", use_llm_parser: bool = True) -> Answer:
    cfg = engine.cfg
    k = k or cfg["rag"]["k"]
    year = int(engine.years.max())
    if use_llm_parser:
        sq, src = llm_parse(question, client, year)
        compiled = sq.compile() or question
    else:
        sq, src, compiled = None, "raw", question
    resp = engine.search(compiled, system=system, k=k, llm_intent=sq.intent if sq else None)
    if not resp.results and sq is not None:  # over-constrained translation: fall back to free text
        compiled = " ".join(sq.terms + sq.phrases) or question
        resp = engine.search(compiled, system=system, k=k)
    ctx = build_context(engine, resp)
    llm_ok = True
    if not client.available():
        llm_ok = False
        text = "No LLM API key is configured, so no answer was generated. The retrieved papers are listed below."
    else:
        user = f"Papers:\n\n{context_block(ctx)}\n\nQuestion: {question}"
        try:
            text = client.complete(cfg["rag"]["system_prompt"], user, max_tokens=900).strip()
        except LLMError as exc:  # retrieval still worked: show the papers, explain the missing answer
            llm_ok = False
            text = ("The language model is unavailable right now (" + str(exc)[:120] + "), so no answer was "
                    "generated. The retrieved papers are listed below.")
    return Answer(question=question, structured=sq, parser_source=src, compiled_query=compiled, retrieval=resp,
                  context=ctx, text=text, checks=checker.check_answer(text, ctx) if llm_ok else [], llm_ok=llm_ok)
