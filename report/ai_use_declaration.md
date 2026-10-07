# AI-use declaration (draft — the team must review and complete it)

| What | Tool / model | How it was used |
|---|---|---|
| Code | Claude (Anthropic), used as an AI coding agent | Wrote most of the Python code from the team's master prompt (data loader, OpenAlex fetcher, indexes, Boolean/phrase search, rankers, authority, adaptive λ, evaluation harness, LLM layer, CLI, notebooks, tests). The team directed the design, ran the data pipeline, reviewed the code and outputs. |
| Debugging the citation source | Claude + team | The team ran probe scripts against OpenAlex; Claude adapted the matcher (three passes, title checks, DOI trust) to the failure cases found. |
| Query set and gold data | Claude drafted; team reviewed/edited | `eval/queries.csv` (40 queries + relevance descriptions), `eval/nl_requests.csv` (25 NL requests with gold parses), `eval/rag_questions.csv` (20 questions). |
| Relevance judgments | **Team members only** | `eval/judgments/*.csv` were graded by hand, blind to system. No AI-generated relevance labels. |
| Answer-sentence labels | **Team members only** (plus programmatically injected corruptions, labelled automatically) | `eval/answer_judgments.csv`. |
| LLM inside the system | Google Gemini via REST API, temperature 0, with a fallback chain of models because of server overload (`gemini-3.5-flash-lite` first, then other Gemini flash/flash-lite models; the model that produced each cached answer is logged in `data/cache/llm/usage.jsonl` — report the actual mix) | Translates plain-English requests into our query language; optional intent label for the adaptive λ; writes cited answers from retrieved abstracts. It never ranks and never sees the corpus beyond the top-5 retrieved abstracts. |
| Report and video script | Claude drafted; team edited | `report/report_draft.md`, `report/video_script.md`. Every number in them is read from `results/`. |

Libraries: numpy, pandas, pyarrow, scipy, matplotlib, PyYAML, NLTK, requests, pydantic, datasets,
pytest, rank_bm25 (cross-check only), Jupyter. Datasets: arXiv metadata (Cornell/Kaggle, CC0),
OpenAlex (CC0).
