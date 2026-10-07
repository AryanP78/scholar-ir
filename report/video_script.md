# Video script (target 6 min; 5–8 allowed; no slides — everything is live in `notebooks/06_demo.ipynb` and the terminal)

Before recording: `make index` done, `results/` regenerated (`make eval`), `.env` has the Gemini key,
notebook kernel restarted. Run each cell live on camera; do not paste outputs.

| Time | Who | Screen | Say (keep it short) |
|---|---|---|---|
| 0:00–0:50 | Member A | Demo notebook, title cell | **Problem + track.** T6 asks for vertical search that uses document structure. Scientific papers have zones (title, abstract), metadata (year, category) and authority (citations). The catch: citation counts grow with age, so adding them to the score pushes old papers above new relevant ones. Our idea: compare a paper only with papers from the same year, and let each query decide how much authority matters. |
| 0:50–1:40 | Member B | Cell 1–2 | **Index.** Show the title-zone and abstract-zone postings for "transformer": doc id, tf, positions. Then the Boolean + phrase + filter query, and the number of postings touched; mention df-ordered intersection. |
| 1:40–2:50 | Member C | Cell 3 | **Ranking, intermediate output.** Same query through S1 (lnc.ltc: point at w_tq, w_td, the products), S3 (BM25F per-zone tf, tf', idf), S4 (raw citations) and S7 (cohort percentile g, λ and its features, the net-score formula). |
| 2:50–3:40 | Member A | Cell 4 | **The bias demo.** Recent query: S4 fills the top with older, highly cited, loosely related papers; S7 keeps the recent relevant ones (read the years and citation counts aloud). Foundational query: S7 raises λ and keeps the classics. Show the age-bias table. |
| 3:40–4:40 | Member D | Cell 5–6 | **LLM layer.** NL request → parsed JSON → compiled query (shown) → top-5 → cited answer → checker scores per sentence. Then the two planted sentences: one unsupported claim, one fabricated arXiv id — both flagged. **Limitation, live:** the share of papers without citation data, and a correct paraphrase the checker flags as unsupported (low word overlap). |
| 4:40–5:40 | Member B | Cell 7 + `notebooks/04_evaluation.ipynb` | **Evaluation.** 40 judged queries (20 foundational, 20 recent), dev/test split, tuned on dev only. Read the test nDCG@10 / P@10 for S1 (baseline), S3, S4, S7 by query type, the significance line (S7 vs S3, S7 vs S4), and one ablation. Say plainly where it did not win. |
| 5:40–6:00 | Member C | Terminal: `python -m scholar_ir.cli search "..." --system S7 --explain` | It's a CLI and notebooks, reproducible from the README with one config file and a seed. Next steps (one sentence). |

Each member must explain the component they built (adjust names/rows to the real work division).
