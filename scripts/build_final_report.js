const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, ShadingType, BorderStyle, HeadingLevel, Footer, PageNumber, LevelFormat, TabStopType,
} = require("docx");

const FIG = "/home/claude/scholar-ir/report/figures";
const OUT = process.argv[2] || "Scholar_IR_Final_Report.docx";
const FONT = "Times New Roman";
const W = 9026; // A4 text width in DXA with 1-inch margins

// ---- inline markup: **bold**, *italic*, `code`, ^sup^ -------------------------------------------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Courier New", size: (base.size || 22) - 2 }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 264 }, indent: opts.indent });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 240, after: 120 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 160, after: 80 } });
const BUL = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 264 } });
const NUM = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 264 } });
const EQ = (t, n) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 },
  tabStops: [{ type: TabStopType.RIGHT, position: W }], children: [...runs(t, { italics: false, size: 22 }), new TextRun({ text: n ? `\t(${n})` : "", font: FONT })] });
const CAP = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 200 }, children: runs(t, { size: 19, italics: false }) });

function IMG(file, widthIn) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthIn * 96, hpx = wpx * h / w;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
      altText: { title: file, description: file, name: file } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
function TABLE(rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      shading: i === 0 ? { fill: "E7EEF7", type: ShadingType.CLEAR, color: "auto" } :
        (opts.highlight && opts.highlight.includes(i) ? { fill: "FDF1E6", type: ShadingType.CLEAR, color: "auto" } : undefined),
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(String(c), { size: 19, bold: i === 0 }) })],
    })) })),
  });
}

// ------------------------------------------------------------------------------------------------
const C = [];
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Scholar-IR: Topic-Local Citation Authority for Scholarly Search", bold: true, size: 32, font: FONT })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs("Aryan, Atul and Rishith  (roll numbers: ____________________)", { size: 22 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs("CSD358 Information Retrieval, Final Project (extension of the mid-term project) · November 2026", { size: 20 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs("Code and data: https://github.com/AryanP78/scholar-ir", { size: 20 }) }));

// 2. Abstract
C.push(H1("Abstract"));
C.push(P("Scholarly search engines rank papers by text relevance and citation counts, but a global citation count rewards fame rather than importance to the query: a highly cited paper from a neighbouring field can outrank the papers that matter for the topic. We extend Scholar-IR, our from-scratch search engine over 80,000 arXiv computer-science papers, with **topic-local citation authority**. For every query we build the citation neighbourhood of its top-200 results (922,768 reference links, including papers outside the corpus) and run HITS on it, weighting each citing paper's vote by its BM25F relevance to the query. The resulting authority is blended with the mid-term's age-fair global authority. On an automatic citation-recommendation task over 300 held-out papers, the new system raises Recall@20 from 0.229 (BM25F + raw citations, the best mid-term system) to **0.277** (+0.048, 95% CI +0.035 to +0.061, p < 0.001) and MRR from 0.380 to 0.476. On 20 hand-built test queries it ties with BM25F on nDCG@10 (0.851 vs 0.855, n.s.) and improves MAP (0.864 vs 0.844, p = 0.017). We also add a multinomial Naive Bayes field classifier (78.6% accuracy, macro F1 0.73 over 7 classes; Rocchio and kNN as comparisons), use it with a rule guard to stop an LLM query parser from adding unrequested filters (exact filters 0.44 → 0.96), and fix a query-analysis bug with hyphenated words."));

// 3. Introduction
C.push(H1("1. Introduction"));
C.push(P("**Background.** Researchers search the literature in two modes: they look for the established papers of a topic, or for its newest work. General web search engines rank pages by text match plus link analysis, and scholarly engines do the same with citations in place of links. Google Scholar, for example, is reported to weight citation counts heavily (Beel and Gipp, 2009)."));
C.push(P("**Why it matters.** A citation count is a *global* signal. It says a paper is famous, not that it is important *for the question asked*. In our mid-term evaluation this produced concrete failures: for “sequence to sequence learning for machine translation”, BART (13,310 citations, a general text-generation model) ranked first, above machine-translation papers, because raw citations dominate. Normalising citations by publication year (our mid-term contribution) removes age bias but not this *topic* bias."));
C.push(P("**Motivation.** Kleinberg's HITS algorithm (1999) was designed for exactly this situation on the web: it computes authority only inside the link neighbourhood of the query's results, so a page is an authority when pages *about the query* point to it. Citation graphs are link graphs, so the same idea should identify the papers a topic's own literature relies on. The course covered HITS, PageRank, Naive Bayes and vector-space classification after the mid-term; this project applies them to the mid-term system and measures the result."));
C.push(P("**Contributions** of the final project (the mid-term system is summarised in Section 5.1):"));
C.push(NUM("**Topic-local, relevance-weighted HITS authority** over the full citation graph of the corpus, including 327,872 papers outside it, blended with the age-fair global authority, with the blend and weight tuned on development queries only (systems S9 and S10).", "c1"));
C.push(NUM("**A field classifier**: multinomial Naive Bayes (with and without mutual-information feature selection), Rocchio and kNN, implemented from scratch on our inverted index and evaluated with accuracy, precision, recall and F1.", "c1"));
C.push(NUM("**Fixes found in the mid-term evaluation**: a filter guard that keeps the LLM parser from inventing category and year filters, and a query-side fix for hyphenated words.", "c1"));
C.push(NUM("**Evaluation**: 1,440 graded query–paper pairs over 40 queries, an automatic citation-recommendation benchmark of 300 held-out papers, significance tests, ablations, and a leakage guard for graph-based methods.", "c1"));

// 4. Literature review
C.push(H1("2. Literature Review and Related Work"));
C.push(P("**Link analysis.** PageRank (Page et al., 1999) assigns each node a query-independent importance from the stationary distribution of a random walk. HITS (Kleinberg, 1999) is query-dependent: it builds a *base set* from the query's root results plus the pages they link to and that link to them, then iterates authority and hub scores, a ← Aᵀh and h ← Aa. Bharat and Henzinger (1998) showed that HITS suffers from *topic drift*, where the base set pulls in a neighbouring community whose own authorities win, and that weighting each node by its content relevance to the query reduces this. Our system applies relevance-weighted HITS to citations rather than web links."));
C.push(P("**Citation-based ranking and its biases.** Beel and Gipp (2009) reverse-engineered Google Scholar's ranking and found citation counts to be its strongest factor, which favours older papers. Radicchi et al. (2008) showed that citation counts become comparable across fields and years only after normalising by the average of papers of the same field and year; our mid-term *cohort percentile* authority follows this idea. McNee et al. (2002) introduced citation recommendation, predicting a paper's references from its content; we use this task as an automatic benchmark."));
C.push(P("**Ranking functions.** BM25F (Robertson et al., 2004) combines term frequencies from several fields with field weights *before* BM25's saturation, which works better than adding per-field BM25 scores. It is our base ranker (Manning et al., 2008, chapters 6 and 11)."));
C.push(P("**Text classification.** McCallum and Nigam (1998) compared Naive Bayes event models and showed the multinomial model is best for text with large vocabularies. Rocchio's centroid method (Rocchio, 1971) and k-nearest-neighbours are the vector-space classifiers of the course (Manning et al., 2008, chapters 13–14). Rahman et al. (2025) classified arXiv abstracts into top-level archives with TF-IDF features and report 0.65 accuracy (F-score 0.62) for Naive Bayes; we compare with this in Section 6.5."));
C.push(P("**LLMs in retrieval.** Retrieval-augmented generation (Lewis et al., 2020) grounds a language model's answer in retrieved documents. Our system keeps all ranking inside the IR engine and uses the LLM only to translate questions and summarise retrieved abstracts."));
C.push(P("**Novelty of our approach.** Prior scholarly rankers use either global citation counts or global graph scores (PageRank). We compute authority *per query*, on the query's citation neighbourhood, with Bharat–Henzinger relevance weighting, *over references that leave the corpus* (so a topic's classic papers count even when they are not indexed), and blend it with year-normalised global authority. We evaluate it both with graded judgments and with an automatic citation-recommendation task, guarding against the leakage that graph features introduce there."));

// 5. Objective
C.push(H1("3. Objective"));
C.push(P("Rank papers by how well they match the query *and* how important they are to the query's topic, rather than how famous they are overall, without hiding recent work. Concretely: (i) build topic-local authority from the citation graph and test whether it improves ranking over BM25F and over global citation authority; (ii) build and evaluate text classifiers that give the field of a paper or query; (iii) fix the parser and analysis errors the mid-term exposed; (iv) report all results with significance tests and an honest account of what did not improve."));

// 6. Proposed model
C.push(H1("4. Proposed Model"));
C.push(P("Figure 1 shows the system. Offline, arXiv metadata is preprocessed and indexed, citation data is collected, and the Naive Bayes classifier is trained. At query time, BM25F ranks the candidates, topic-local HITS scores the top-200 using their citation neighbourhood, and the net score combines relevance with authority using a weight λ(q) chosen from the query's intent. The results page shows each paper's relevance, topic authority and global authority, the predicted field, and the topic's main hub papers (usually surveys)."));
C.push(IMG("fig1_pipeline.png", 6.3));
C.push(CAP("**Figure 1.** Scholar-IR pipeline. Thick-bordered boxes are new in the final project."));

// 7. Methodology
C.push(H1("5. Methodology"));
C.push(H2("5.1 Mid-term system (summary)"));
C.push(P("**Corpus and indexes.** 80,000 arXiv papers (2010–2023) from cs.IR, cs.CL, cs.LG, cs.AI, cs.CV, cs.DB and cs.SI. Text is case-folded, tokenised, stop-worded and Porter-stemmed. We build a positional index with title and abstract zones plus year and category indexes: 132,278 terms, 6.06 million postings, 297 MB, built in about 30 s. Boolean AND/OR/NOT, phrase, NEAR/k and field queries are supported, with skip pointers."));
C.push(P("**BM25F.** For query terms t and document d with zone frequencies tf_{z}, zone lengths and weights w_{z}:"));
C.push(EQ("tf̃(t,d) = Σ_z  w_z · tf_z(t,d) / (1 − b_z + b_z · len_z(d)/avglen_z),     BM25F(q,d) = Σ_t  idf(t) · tf̃ / (k₁ + tf̃)", 1));
C.push(P("with w_{title} = 4.0 and w_{abstract} = 1.0 tuned on development queries, k₁ = 1.2, b_{title} = 0.5, b_{abstract} = 0.75."));
C.push(P("**Global authority and the net score.** Each paper gets an age-fair authority A2(d): its citation percentile among papers of the same year (cohort). The top-N = 200 BM25F candidates are re-ranked by"));
C.push(EQ("net(q,d) = (1 − λ(q)) · R(q,d) + λ(q) · g(d)", 2));
C.push(P("where R is min–max normalised BM25F and λ(q) = λ_{min} + (λ_{max} − λ_{min}) · σ(wᵀx(q)) comes from a logistic model of query features (foundational or recent cue words, mean idf, query length, score flatness)."));

C.push(H2("5.2 Topic-local relevance-weighted HITS (new)"));
C.push(P("**Citation graph.** For 59,369 papers matched in OpenAlex we kept their *full* reference lists: 922,768 citation links, of which only 48,857 point to papers inside our corpus. The other targets become 327,872 extra graph nodes. They are never returned as results, but they let papers that cite the same outside classic reinforce each other."));
C.push(P("**Base set.** For query q with root set R (the top-200 BM25F results), the base set is"));
C.push(EQ("B = R ∪ {papers cited by any r ∈ R} ∪ {up to 100 corpus papers citing each r ∈ R}", 3));
C.push(P("(on average 2,080 papers and 4,150 links per query). **Relevance weighting** (Bharat and Henzinger, 1998): the edge u → v gets weight ω(u) = min(1, BM25F(q,u) / max_{r∈R} BM25F(q,r)), so a citing paper votes only as much as it is about the query. **Iteration** with the weighted adjacency matrix A of B, starting from h = 1/√|B|:"));
C.push(EQ("a ← Aᵀh / ‖Aᵀh‖₂ ,     h ← A a / ‖A a‖₂     (until Σ|Δa| + Σ|Δh| < 10⁻⁸)", 4));
C.push(P("Convergence takes 49 iterations on average. The **topic authority** of a root paper is the rank percentile of a(d) among the root papers (0 when nobody in B cites it). The final authority blends it with the global one, and **S9** uses it in Eq. 2 with a fixed λ; **S10** uses the query-adaptive λ(q):"));
C.push(EQ("g(d) = β · pct_R(a(d)) + (1 − β) · A2(d)", 5));
C.push(P("β ∈ {0.5, 1.0} and λ ∈ {0.1, …, 0.6} were chosen on the 20 development queries only: β = 0.5, λ = 0.3. Hub scores h(d) are shown to the user as “key surveys for this topic”. The extra work costs about 6 ms per query (mean latency 13.0 ms for S9 vs 7.1 ms for BM25F)."));

C.push(H2("5.3 Text classification (new)"));
C.push(P("Each paper's label is its primary arXiv category among the seven. The document–term matrix is built directly from the index postings (same stemmed vocabulary as search). With a stratified 80/20 split (64,000 / 16,000 papers):"));
C.push(BUL("**Multinomial Naive Bayes** with add-one smoothing:"));
C.push(EQ("ĉ = argmax_c [ log P(c) + Σ_t tf(t,d) · log P(t|c) ],     P(t|c) = (T_ct + 1) / (Σ_t' T_ct' + |V|)", 6));
C.push(BUL("**Naive Bayes with feature selection**, keeping the k terms with the highest mutual information I(U_t; C_c) (Manning et al., 2008, eq. 13.17), for k ∈ {1,000; 5,000; 20,000}."));
C.push(BUL("**Rocchio**: centroid of length-normalised (1 + log tf)·idf vectors per class, nearest centroid by cosine."));
C.push(BUL("**kNN**: the k = 10 nearest training papers by cosine, votes weighted by similarity."));
C.push(P("The trained Naive Bayes model also predicts the field of each query and shows it in the app as a soft hint (e.g. “cs.CL 85%, add `cat:cs.CL` to filter”) instead of applying a filter automatically."));

C.push(H2("5.4 Fixes from the mid-term evaluation (new)"));
C.push(P("**LLM filter guard.** The Gemini query parser added category filters nobody asked for in 13 of its 14 errors, and turned words like “new” into year ranges. Filters remove papers outright, so the guarded parser keeps the LLM's topic words, phrases and intent but takes categories only from fields the user names and years only when the request states a year."));
C.push(P("**Hyphenated query words.** A hyphenated word is indexed as its parts and as the joined form. On the query side this counted “LLM-based” twice (llm + llmbased) against a single “recommender”, so “LLM-based recommender systems” returned LLM-agent papers. Queries now use the parts only; documents still index both forms."));

// 8. Experiments
C.push(H1("6. Experimentation and Results"));
C.push(H2("6.1 Datasets and protocol"));
C.push(BUL("**Corpus**: 80,000 arXiv CS papers (titles, abstracts, years, categories) with Semantic Scholar/OpenAlex citation counts (99% coverage) and OpenAlex reference lists."));
C.push(BUL("**Queries**: 40 queries, 20 *foundational* (e.g. “seminal work on generative adversarial networks”) and 20 *recent* (e.g. “LLM-based recommender systems”), each with a written relevance description. They are split into 20 development queries (all tuning) and 20 test queries (all reported ranking results)."));
C.push(BUL("**Judgments**: the top 15 of every system were pooled and graded 0/1/2 blind to the system, giving 1,440 graded pairs (411 / 468 / 561). The grades were produced by an AI judge (Claude) against each query's description; 358 of them are new in this project. A team member re-grades six overlapping queries to measure agreement (Section 7)."));
C.push(BUL("**Citation recommendation** (no human labels): for 300 random papers with at least 5 in-corpus references, the query is the paper's 30 highest tf-idf terms; relevant = its in-corpus references; candidates are restricted to papers published no later than it. For graph methods the held-out paper is **removed from the citation graph**, so its own reference list cannot vote for the answers."));
C.push(BUL("**Metrics**: P@5, P@10, R@10, R@20, MAP, nDCG@10 (gain 2^g − 1), MRR; for classification accuracy, macro precision/recall/F1. Significance: paired bootstrap (10,000 resamples) and Wilcoxon signed-rank test."));

C.push(H2("6.2 Ranking results on judged test queries"));
C.push(TABLE([
  ["System", "P@5", "P@10", "R@20", "MAP", "nDCG@10", "nDCG@10 found.", "nDCG@10 recent"],
  ["S1 tf-idf (baseline)", "0.820", "0.785", "0.583", "0.665", "0.665", "0.659", "0.671"],
  ["S3 BM25F", "0.930", "0.910", "0.694", "0.844", "**0.855**", "**0.849**", "0.861"],
  ["S4 BM25F + raw citations", "0.950", "0.910", "**0.729**", "**0.867**", "0.848", "0.842", "0.854"],
  ["S7 mid-term (cohort + λ(q))", "**0.960**", "0.910", "0.716", "0.854", "0.851", "0.837", "0.865"],
  ["S9 BM25F + topic HITS (new)", "0.950", "0.915", "0.721", "0.864", "0.851", "0.845", "0.856"],
  ["S10 S9 + adaptive λ(q) (new)", "0.950", "**0.920**", "0.719", "0.864", "0.853", "0.837", "**0.870**"],
], [2350, 760, 760, 760, 760, 960, 1000, 1000], { highlight: [5, 6] }));
C.push(CAP("**Table 1.** Test-split results (20 queries). Bold = best in column."));
C.push(IMG("fig2_judged_results.png", 6.2));
C.push(CAP("**Figure 2.** nDCG@10 and MAP on the test queries by query type."));
C.push(P("Zone-aware BM25F is the largest single improvement over tf-idf (+0.190 nDCG@10, 95% CI +0.121 to +0.260, p < 0.001). Above BM25F the systems are close: S9 vs S3 differs by −0.005 nDCG@10 (p = 0.74) but **+0.020 MAP (CI +0.004 to +0.037, p = 0.017)**; S10 vs S3 gains +0.020 MAP (p = 0.003). S9 and S4 do not differ significantly on any metric. With BM25F already at 0.855 nDCG@10 and only 20 test queries, differences smaller than about 0.03 cannot be detected."));

C.push(H2("6.3 Citation recommendation"));
C.push(IMG("fig3_citerec.png", 6.2));
C.push(CAP("**Figure 3.** Recall@20 on 300 held-out papers."));
C.push(TABLE([
  ["System", "Recall@10", "Recall@20", "MAP", "MRR"],
  ["S3 BM25F", "0.107", "0.149", "0.058", "0.245"],
  ["S4 BM25F + raw citations", "0.174", "0.229", "0.105", "0.380"],
  ["S7 mid-term", "0.119", "0.172", "0.068", "0.275"],
  ["S8 PageRank authority", "0.128", "0.183", "0.076", "0.297"],
  ["S9 topic HITS (new)", "**0.227**", "**0.277**", "**0.143**", "**0.476**"],
  ["S10 topic HITS + λ(q) (new)", "0.137", "0.192", "0.080", "0.315"],
], [3200, 1450, 1450, 1450, 1450], { highlight: [5] }));
C.push(CAP("**Table 2.** Citation recommendation (300 papers)."));
C.push(P("Topic-local HITS is the best system on this task: **+0.048 Recall@20 over S4** (95% CI +0.035 to +0.061, p < 0.001), +0.128 over BM25F and +0.105 over the mid-term system, and MRR rises from 0.380 to 0.476. A paper's references are the papers its topic relies on, which is what topic authority measures. Global PageRank does much worse (0.183), because a global score cannot tell which important papers belong to *this* topic. S10 gains less than S9 because the adaptive λ(q) reads these long, cue-free queries as non-foundational and gives authority a small weight; a fixed λ = 0.3 suits this task better."));

C.push(H2("6.4 Ablations and case studies"));
C.push(TABLE([
  ["Variant (test, nDCG@10)", "Found.", "Recent", "All"],
  ["S9 full: relevance-weighted HITS, blended with A2", "0.845", "0.856", "0.851"],
  ["Classic (unweighted) HITS, blended", "0.840", "0.856", "0.848"],
  ["Relevance-weighted HITS only (β = 1)", "0.844", "0.823", "0.834"],
  ["Classic HITS only (Kleinberg, β = 1)", "0.837", "0.825", "0.831"],
  ["S3 without stemming", "0.769", "0.647", "0.708"],
  ["S3 champion lists (r = 500)", "0.830", "0.828", "0.829"],
], [5000, 1340, 1340, 1340]));
C.push(CAP("**Table 3.** Ablations. Relevance weighting and blending each help; HITS alone hurts recent queries."));
C.push(P("HITS alone penalises recent papers, which have had little time to be cited inside the topic (recent nDCG@10 0.823). Blending with the age-fair global authority restores them (0.856), so the mid-term contribution and the new one are complementary. Relevance weighting beats Kleinberg's original algorithm, as Bharat and Henzinger (1998) found on the web."));
C.push(P("**Case study, “sequence to sequence learning for machine translation”.** With raw citations (S4) BART (13,310 citations) ranks first. With topic HITS (S9) BART leaves the top five, and the first result is *Sequence Transduction with Recurrent Neural Networks* (2012), an early sequence-to-sequence paper that the topic's own literature cites. nDCG@10 rises from 0.80 (S4) to 0.81 (S9), though BM25F alone scores 0.96 on this query."));
C.push(P("**Case study, hyphen fix.** For “LLM-based recommender systems” the top five are now all LLM-recommender papers (Chat-REC, GenRec, RecMind…). At the mid-term they were LLM-agent papers and only 4 of 22 pooled papers were relevant."));

C.push(H2("6.5 Classification results and comparison with an existing paper"));
C.push(TABLE([
  ["Classifier (16,000 test papers, 7 classes)", "Accuracy", "Macro P", "Macro R", "Macro F1", "Lenient acc."],
  ["Majority class (baseline)", "0.328", "0.047", "0.143", "0.071", "0.450"],
  ["Multinomial Naive Bayes, all terms", "**0.786**", "**0.779**", "0.702", "0.728", "**0.854**"],
  ["Naive Bayes, top-1,000 MI terms", "0.769", "0.705", "0.772", "0.732", "0.837"],
  ["Naive Bayes, top-5,000 MI terms", "0.776", "0.713", "**0.780**", "0.741", "0.842"],
  ["Naive Bayes, top-20,000 MI terms", "0.783", "0.729", "0.772", "**0.749**", "0.849"],
  ["Rocchio (centroids)", "0.762", "0.703", "0.767", "0.730", "0.830"],
  ["kNN (k = 10, cosine)", "0.764", "0.750", "0.690", "0.714", "0.834"],
], [3426, 1120, 1120, 1120, 1120, 1120]));
C.push(CAP("**Table 4.** Classification results. “Lenient” counts a prediction correct if it is any of the paper's categories."));
C.push(IMG("fig5_confusion.png", 3.4));
C.push(CAP("**Figure 4.** Row-normalised confusion matrix of the best classifier."));
C.push(P("Naive Bayes is the most accurate (78.6%) and the fastest; mutual-information feature selection trades a little accuracy for better recall on small classes (macro F1 0.749 with 20,000 terms). kNN is 600× slower at prediction and less accurate. As a correctness check, our Naive Bayes predicts the same class as scikit-learn's MultinomialNB for all 16,000 test papers. Most errors are between overlapping fields: cs.AI papers are often predicted as cs.LG (18%), and cs.IR has the lowest recall (0.43 for plain Naive Bayes), since many retrieval papers are filed under NLP or databases."));
C.push(P("**Comparison with Rahman et al. (2025).** They classify arXiv abstracts with TF-IDF features and report accuracy 0.65 and F-score 0.62 for Naive Bayes, and 0.69 accuracy for logistic regression. Our Naive Bayes reaches 0.786 accuracy and 0.73–0.75 macro F1. The settings differ, so this is not a like-for-like comparison: they predict broad archives (physics, mathematics, computer science…) in a multi-label set-up on 1.2 million records, while we predict the primary label among seven overlapping CS sub-fields. Their paper is also a non-peer-reviewed preprint whose text and tables disagree on some numbers. What the comparison does show is that a carefully preprocessed multinomial Naive Bayes on stemmed, stop-worded terms is a strong baseline for abstract classification."));

C.push(H2("6.6 LLM query parser"));
C.push(TABLE([
  ["Parser (25 requests with gold parses)", "Term F1", "Categories right", "Years right", "All filters exact", "Intent right"],
  ["Rule-based", "0.940", "0.96", "1.00", "0.96", "0.92"],
  ["Gemini (mid-term)", "0.975", "0.48", "0.88 / 0.92", "0.44", "0.72"],
  ["Gemini + filter guard (new)", "**0.975**", "**0.96**", "**1.00**", "**0.96**", "0.72"],
], [3226, 1160, 1160, 1160, 1160, 1160]));
C.push(CAP("**Table 5.** Query parsing. Years: year_min / year_max accuracy."));
C.push(P("The guard keeps the LLM's better topic words and the rules' correct filters. Caveat: the gold parses were drafted with the same AI assistant that wrote the rule cues, which favours the rules."));

// 9. Conclusions
C.push(H1("7. Conclusions and Limitations"));
C.push(P("**Conclusions.** Topic-local, relevance-weighted HITS turns citation data into a topic-specific signal. It is the clear winner on citation recommendation (+21% relative Recall@20 over raw citations, p < 0.001) and improves MAP over BM25F on judged queries, while it ties on nDCG@10, where BM25F is already near the ceiling. HITS needs the age-fair global authority beside it, because on its own it penalises recent papers. Naive Bayes on our own index classifies papers into seven CS fields with 78.6% accuracy, and using rules rather than the LLM for filters fixes the parser's main error."));
C.push(P("**Limitations.**"));
C.push(BUL("**AI-generated relevance judgments.** All grades were produced by an AI judge (Claude) using written descriptions that the same assistant drafted. This may favour the descriptions' wording. Human re-grading of six queries is in progress; until it is complete, agreement with human judgment is not measured."));
C.push(BUL("**Small test set.** 20 test queries detect only differences of about 0.03 nDCG@10 or more."));
C.push(BUL("**Graph coverage.** 34,540 of 80,000 papers have reference lists; new papers have few in-topic citations, which is why the blend is needed."));
C.push(BUL("**Unfixed failure.** “Tutorial on graph neural networks” still returns famous off-topic tutorials, because the BM25F root set itself is dominated by the word “tutorial”; HITS can only re-rank what retrieval finds."));
C.push(BUL("**Citation-recommendation caveats.** Global citation counts (S4–S7) include the held-out paper's own citation (+1), and the task rewards citation-based methods by design."));
C.push(P("**Future work:** learn the blend weight per query from the graph's density, treat document-type words (tutorial, survey) as intent only, and measure human–AI judgment agreement on all 40 queries."));

// 10. References
C.push(H1("References"));
const refs = [
  "Beel, J. and Gipp, B., 2009, July. Google Scholar's ranking algorithm: an introductory overview. In Proceedings of the 12th International Conference on Scientometrics and Informetrics (ISSI'09) (Vol. 1, pp. 230–241).",
  "Bharat, K. and Henzinger, M.R., 1998, August. Improved algorithms for topic distillation in a hyperlinked environment. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 104–111).",
  "Kleinberg, J.M., 1999. Authoritative sources in a hyperlinked environment. Journal of the ACM, 46(5), pp. 604–632.",
  "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.T., Rocktäschel, T. and Riedel, S., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33 (pp. 9459–9474).",
  "Manning, C.D., Raghavan, P. and Schütze, H., 2008. Introduction to Information Retrieval. Cambridge University Press.",
  "McCallum, A. and Nigam, K., 1998, July. A comparison of event models for naive Bayes text classification. In AAAI-98 Workshop on Learning for Text Categorization (Vol. 752, pp. 41–48).",
  "McNee, S.M., Albert, I., Cosley, D., Gopalkrishnan, P., Lam, S.K., Rashid, A.M., Konstan, J.A. and Riedl, J., 2002, November. On the recommending of citations for research papers. In Proceedings of the 2002 ACM Conference on Computer Supported Cooperative Work (pp. 116–125).",
  "Page, L., Brin, S., Motwani, R. and Winograd, T., 1999. The PageRank citation ranking: Bringing order to the web. Technical Report, Stanford InfoLab.",
  "Radicchi, F., Fortunato, S. and Castellano, C., 2008. Universality of citation distributions: Toward an objective measure of scientific impact. Proceedings of the National Academy of Sciences, 105(45), pp. 17268–17272.",
  "Rahman, S., Shanto, H.K., Koana, U.A. and Danish, S.M., 2025. Automated research article classification and recommendation using NLP and ML. arXiv preprint arXiv:2510.05495.",
  "Robertson, S., Zaragoza, H. and Taylor, M., 2004, November. Simple BM25 extension to multiple weighted fields. In Proceedings of the 13th ACM International Conference on Information and Knowledge Management (pp. 42–49).",
  "Rocchio, J.J., 1971. Relevance feedback in information retrieval. In G. Salton (ed.), The SMART Retrieval System: Experiments in Automatic Document Processing (pp. 313–323). Prentice-Hall.",
];
refs.forEach((r) => C.push(new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(r, { size: 20 }), spacing: { after: 60 }, alignment: AlignmentType.LEFT })));

C.push(H1("Appendix: AI use and reproducibility"));
C.push(P("Claude (Anthropic) was used as a coding assistant for most of the code and drafted this report from the team's results; Claude also produced the relevance grades (Section 6.1). Google Gemini is used inside the system only to parse questions and write cited answers from retrieved abstracts. All numbers in this report come from files in `results/` produced by `python -m scholar_ir.experiments all`, `python -m scholar_ir.classify eval` and `python -m scholar_ir.llm_experiments parser`; the 49 unit tests run with `make test`."));

const doc = new Document({
  creator: "Aryan, Atul and Rishith", title: "Scholar-IR: Topic-Local Citation Authority for Scholarly Search",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "c1", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 540 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
