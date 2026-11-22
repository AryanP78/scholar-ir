// Builds report/Scholar_IR_Final_Report.docx from the verified results. Run: node scripts/build_report_v2.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  ShadingType, BorderStyle, HeadingLevel, Footer, Header, PageNumber, LevelFormat, TabStopType, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const A = path.join(ROOT, "report", "assets");
const OUT = path.join(ROOT, "report", "Scholar_IR_Final_Report.docx");
const FONT = "Times New Roman";
const NAVY = "1F3A5F";
const W = 9026;

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Courier New", size: (base.size || 22) - 3 }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, line: 276 }, indent: o.indent, keepNext: o.keepNext });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ keepNext: true, spacing: { before: 100, after: 60 }, children: [new TextRun({ text: t, font: FONT, bold: true, italics: true, size: 22, color: NAVY })] });
const BUL = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 50, line: 276 } });
let numRefCount = 0;
const NUMLIST = (items) => { const ref = `n${++numRefCount}`; numberingConfigs.push(ref); return items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 50, line: 276 } })); };
const numberingConfigs = [];
let eqn = 0;
const EQ = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  tabStops: [{ type: TabStopType.RIGHT, position: W }],
  children: [...runs(t.replace(/_([A-Za-z0-9]+)/g, "_{$1}"), { size: 22 }), new TextRun({ text: `\t(${++eqn})`, font: FONT, size: 21 })] });
const CAP = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 220 }, children: runs(t, { size: 19 }) });
const TCAP = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true, children: runs(t, { size: 19 }) });

function IMG(file, widthIn) {
  const buf = fs.readFileSync(path.join(A, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthIn * 96, hpx = wpx * h / w;
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx }, altText: { title: file, description: file, name: file } })] });
}
function FIG(file, widthIn, cap) {
  const buf = fs.readFileSync(path.join(A, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthIn * 96, hpx = wpx * h / w;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 200 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx }, altText: { title: file, description: file, name: file } }),
      new TextRun({ break: 1, text: "", size: 8 }), ...runs(cap, { size: 19 })] });
}
const B = { style: BorderStyle.SINGLE, size: 4, color: "B8BEC6" };
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, cantSplit: true, children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders: { top: B, bottom: B, left: B, right: B },
      shading: i === 0 ? { fill: "E3E8EF", type: ShadingType.CLEAR, color: "auto" } : (o.hl && o.hl.includes(i) ? { fill: "F3F5F8", type: ShadingType.CLEAR, color: "auto" } : undefined),
      margins: { top: 45, bottom: 45, left: 80, right: 80 },
      children: [new Paragraph({ keepNext: i < rows.length - 1 ? true : undefined, alignment: (o.left && o.left.includes(j)) || j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(String(c), { size: o.size || 19, bold: i === 0 }) })] })) })) });
}
function CODE(lines) {
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({ children: [new TableCell({
    width: { size: W, type: WidthType.DXA }, borders: { top: B, bottom: B, left: B, right: B },
    shading: { fill: "F6F7F9", type: ShadingType.CLEAR, color: "auto" }, margins: { top: 90, bottom: 90, left: 140, right: 140 },
    children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 252 }, children: [new TextRun({ text: l, font: "Courier New", size: 17 })] })) })] })] });
}
const SP = (n = 80) => new Paragraph({ spacing: { after: n }, children: [] });

// ---- report v3 body (spliced between the helpers and the DOCUMENT block of build_report_v2.js)
function IMG2(f1, c1, f2, c2, w1, w2) {
  const cell = (f, c, wIn) => {
    const buf = fs.readFileSync(path.join(A, f));
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
    const wpx = wIn * 96, hpx = wpx * h / w, dxa = Math.round(W * wIn / (w1 + w2));
    return new TableCell({ width: { size: dxa, type: WidthType.DXA }, borders: { top: NB, bottom: NB, left: NB, right: NB },
      verticalAlign: "bottom", margins: { left: 40, right: 40 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx }, altText: { title: f, description: f, name: f } })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 }, children: runs(c, { size: 19 }) })] });
  };
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [Math.round(W * w1 / (w1 + w2)), W - Math.round(W * w1 / (w1 + w2))],
    rows: [new TableRow({ cantSplit: true, children: [cell(f1, c1, w1), cell(f2, c2, w2)] })] });
}

// ================================================================ TITLE PAGE
const T = [];
T.push(SP(1500));
T.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "Scholar-IR", font: FONT, size: 52, bold: true, color: NAVY })] }));
T.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 500 }, children: [new TextRun({ text: "Academic Paper Retrieval with Topic-Specific Citation Authority", font: FONT, size: 30, color: NAVY })] }));
T.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: runs("Final Project Report", { size: 24, bold: true }) }));
T.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 500 }, children: runs("CSD358 — Information Retrieval", { size: 24 }) }));
T.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs("Submitted by", { size: 21 }) }));
T.push(new Table({ width: { size: 5200, type: WidthType.DXA }, alignment: AlignmentType.CENTER, columnWidths: [2600, 2600],
  rows: [["Name", "Roll number"], ["Aryan [SURNAME]", "[ROLL NUMBER]"], ["Atul [SURNAME]", "[ROLL NUMBER]"], ["Rishith [SURNAME]", "[ROLL NUMBER]"]].map((r, i) =>
    new TableRow({ children: r.map((c) => new TableCell({ width: { size: 2600, type: WidthType.DXA }, borders: { top: NB, bottom: i === 0 ? B : NB, left: NB, right: NB },
      margins: { top: 40, bottom: 40, left: 80, right: 80 }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: runs(c, { size: 22, bold: i === 0 }) })] })) })) }));
T.push(SP(420));
for (const l of ["Course instructor: Sonia Khetarpaul", "Department of Computer Science and Engineering [VERIFY]", "Shiv Nadar Institution of Eminence, Delhi-NCR [VERIFY]",
  "Monsoon Semester 2026 [VERIFY] · 22 November 2026", "Code and data: https://github.com/AryanP78/scholar-ir"])
  T.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: runs(l, { size: 21 }) }));

// ================================================================ BODY
const C = [];
C.push(H1("Abstract"));
C.push(P("Academic search engines rank papers by text relevance and citation counts, but a citation count measures how famous a paper is overall, not how important it is for the topic searched. In our mid-term system this let BART, a general text-generation model with 13,310 citations, rank first for “sequence to sequence learning for machine translation”. For the final project we extended Scholar-IR, our from-scratch search engine over 80,000 arXiv computer-science papers, with **topic-specific citation authority**: for each query, HITS runs on the citation network of its 200 best BM25F matches (922,768 citation links in total), with each citing paper weighted by its relevance to the query, and the result is blended with year-normalised global authority. We also added Naive Bayes, Rocchio and kNN field classifiers and fixed two query-processing errors. In a citation-recommendation test on 300 papers, Recall@20 rises from 0.229 (BM25F with raw citation counts) to 0.277 (p < 0.001). On 20 graded test queries the new ranking ties with BM25F on nDCG@10 (0.851 vs 0.855) and raises MAP from 0.844 to 0.864. Naive Bayes classifies papers into seven fields with 78.6% accuracy. The ranking grades were produced by an AI judge and have not been checked against human judgments."));

// ---- 1
C.push(H1("1. Introduction"));
C.push(H2("1.1 Background and motivation"));
C.push(P("Finding prior work is one of the most common search tasks in research, and a typical query matches thousands of papers. Ranking functions such as BM25 (Robertson and Zaragoza, 2009) decide which papers *talk about* a topic, not which ones *matter* for it, so academic search engines also use citation counts. Citation counts have two biases. Older papers have had more time to be cited (**age bias**), and a paper famous in a neighbouring field outranks the papers a topic actually builds on (**topic bias**). Our mid-term project corrected age bias by comparing each paper only with papers from the same year. Its evaluation exposed topic bias: with raw citation authority, BART ranked first for a machine-translation query because of 13,310 citations that came almost entirely from outside the topic."));
C.push(P("Kleinberg (1999) faced the same problem on the web. His HITS algorithm computes authority only inside the link neighbourhood of a query's results, so a page becomes an authority when pages about the query link to it. A citation network is a link network between papers, so a paper should count as important *for a topic* when the topic's own literature cites it. The final project tests this idea."));
C.push(H2("1.2 What the final project adds"));
C.push(P("We chose to continue our mid-term project rather than start a new one. Everything from the mid-term (indexes, Boolean and phrase search, BM25F ranking, year-normalised citation authority, the adaptive weight λ(q), the Gemini parser, RAG answers and the citation checker) is kept. Table 1 lists everything added in the final project and where each item is described."));
C.push(TCAP("**Table 1.** What the final project adds to the mid-term system."));
C.push(TABLE([
  ["Area", "What we added", "Section"],
  ["Citation data", "Full OpenAlex reference lists: 922,768 links (the mid-term used 48,857 links inside the collection).", "5.4"],
  ["Ranking (novelty)", "Topic-specific, relevance-weighted HITS blended with global authority; new systems S9 and S10.", "4, 5.4"],
  ["Classification", "Naive Bayes (with mutual-information feature selection), Rocchio and kNN, compared on seven fields.", "5.5, 7.6"],
  ["Query processing", "Filter guard for the LLM parser; hyphenated query words no longer counted twice.", "5.6, 7.7"],
  ["Interface", "Old-vs-new ranking view, score bars, predicted field and topic hubs.", "5.7"],
  ["Evaluation", "Pool extended to 1,440 graded pairs; citation recommendation; HITS ablations; significance tests.", "6, 7"],
], [1800, 6226, 1000], { left: [1], size: 18 }));
C.push(H2("1.3 Contributions"));
C.push(P("The contributions are: (i) a query-specific citation authority based on relevance-weighted HITS, blended with year-normalised global authority; (ii) a from-scratch comparison of Naive Bayes, Rocchio and kNN for classifying abstracts; and (iii) an evaluation with graded queries, citation recommendation, ablations and significance tests."));

// ---- 2
C.push(H1("2. Literature Review"));
C.push(P("**Text ranking.** BM25 scores a document by summing, over query terms, an idf weight times a saturating function of term frequency (Robertson and Zaragoza, 2009). Robertson, Zaragoza and Taylor (2004) showed that adding separate BM25 scores per field breaks this saturation and proposed BM25F, which combines weighted field frequencies before saturating. Our indexing, tf-idf and evaluation methods follow Manning, Raghavan and Schütze (2008)."));
C.push(P("**Link analysis.** PageRank (Page et al., 1999) gives each node one query-independent score from a random walk on the link graph. HITS (Kleinberg, 1999) is query-dependent: it expands the retrieved pages into a base set of linked pages and computes mutually reinforcing *authority* and *hub* scores. Bharat and Henzinger (1998) showed that HITS suffers from *topic drift* towards densely linked neighbouring communities and reduced it by weighting nodes by their similarity to the query."));
C.push(P("**Text classification.** McCallum and Nigam (1998) found multinomial Naive Bayes better than the Bernoulli model for large vocabularies. Rocchio classification represents each class by its centroid, and kNN uses the labels of the nearest training documents (Manning, Raghavan and Schütze, 2008, Ch. 13–14). Rahman et al. (2025) classify arXiv abstracts with TF-IDF features and report 0.65 accuracy for Naive Bayes."));
C.push(P("**Position of this work.** Citation counts and PageRank are query-independent, so they cannot tell which important papers belong to the current topic; HITS is query-dependent but was designed for web pages and drifts in its original form. We do not claim these algorithms as new. Our contribution is applying relevance-weighted HITS to each query's citation neighbourhood, with BM25F as the relevance measure, blending it with age-fair global authority, and measuring the effect."));

// ---- 3
C.push(H1("3. Problem Statement and Objectives"));
C.push(P("**Problem.** Given a query and a collection of papers with citation data, rank papers by how well they match the query and by how important they are *to the query's topic*, without penalising recent papers for having had less time to be cited."));
C.push(P("**Objectives.** (1) Build a topic-specific authority signal from the citation network and measure whether it improves ranking over BM25F and over global citation authority. (2) Implement Naive Bayes, Rocchio and kNN for predicting a paper's field and compare them on accuracy, precision, recall and F1. (3) Fix the query-parsing and query-analysis errors found in the mid-term evaluation. (4) Report every result with significance tests, including where the new method does not help."));

// ---- 4
C.push(H1("4. Proposed System and Architecture"));
C.push(P("Figure 1 shows the three paths of the system. The **offline path** builds the indexes, the citation graph and the classifier once. The **search path** runs per query: filters restrict the candidates, BM25F keeps the best 200 as the root set, topic HITS computes authority on their citation neighbourhood, and the net score produces the final ranking. The optional **ask path** needs a Gemini API key; it wraps the search path with an LLM parser and an answer generator, and our own code checks each answer sentence against the abstract it cites."));
C.push(FIG("fig1_architecture.png", 6.2, "**Figure 1.** Scholar-IR architecture. Components new in the final project are marked as in the legend. The field classifier feeds the results page as a displayed hint, not the ranking."));
C.push(P("Figure 2 shows part of the real topic graph for the machine-translation query. Hub papers on the left cite many of the topic's important papers on the right. *Sequence to Sequence Learning with Neural Networks* (2014) receives the highest authority, while BART, despite 13,310 global citations, receives zero because no paper in the weighted base set cites it."));
C.push(FIG("fig2_citation_network.png", 5.0, "**Figure 2.** Part of the topic citation graph for “sequence to sequence learning for machine translation”: the five strongest hubs, the five strongest root-set authorities and BART."));
//

// ---- 5
C.push(H1("5. Methodology and Implementation"));
C.push(H2("5.1 Data and indexing"));
C.push(P("The corpus has 80,000 arXiv papers first submitted between 2010 and 2023 in seven categories (cs.LG, cs.CV, cs.CL, cs.AI, cs.SI, cs.IR, cs.DB), sampled per year from the public arXiv metadata snapshot. Text is case-folded, tokenised (hyphenated words are indexed as parts and as the joined form), stop-worded and Porter-stemmed. The positional index keeps separate title, abstract and combined zones plus year and category postings: 132,278 terms, 6.06 million postings, 297 MB, built in 31 s. AND queries intersect postings in increasing document-frequency order with skip pointers, which cut the postings touched per query from 45,052 to 19,154 (4.2 ms to 2.5 ms)."));
C.push(H2("5.2 BM25F relevance"));
C.push(P("For term t, document d and zone z ∈ {title, abstract}, with zone frequency tf_{z}, zone length len_{z}, average length avglen_{z}, zone weight w_{z} and length parameter b_{z}:"));
C.push(EQ("t̃f(t,d) = Σ_z  w_z · tf_z(t,d) / ( 1 − b_z + b_z · len_z(d) / avglen_z )"));
C.push(EQ("BM25F(q,d) = Σ_{t∈q}  idf(t) · t̃f(t,d) / ( k₁ + t̃f(t,d) ),     idf(t) = ln( 1 + (N − df_t + 0.5) / (df_t + 0.5) )"));
C.push(P("N is the collection size and df_{t} the document frequency. The title weight was tuned on the development queries over {1.0, 1.5, 2.5, 4.0}; 4.0 was best and lies at the edge of that grid. The other values are fixed: w_{abstract} = 1.0, k₁ = 1.2, b_{title} = 0.5, b_{abstract} = 0.75."));
C.push(H2("5.3 Global authority and adaptive weight (mid-term)"));
C.push(P("The **global authority** A2(d) is the mid-rank percentile of d's citation count among papers from the same year (adjacent years are pooled until a cohort has 200 papers; papers without data get 0.5). Relevance is min–max normalised over the top 200 candidates, R(q,d) ∈ [0,1], and combined with an authority score g(d):"));
C.push(EQ("net(q,d) = ( 1 − λ(q) ) · R(q,d) + λ(q) · g(d)"));
C.push(P("λ(q) comes from a logistic model over query features x(q): foundational cue words (*survey*, *seminal*), recency cues (*recent*, a year ≥ 2020), standardised mean idf and length, and flatness of the BM25F scores, with weights w fitted on the development queries:"));
C.push(EQ("λ(q) = λ_min + ( λ_max − λ_min ) · σ( wᵀ x(q) ),     λ_min = 0.05,  λ_max = 0.7"));
C.push(H2("5.4 Topic-specific HITS authority (new)"));
C.push(P("**Citation graph.** The mid-term pipeline kept only references between papers in our collection (48,857 links). We recovered the full OpenAlex reference lists: 59,369 papers have an OpenAlex record and 34,540 of them a non-empty reference list, giving 922,768 distinct links. The 327,872 cited papers outside the collection become extra nodes. They are never returned as results, but they connect papers that cite the same outside classic. Nodes are papers, and a directed edge u → v (M_{uv} = 1) means u cites v."));
C.push(P("**Base set.** For the root set R (top 200 BM25F candidates), following Kleinberg (1999), with at most 100 citing papers added per root paper (lowest identifiers first, so the result is deterministic):"));
C.push(EQ("B = R  ∪  { v : u ∈ R, u cites v }  ∪  { u : u cites some r ∈ R }"));
C.push(P("Over the 40 evaluation queries, B averages 2,080 papers and 4,150 links."));
C.push(P("**Relevance weighting.** In classic HITS every citing paper votes with full weight, even if it belongs to another community. Following Bharat and Henzinger (1998), the edge u → v is weighted by the BM25F relevance of the *citing* paper u (outside papers have no reference list, so they never cite and need no weight):"));
C.push(EQ("ω(u) = min( 1,  BM25F(q,u) / max_{r∈R} BM25F(q,r) ),     A_{uv} = ω(u) · M_{uv}   for u, v ∈ B"));
C.push(P("**Iteration.** A good authority is cited by good hubs, and a good hub cites good authorities. Starting from h = 1/√|B|, HITS repeats the two updates below until Σ|Δa| + Σ|Δh| < 10⁻⁸ or 100 iterations. The L2 normalisation keeps the vectors bounded; the iteration is a power method that converges to the principal eigenvectors of AᵀA and AAᵀ (Kleinberg, 1999). On the evaluation queries it took 49 iterations on average."));
C.push(EQ("a ← Aᵀh / ‖Aᵀh‖₂ ,     h ← A a / ‖A a‖₂"));
C.push(TCAP("**Algorithm 1.** Topic-specific relevance-weighted HITS, as implemented in `src/scholar_ir/hits.py`."));
C.push(CODE([
  "input : root set R (top-200 BM25F docs), BM25F scores s(.), citation matrix M",
  "1  B <- R ∪ cited(R) ∪ first-100-citers(r) for each r in R",
  "2  for u in B: w(u) <- min(1, s(u) / max_{r in R} s(r))     # 0 for outside papers",
  "3  A <- diag(w) . M[B, B]",
  "4  h <- 1/sqrt(|B|);  repeat up to 100 times:",
  "5      a <- A^T h / ||A^T h||;   h <- A a / ||A a||;   stop if change < 1e-8",
  "6  pct(d) <- percentile of a(d) among {r in R : a(r) > 0}  (0 if a(d) = 0)",
  "7  return pct (used in ranking), h (displayed as topic hubs)",
]));
C.push(P("**Combining topic and global authority.** In-topic citations take time to accumulate, so HITS alone re-introduces age bias. The final authority blends it with the age-fair global authority:"));
C.push(EQ("g(d) = β · pct( a(d) ) + ( 1 − β ) · A2(d)"));
C.push(P("β and the fixed λ were chosen on the 20 development queries only: β = 0.5 and λ = 0.3 gave the best nDCG@10 (0.730, against 0.699 for BM25F; Figure 6). **S9** uses Eq. 8 in Eq. 3 with λ = 0.3, and **S10** uses the adaptive λ(q). Hub scores are shown to the user as “key papers for this topic” but are not used in ranking. **Empty graphs:** if no paper in B has a reference list, every a(d) is zero and g(d) = 0.5 · A2(d), so the ranking keeps relevance and global authority at half weight; the interface states this."));
C.push(H2("5.5 Field classification (new)"));
C.push(P("Each paper is labelled with its primary arXiv category. The document–term matrix is built from the index postings, so the classifiers use the search engine's own stemmed vocabulary. A stratified split gives 64,000 training and 16,000 test papers."));
//
C.push(P("**Multinomial Naive Bayes** picks the class with the highest posterior, with T_{ct} the count of term t in class c and add-one smoothing. A variant keeps only the k terms with the highest **mutual information** I(U_{t}; C_{c}) with some class (Manning, Raghavan and Schütze, 2008, Eq. 13.17):"));
C.push(EQ("ĉ(d) = argmax_c [ log P(c) + Σ_{t∈d} tf(t,d) · log P(t|c) ],     P(t|c) = (T_ct + 1) / ( Σ_{t'} T_ct' + |V| )"));
C.push(EQ("I(U_t ; C_c) = Σ_{x∈(0,1)} Σ_{y∈(0,1)} P(U_t=x, C_c=y) · log₂ [ P(U_t=x, C_c=y) / ( P(U_t=x) · P(C_c=y) ) ]"));
C.push(P("**Rocchio** represents each class by the centroid of its L2-normalised (1 + log tf)·idf vectors and assigns the nearest centroid by cosine. **kNN** lets the 10 most cosine-similar training papers vote, weighted by similarity. In the interface, Naive Bayes predicts the field of each query and suggests a `cat:` filter. It does not change the ranking, because a wrong hard filter would delete relevant papers and the classifier is weakest on short queries (Section 7.6)."));
C.push(H2("5.6 Query parsing and the LLM layer"));
C.push(P("Users can type our query language directly or a plain-English request. For the latter, Google Gemini (temperature 0, responses cached) translates the request into terms, phrases, years and categories, which are compiled into the same query language, so retrieval and ranking always run in our own engine. An optional answer mode sends only the titles and abstracts of the top five results to Gemini, and our own citation checker compares every answer sentence with the abstract it cites (tf-idf cosine plus idf-weighted term coverage) and flags unsupported sentences."));
C.push(P("**Filter guard (improved).** In the mid-term evaluation, 13 of Gemini's 14 parsing errors were category filters nobody asked for, and words such as *new* became year ranges. Since a filter deletes candidates before ranking, a wrong filter costs recall that no ranking function can recover. The guarded parser keeps the LLM's terms and phrases but takes categories only from fields the user names and years only from an explicit year or span. **Hyphenated words (improved).** The query side counted “LLM-based” twice (llm, llmbased) against one “recommender”, which doubled the weight of one concept and pulled results toward LLM-agent papers; queries now use only the parts, while documents still index both forms."));
C.push(H2("5.7 Implementation and interface"));
C.push(P("The system is about 5,700 lines of Python in 27 modules, using numpy, scipy (sparse matrices), pandas, NLTK (Porter stemmer only), pydantic and ipywidgets; no search library is used. The new code is in `hits.py` (citation graph in CSR form, base set, weighted HITS), `classify.py` (the three classifiers and metrics), `engine.py` (systems S1–S10 and topic authority), `llm_query_parser.py` (filter guard), `ranking.py` (hyphen fix) and `scripts/export_refs.py` (reference lists). The test suite has 50 unit tests: 49 pass and one is skipped when no API key is set. Code coverage was not measured. Every number in this report is read from files under `results/`, regenerated by `python scripts/export_refs.py`, `python -m scholar_ir.experiments all`, `python -m scholar_ir.classify eval` and `python -m scholar_ir.llm_experiments parser`."));
C.push(P("The Jupyter interface has two tabs. **Search papers** takes the full query language, a choice of ranking system and an optional side-by-side view of the old (S4) and new ranking. It reports candidates, latency, λ(q), the predicted field and the size and main hubs of the topic graph; each result shows bars for relevance, topic authority and global authority, and a term-by-term explanation (Figure 3). **Ask a question** shows the parsed query, Gemini's answer sentence by sentence with the checker's verdict, and the retrieved papers."));
C.push(FIG("fig12_ui_search.png", 5.0, "**Figure 3.** Search tab for the machine-translation query with S10 (actual interface output)."));

// ---- 6
C.push(H1("6. Experimental Setup and Evaluation"));
C.push(P("**Dataset.** Primary labels: cs.LG 26,234, cs.CV 23,327, cs.CL 10,016, cs.AI 9,858, cs.SI 5,578, cs.IR 2,766, cs.DB 2,221. Citation counts come from Semantic Scholar where matched and otherwise from OpenAlex (99% of papers covered); reference lists are available for 34,540 papers."));
C.push(P("**Queries.** 40 queries with written relevance descriptions: 20 *foundational* (e.g. “dropout regularization in neural networks”) and 20 *recent* (e.g. “instruction tuning of language models”), split into 20 development queries used for all tuning and 20 test queries used for all reported results, 10 of each type per split. The queries were drafted with an AI assistant and reviewed by the team."));
C.push(P("**Relevance judgments.** The top 15 results of every system and every tuning configuration were pooled and graded 0/1/2 against the query description, without showing which system produced them: 1,440 pairs (411 non-relevant, 468 relevant, 561 highly relevant), 7–45 relevant papers per query (mean 25.7). **All grades were produced by an AI judge (Claude)**, 1,082 at the mid-term and 358 for papers the new systems added. A human re-grading to measure agreement was planned but not completed, so no agreement figure is reported."));
C.push(P("**Systems.** Table 2 lists the ten compared ranking systems. All use the same index and candidate generation, so differences between S3 and S4–S10 come only from the authority signal and its weight."));
C.push(TCAP("**Table 2.** Ranking systems. S4–S10 re-rank the top 200 BM25F candidates with Eq. 3."));
C.push(TABLE([
  ["System", "Relevance", "Authority g(d)", "λ"],
  ["S1 / S2 / S3", "tf-idf / BM25 / BM25F", "—", "—"],
  ["S4", "BM25F", "log raw citations", "0.3"],
  ["S6", "BM25F", "A2 year-cohort percentile", "0.3"],
  ["S7 (mid-term)", "BM25F", "A2", "λ(q)"],
  ["S8", "BM25F", "PageRank, cohort percentile", "λ(q)"],
  ["S9 (new)", "BM25F", "0.5 topic HITS + 0.5 A2", "0.3"],
  ["S10 (new)", "BM25F", "0.5 topic HITS + 0.5 A2", "λ(q)"],
], [2000, 2500, 3326, 1200], { size: 18 }));
C.push(P("**Metrics.** P@k, R@k against the pooled relevant set, MAP, nDCG@10 with gain 2^g^ − 1 and MRR. For classification: accuracy, macro-averaged precision, recall and F1. **Significance.** Paired bootstrap (10,000 resamples, two-sided, 95% interval) and Wilcoxon signed-rank test; a difference is called significant only at p < 0.05, and both tests are reported."));
C.push(P("**Citation recommendation.** For 300 random papers with at least five references inside the corpus, the query is the paper's 30 highest tf-idf terms, the relevant set is those references, and candidates are limited to papers no newer than the query paper. For graph-based systems the query paper is **removed from the citation graph**, otherwise its own references would vote for the answers. Global citation counts (S4–S8) still include the query paper's citation, which slightly favours those baselines."));

// ---- 7
C.push(H1("7. Results and Discussion"));
C.push(P("The results follow the order of the retrieval pipeline: the index and query processing (7.1), ranking effectiveness on the graded queries (7.2), the citation-recommendation test (7.3) and the ablations (7.4), followed by examples (7.5), text classification (7.6) and the LLM parser (7.7). For each table we first say how to read it, then what it shows."));

C.push(H2("7.1 Indexing and query processing"));
C.push(TCAP("**Table 3.** Inverted index statistics by zone (80,000 documents)."));
C.push(TABLE([
  ["Zone", "Terms", "Postings", "Positions", "Avg. length", "Doc-ID lists, raw", "Gap + VB code", "Ratio"],
  ["Title", "38,355", "635,011", "645,653", "8.1", "2.54 MB", "0.96 MB", "2.6×"],
  ["Abstract", "127,263", "5,960,134", "8,534,933", "106.7", "23.84 MB", "7.01 MB", "3.4×"],
  ["All (title + abstract)", "132,278", "6,061,229", "9,180,586", "114.8", "24.24 MB", "7.14 MB", "3.4×"],
], [1826, 950, 1150, 1150, 950, 1100, 1100, 800], { size: 17 }));
C.push(P("**How to read Table 3.** *Terms* is the vocabulary of a zone after stemming and stop-word removal, *postings* the number of (term, document) entries, *positions* the stored word offsets that phrase and NEAR queries need, and *average length* the mean number of indexed tokens per document. The last three columns compare storing document IDs as raw 32-bit integers with storing the gaps between them in variable-byte code (Manning, Raghavan and Schütze, 2008, Ch. 5)."));
C.push(P("**What it shows.** The abstract zone holds 98% of the postings because abstracts are thirteen times longer than titles. Gap encoding would shrink the document-ID lists 3.4 times; titles compress less (2.6×) because their short postings lists have large gaps. We implemented and measured the encoding but keep raw IDs at query time, since the whole index (297 MB) fits in memory. Preprocessing also shrinks the index: Porter stemming reduces the vocabulary by 17.9% (161,166 → 132,278 terms) and stop-word removal cuts the indexed tokens by 38.3% (14.9 M → 9.2 M) with almost no effect on the vocabulary."));
C.push(TCAP("**Table 4.** Query-processing efficiency on the 40 evaluation queries."));
C.push(TABLE([
  ["Method", "Work per query", "Time (ms)", "Recall@10 vs exact top 10"],
  ["Boolean AND, terms in query order", "45,052 postings", "4.24", "exact"],
  ["Boolean AND, rarest term first", "44,082 postings", "3.99", "exact"],
  ["Boolean AND, rarest first + skip pointers", "19,154 postings", "2.50", "exact"],
  ["Ranked, full postings (exact)", "33,199 candidates", "7.43", "1.000"],
  ["Ranked, champion lists (r = 500)", "1,885 candidates", "1.59", "0.918"],
  ["Ranked, index elimination", "9,416 candidates", "2.83", "0.842"],
], [3626, 1900, 1300, 2200], { size: 18 }));
C.push(P("**How to read Table 4.** The upper rows are Boolean AND queries; *work* is the number of postings visited during intersection and time is the mean. The lower rows are ranked retrieval; *work* is the number of candidate documents scored, time is the median, and the last column is the share of the exact top 10 that the faster method still returns."));
C.push(P("**What it shows.** Intersecting from the rarest term saves little on its own, but combined with skip pointers it visits 57% fewer postings and is 41% faster. For ranked retrieval, champion lists (the 500 highest-weighted documents per term) score 18 times fewer candidates and are 4.7 times faster while keeping 92% of the exact top 10; index elimination keeps less (84%) at a higher cost. Because champion lists cost 0.026 nDCG@10 in our ablation (0.829 vs 0.855), the evaluated systems use exact scoring. Full ranked queries take 7.1 ms with BM25F, 8.6 ms with S7 and 13.0 ms with S9 (95th percentile 19.0 ms); the extra 6 ms of S9 is building the base set and running HITS."));

C.push(H2("7.2 Ranking effectiveness on the graded test queries"));
C.push(TCAP("**Table 5.** Ranking effectiveness on the 20 test queries (best value per column in bold)."));
C.push(TABLE([
  ["System", "P@5", "P@10", "R@10", "R@20", "MAP", "nDCG@10", "MRR"],
  ["S1 tf-idf", "0.820", "0.785", "0.358", "0.583", "0.665", "0.665", "0.867"],
  ["S2 BM25", "0.880", "0.890", "0.421", "0.660", "0.785", "0.783", "0.967"],
  ["S3 BM25F", "0.930", "0.910", "0.436", "0.694", "0.844", "**0.855**", "**1.000**"],
  ["S4 + raw citations", "0.950", "0.910", "0.433", "**0.729**", "**0.867**", "0.848", "**1.000**"],
  ["S6 + year cohort", "**0.960**", "0.910", "0.438", "0.716", "0.850", "0.852", "**1.000**"],
  ["S7 mid-term", "**0.960**", "0.910", "0.438", "0.716", "0.854", "0.851", "**1.000**"],
  ["S8 PageRank", "0.920", "0.890", "0.423", "0.704", "0.840", "0.831", "0.975"],
  ["S9 topic HITS", "0.950", "0.915", "0.439", "0.721", "0.864", "0.851", "**1.000**"],
  ["S10 topic HITS + λ(q)", "0.950", "**0.920**", "**0.442**", "0.719", "0.864", "0.853", "**1.000**"],
], [2326, 900, 900, 900, 900, 900, 1100, 1100], { hl: [8, 9], size: 18 }));
C.push(P("**How to read Table 5.** Each value is a mean over the 20 test queries. *P@k* is the share of the top k results that are relevant (grade 1 or 2). *R@k* is the share of all known relevant papers for the query that appear in the top k; it stays low (about 0.44 at k = 10) because queries have 25.7 relevant papers on average, so even a perfect top 10 cannot reach 1.0. *MAP* averages precision at the rank of every relevant paper, so it rewards placing all relevant papers high, not only the first few. *nDCG@10* uses the graded labels, so a highly relevant paper counts more than a relevant one, discounted by its rank. *MRR* is the reciprocal rank of the first relevant result; 1.000 means a relevant paper is first for every query."));
C.push(P("**What it shows.** The rows fall into three groups. (i) *Text models.* Moving from tf-idf (S1) to BM25 (S2) adds 0.118 nDCG@10, through term-frequency saturation and length normalisation; separating titles from abstracts (BM25F, S3) adds another 0.072. This 0.190 gain over tf-idf is the largest effect in the table. (ii) *Authority re-ranking.* S4–S10 change the order of BM25F's candidates but not the candidates themselves, so P@10 hardly moves (0.890–0.920). They differ in MAP and R@20, where raw citations (S4) and topic HITS (S9, S10) reach 0.864–0.867 MAP against 0.844 for BM25F. (iii) *Global PageRank* (S8) is the only authority signal that lowers nDCG@10, MAP and MRR relative to BM25F, because the in-corpus citation graph it runs on is sparse. Figure 4 splits nDCG@10 and MAP by query type: S10 is the best system on recent queries (0.870 nDCG@10), while BM25F stays best on foundational ones (0.849)."));
C.push(FIG("fig6_judged_results.png", 5.1, "**Figure 4.** nDCG@10 and MAP on the test queries, by query type."));
C.push(TCAP("**Table 6.** Paired significance tests on the 20 test queries (difference = A − B)."));
C.push(TABLE([
  ["A vs B", "Metric", "Difference", "95% CI", "p (bootstrap)", "p (Wilcoxon)"],
  ["S3 vs S1", "nDCG@10", "+0.190", "[+0.121, +0.260]", "< 0.001", "< 0.001"],
  ["S9 vs S3", "nDCG@10", "−0.005", "[−0.031, +0.023]", "0.74", "0.83"],
  ["S9 vs S3", "MAP", "+0.020", "[+0.004, +0.037]", "0.017", "0.114"],
  ["S10 vs S3", "MAP", "+0.020", "[+0.007, +0.034]", "0.003", "0.022"],
  ["S9 vs S4", "nDCG@10", "+0.002", "[−0.013, +0.018]", "0.74", "0.59"],
  ["S9 vs S4", "MAP", "−0.003", "[−0.013, +0.007]", "0.51", "0.38"],
  ["S10 vs S7", "MAP", "+0.010", "[−0.008, +0.025]", "0.21", "0.036"],
], [1500, 1200, 1300, 1926, 1550, 1550], { size: 18 }));
C.push(P("**How to read Table 6.** Each row compares two systems on the same 20 queries. *Difference* is the mean per-query difference; the *95% CI* is its bootstrap confidence interval, and an interval that contains zero means the data are consistent with no difference. The two p-values come from different tests: the bootstrap resamples queries, and the Wilcoxon test ranks the per-query differences. We call a difference significant only when p < 0.05."));
C.push(P("**What it shows.** BM25F's gain over tf-idf is certain under both tests. On nDCG@10, S9 is indistinguishable from BM25F (p = 0.74 and 0.83) and from S4, so the new systems are *comparable*, not better, on that metric. On MAP, S10's +0.020 over BM25F is significant under both tests, while S9's identical gain is significant only under the bootstrap and should be read as suggestive. S10's MAP gain over the mid-term S7 passes only the Wilcoxon test. With 20 queries, differences below about 0.03 nDCG@10 cannot be detected."));

C.push(H2("7.3 Citation recommendation"));
C.push(TCAP("**Table 7.** Citation recommendation over 300 held-out papers."));
C.push(TABLE([
  ["System", "R@10", "R@20", "MAP", "MRR"],
  ["S1 tf-idf", "0.102", "0.145", "0.055", "0.231"],
  ["S3 BM25F", "0.107", "0.149", "0.058", "0.245"],
  ["S4 + raw citations", "0.174", "0.229", "0.105", "0.380"],
  ["S6 + year cohort", "0.160", "0.219", "0.090", "0.330"],
  ["S7 mid-term", "0.119", "0.172", "0.068", "0.275"],
  ["S8 PageRank", "0.128", "0.183", "0.076", "0.297"],
  ["S9 topic HITS", "**0.227**", "**0.277**", "**0.143**", "**0.476**"],
  ["S10 topic HITS + λ(q)", "0.137", "0.192", "0.080", "0.315"],
], [3226, 1450, 1450, 1450, 1450], { hl: [7], size: 18 }));
C.push(P("**How to read Table 7.** Each held-out paper is turned into a query from its own text, and the relevant set is the papers it actually cites in our collection. *R@20* is therefore the share of a paper's references found in the top 20, and *MRR* the reciprocal rank of the first reference found (roughly, 0.476 corresponds to the first reference at about rank 2 and 0.380 to about rank 2.6). Because the ground truth comes from the authors, not the AI judge, this is the most objective test in the report."));
C.push(FIG("fig7_citerec.png", 4.4, "**Figure 5.** Recall@20 in citation recommendation."));
C.push(P("**What it shows.** Text alone finds few references: tf-idf and BM25F reach only 0.145 and 0.149 Recall@20, because a paper's references are rarely the papers whose wording is most similar to its own. Any citation signal helps, and raw counts (S4) lift Recall@20 to 0.229 because references are often well-cited classics. Topic HITS (S9) is best on every metric (Figure 5), with Recall@20 0.277 (+0.048 over S4, 95% CI [+0.035, +0.061]; +0.128 over BM25F) and MRR 0.476; all differences have p < 0.001 under both tests. This is the behaviour topic authority was designed for: it rewards papers that the topic's own literature cites. S10 gains much less than S9 because these long queries contain no foundational cue words, so λ(q) gives authority little weight; S7 falls below S4 for the same reason. The adaptive weight helps on the graded queries (Table 6) and hurts here, so neither weighting is best in every setting."));

C.push(H2("7.4 Ablation study"));
C.push(TCAP("**Table 8.** Ablations on the 20 test queries (nDCG@10)."));
C.push(TABLE([
  ["Variant", "Foundational", "Recent", "All"],
  ["S9: weighted HITS blended with A2 (β = 0.5)", "0.845", "0.856", "0.851"],
  ["Classic (unweighted) HITS, blended", "0.840", "0.856", "0.848"],
  ["Weighted HITS only (β = 1)", "0.844", "0.823", "0.834"],
  ["Classic HITS only (β = 1)", "0.837", "0.825", "0.831"],
  ["S3 BM25F", "0.849", "0.861", "0.855"],
  ["S3 without stemming", "0.769", "0.647", "0.708"],
  ["S3 without stop-word removal", "0.837", "0.856", "0.846"],
  ["S3 with champion lists (r = 500)", "0.830", "0.828", "0.829"],
  ["Sum of per-zone BM25 instead of BM25F", "0.772", "0.769", "0.771"],
], [5026, 1330, 1330, 1340], { size: 18 }));
C.push(P("**How to read Table 8.** Each row removes or replaces one component and reports nDCG@10, split by query type. The upper block varies the HITS design; the lower block varies the text-retrieval components."));
C.push(P("**What it shows.** *Blending is essential:* without the global half (β = 1), recent-query nDCG@10 drops from 0.856 to 0.823, because 2022–2023 papers have no in-topic citations yet and pure HITS gives them zero authority. *Relevance weighting helps slightly:* weighted HITS beats Kleinberg's original algorithm in both settings (0.851 vs 0.848 and 0.834 vs 0.831), consistent with Bharat and Henzinger (1998), but the margin is small. Among the text components, *stemming* matters most (−0.147 without it and −0.214 on recent queries, probably because their terms occur in many inflected forms), while stop-word removal costs little to remove (−0.009) because idf already gives stop words almost no weight. Combining zones inside BM25F beats adding separate per-zone BM25 scores by 0.084, which reproduces the saturation argument of Robertson, Zaragoza and Taylor (2004). Figure 6 shows the development-set tuning: pure HITS degrades as λ grows, while the blend stays above BM25F up to λ = 0.5."));
C.push(FIG("fig5_hits_grid.png", 3.4, "**Figure 6.** Development-set tuning of β and λ for S9."));

C.push(H2("7.5 Qualitative examples"));
C.push(P("**Machine translation.** In Figure 7, BART is first under S4 and leaves the top six under S10. *Sequence Transduction with Recurrent Neural Networks* (2012) rises from fifth to first with topic authority 0.91, and a 2015 sequence-to-sequence paper enters. This is one example of how the ranking changes, not proof that it improves: on this query S4 scores 0.80 nDCG@10, S9 0.81 and BM25F alone 0.96. The strongest topic authority, *Sequence to Sequence Learning with Neural Networks* (2014), is not in S10's top six because its BM25F relevance is lower than that of papers whose titles repeat the query."));
C.push(FIG("fig11_ui_compare_crop.png", 4.6, "**Figure 7.** Old (S4) vs new (S10) ranking with rank movements (actual interface output)."));
C.push(P("**Hyphenated query.** After the fix, the top five for “LLM-based recommender systems” are all LLM-based recommender papers (Chat-REC, GenRec, RecMind and others); before it, LLM-agent papers dominated. **Empty graph.** For `\"dense retrieval\" year:>=2021`, the 29 candidates have no reference lists in our data, so the base set has no links; the ranking falls back as described in Section 5.4 and the interface says so. The Naive Bayes prediction for this short query, cs.CV (66%), is wrong."));
//

C.push(H2("7.6 Text classification and comparison with existing work"));
C.push(TCAP("**Table 9.** Field classification on 16,000 test papers (seven classes)."));
C.push(TABLE([
  ["Classifier", "Accuracy", "Macro P", "Macro R", "Macro F1"],
  ["Majority class", "0.328", "0.047", "0.143", "0.071"],
  ["Naive Bayes, all terms", "**0.786**", "**0.779**", "0.702", "0.728"],
  ["Naive Bayes, top 1,000 MI terms", "0.769", "0.705", "0.772", "0.732"],
  ["Naive Bayes, top 5,000 MI terms", "0.776", "0.713", "**0.780**", "0.741"],
  ["Naive Bayes, top 20,000 MI terms", "0.783", "0.729", "0.772", "**0.749**"],
  ["Rocchio", "0.762", "0.703", "0.767", "0.730"],
  ["kNN (k = 10)", "0.764", "0.750", "0.690", "0.714"],
], [3626, 1350, 1350, 1350, 1350], { size: 18 }));
C.push(P("**How to read Table 9.** *Accuracy* is the share of test papers assigned their correct primary field. Precision, recall and F1 are computed per class and then averaged with equal weight (*macro* averaging), so the small classes cs.IR (2,766 papers) and cs.DB (2,221) count as much as cs.LG (26,234). The majority-class row always predicts cs.LG and shows what accuracy is possible with no learning."));
C.push(P("**What it shows.** All three methods are far above the baseline. Naive Bayes on all terms has the highest accuracy (78.6%) and is the cheapest: prediction is one sparse matrix product. Mutual-information feature selection costs a little accuracy but raises macro recall from 0.70 to 0.77–0.78, because the small classes are no longer outvoted by frequent terms of the large ones; with 20,000 terms it has the best macro F1 (0.749). Rocchio's centroids are almost as good. kNN is close in accuracy but several hundred times slower (67 s vs 0.1 s for the test set) and has the lowest recall. Our Naive Bayes predicts the same class as scikit-learn's MultinomialNB for all 16,000 papers. The confusion matrix (Figure 8) shows that errors follow genuine overlaps between fields: 18% of cs.AI papers are predicted as cs.LG. Counting a prediction as correct when it matches *any* of a paper's listed categories raises accuracy to 85.4%."));
C.push(FIG("fig9_confusion.png", 3.0, "**Figure 8.** Row-normalised confusion matrix of Naive Bayes with 20,000 MI terms."));
C.push(P("**Comparison with Rahman et al. (2025).** They report 0.65 accuracy and 0.62 F-score for TF-IDF Naive Bayes on arXiv abstracts, and 0.69 accuracy for logistic regression; our Naive Bayes reaches 0.786 accuracy and 0.728 macro F1. The comparison is indicative only: they predict broad archives (physics, mathematics, computer science and others) in a multi-label setting on about 1.2 million records, while we predict the primary label among seven overlapping computer-science fields. Their paper is a non-peer-reviewed preprint."));

C.push(H2("7.7 LLM query parsing"));
C.push(P("On 25 natural-language requests with gold parses, Gemini extracts search terms better than our rule-based parser (term F1 0.975 vs 0.940) but got only 44% of requests fully right on filters, because it added categories and year ranges nobody asked for. With the filter guard, filters are exactly right for 96% of requests while the better term extraction is kept. The gold parses were drafted by the same AI assistant that wrote the rule cues the guard relies on, which favours the rules."));

C.push(H2("7.8 What the results establish"));
C.push(P("**Established by the experiments.** Topic HITS gives the best citation recommendation of all ten systems (p < 0.001). On the graded queries S10 improves MAP over BM25F under both tests. Field-weighted BM25F is the largest single gain over tf-idf, and stemming is the most important preprocessing step. HITS without the global blend hurts recent queries. Skip pointers more than halve the postings visited by AND queries. Naive Bayes reaches 78.6% accuracy on seven fields."));
C.push(P("**Interpretation.** Topic authority helps most when a task rewards finding what a topic relies on, and it adds little to the top ten of short queries where BM25F is already strong. The machine-translation example shows the intended mechanism (BART demoted, topic papers promoted) but is a single case."));
C.push(P("**Not established.** An nDCG@10 improvement over BM25F or S4; S9's MAP gain under the Wilcoxon test; agreement between the AI grades and human judgments; the accuracy of the citation checker and the usefulness of the displayed hubs, neither of which was evaluated."));

// ---- 8
C.push(H1("8. Conclusions, Limitations and Future Scope"));
C.push(P("Topic-specific, relevance-weighted HITS turns citation data into a signal about the query's topic rather than about fame. It is clearly the best system where the task measures topic importance directly (citation recommendation, Recall@20 0.277 vs 0.229), and comparable to BM25F on the graded queries with a modest MAP gain. It works only alongside the age-fair global authority. Naive Bayes on our own index classifies abstracts into seven fields with 78.6% accuracy, and taking filters from the user's words instead of the LLM's guesses fixes the parser's main error."));
C.push(P("**Limitations.** (1) *Reference coverage:* reference lists exist for 59.9% of 2010–2020 papers but 46% of 2021 and under 1% of 2022–2023 papers, so for recent topics the graph is thin or empty and the system falls back on global authority. This is a limit of our data snapshot, not of the method. (2) *AI judgments:* all ranking grades came from an AI judge, and the same assistant drafted the queries; citation recommendation is unaffected. (3) *Small query set:* 20 test queries detect only differences of about 0.03 nDCG@10 or more. (4) *Short queries:* the field prediction is unreliable on two- or three-word queries and is shown only as a hint. (5) *Retrieval bound:* HITS re-ranks the BM25F root set and cannot recover papers lexical retrieval misses; “tutorial on graph neural networks” still returns tutorials on other topics."));
//
C.push(P("**Future scope** (proposals, not implemented): fetch reference lists for 2021–2023 papers from Semantic Scholar; choose β per query from the density of the topic graph; treat document-type words such as *tutorial* and *survey* as intent only; evaluate the hub scores for survey-seeking queries; and validate the relevance grades with human assessors."));

// ---- References
C.push(H1("Declaration of AI Use"));
C.push(P("Claude (Anthropic) was used as a coding assistant for most of the code, drafted the evaluation queries and this report from the team's results, and produced all relevance grades (Section 6). Google Gemini is used inside the system to parse questions and to write cited answers from retrieved abstracts. The team directed the design, ran the pipeline, checked the outputs and is responsible for the submission."));
C.push(H1("References"));
[
  "Bharat, K. and Henzinger, M.R., 1998, August. Improved algorithms for topic distillation in a hyperlinked environment. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 104–111).",
  "Kleinberg, J.M., 1999. Authoritative sources in a hyperlinked environment. Journal of the ACM, 46(5), pp. 604–632.",
  "Manning, C.D., Raghavan, P. and Schütze, H., 2008. Introduction to Information Retrieval. Cambridge University Press.",
  "McCallum, A. and Nigam, K., 1998, July. A comparison of event models for naive Bayes text classification. In AAAI-98 Workshop on Learning for Text Categorization (Vol. 752, pp. 41–48).",
  "Page, L., Brin, S., Motwani, R. and Winograd, T., 1999. The PageRank citation ranking: Bringing order to the web. Technical Report, Stanford InfoLab.",
  "Rahman, S., Shanto, H.K., Koana, U.A. and Danish, S.M., 2025. Automated research article classification and recommendation using NLP and ML. arXiv preprint arXiv:2510.05495.",
  "Robertson, S. and Zaragoza, H., 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), pp. 333–389.",
  "Robertson, S., Zaragoza, H. and Taylor, M., 2004, November. Simple BM25 extension to multiple weighted fields. In Proceedings of the 13th ACM International Conference on Information and Knowledge Management (pp. 42–49).",
].forEach((r) => C.push(new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(r, { size: 19 }), spacing: { after: 40, line: 240 }, alignment: AlignmentType.LEFT })));


// ================================================================ DOCUMENT
const header = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "B8BEC6", space: 4 } },
  children: [new TextRun({ text: "Scholar-IR · CSD358 Information Retrieval · Final Project Report", font: FONT, size: 17, color: "555555" })] })] });
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] });
const numberingCfg = [
  { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 560 } } } }] },
  ...numberingConfigs.map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
];
const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1300, left: 1440, header: 600, footer: 600 } };
const doc = new Document({
  creator: "Aryan, Atul and Rishith", title: "Scholar-IR: Topic-Specific Citation Authority for Academic Paper Search",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 200, after: 90 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: numberingCfg },
  sections: [
    { properties: { page }, children: T },
    { properties: { page }, headers: { default: header }, footers: { default: footer }, children: C },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
