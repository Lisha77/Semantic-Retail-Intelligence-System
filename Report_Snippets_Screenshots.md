# Report Code Snippets & Screenshot Guide

This file tells you exactly which code snippets and screenshots to embed in your
printed/PDF academic report. It is a practical companion to Academic_Report.md.

---

## 9.1 Code Snippets to Embed (Priority Order)

| # | Snippet | Where it goes in report | Why it matters |
|---|---|---|---|
| C1 | `discover_schema()` | Section 7.2 (Stage 0) | Proves zero hard-coding: no column name is written literally; schema is inferred from dtypes + cardinality |
| C2 | `_tokenize_label()` + `build_search_index()` | Section 7.3 (Stage 1) | Proves the hybrid matching engine: exact labels + char n-gram TF-IDF cosine similarity |
| C3 | `interpret_segment()` | Section 7.4 (Stage 2) | Proves the "Logical Query Conversion" — the most important function in the whole pipeline |
| C4 | `compute_segment()` | Section 7.5 (Stage 3) | Proves the actual pandas execution logic (groupby/agg/filters/date filter) |
| C5 | `generate_pandas_code()` | Section 7.6 (Stage 4) | Proves the "Code" panel on screen — it shows the full runnable query, not just the final line |
| C6 | `render_proof_and_chart()` (LaTeX block) | Section 7.6 (Stage 4) | Proves the "Mathematical Proof" — `\Sigma x`, `\bar{x} = \Sigma x / n` |
| C7 | `render_single_query()` | Section 7.7 (Stage 5) | Proves the full UI rendering pipeline — trace + logic + proof + chart + conclusion |
| C8 | `split_comparison_segments()` + comparison render | Section 4.7 (System Design) | Proves "X versus Y" comparison mode |

### How to embed
- Put **C1** in Section 7.2, **C2** in Section 7.3, **C3** in Section 7.4,
  **C4** in Section 7.5, **C5** in Section 7.6, **C6** in Section 7.6,
  **C7** in Section 7.7, **C8** in Section 4.7.
- Each snippet should be wrapped in a ` ```python ` code block in the report.
- Add a 2-3 line caption under each snippet explaining what it does and why it
  proves a specific design claim.

---

## 9.2 Screenshots to Capture (5 panels per query)

For each representative query in the "Try an example question" dropdown, take
screenshots of **7 panels**. Recommended query matrix:

| Query | Metric | Aggregation | Chart | Panel 4 proof |
|---|---|---|---|---|
| `total sales by product line` | Total | sum | Bar | Sigma x expansion |
| `average rating by branch` | Rating | mean | Bar | x-bar = Sigma x / n |
| `how many transactions in mandalay` | (none) | count (fallback) | Bar | Row-count proof |
| `how many different payment methods are used in yangon` | Payment | nunique | Pie | Distinct-count proof |
| `sales of yangon versus mandalay` | Total | sum | Bar | Comparison-delta proof |
| `relationship between rating and gross income` | Rating, Gross income | Pearson r | Scatter | df.corr() proof |
| `share of sales by payment method` | Total | sum | Pie | Share-of-total proof |
| `sales trend by hour` | Total | sum | Line | Time-aggregate proof |

### Panel layout per query

| Panel | Screenshot |
|---|---|
| **S1** — Top-level landing | Header, "Detected metrics N x dimensions N" caption, query box pre-filled |
| **S2** — Text Mining Trace | Evidence table: Match type / Matched against / Similarity score |
| **S3** — Logical Query Conversion | Logic slot vs Resolved value table |
| **S4** — Analysis & Proof | Runnable pandas code box + LaTeX proof + interactive Plotly chart |
| **S5** — Interpretation & Conclusion | Narrative paragraph + any extra breakdown charts |
| **S6** — Sidebar (stage 0 proof) | Data source radio + Detected filters + WordNet toggle + schema caption |
| **S7** — Comparison mode | "sales of yangon versus mandalay": two side-by-side cards + % delta conclusion |

### How to organize the screenshots in the report
1. Put **S1, S2, S3** together as "Figure 1: Query pipeline — from input to logic."
2. Put **S4 + S5** together as "Figure 2: From pandas code to mathematical proof to narrative."
3. Put **S6** together as "Figure 3: Global controls and schema discovery (Stage 0)."
4. Put **S7** as "Figure 4: Head-to-head comparison with percentage delta."

---

## 9.3 LaTeX Proof Examples (embed as "Figures")

### Example A — Mean proof ("average rating by branch")
```latex
\Sigma x = 6.8 + 5.4 + ... = 1274.6
\bar{x} = \frac{\Sigma x}{n} = \frac{1274.6}{238} = 5.36
```

### Example B — Sum proof ("total sales by product line")
```latex
\Sigma x = 386.6 + 731.1 + ... + 126.4 = 245,012.78
```

### Example C — r proof ("relationship between rating and gross income")
```latex
r = \frac{\Sigma (x - \bar{x})(y - \bar{y})}{\sqrt{\Sigma (x - \bar{x})^2 \cdot \Sigma (y - \bar{y})^2}} = 0.742
```

---

## 9.4 Comparative Snippet (how app_1.py/app_2.py differ)

The report should mention that `app_1.py` and `app_2.py` hard-coded
`supermarket_sales.csv` column names (e.g. `df["Total"]`, `df["City"]`),
whereas `semantic_retail_dashboard.py` uses `discover_schema()` and
`interpret_segment()` to infer everything at runtime. Include a short
code-side-by-side comparison like this:

```python
# app_1.py (hard-coded)
total_sales = int(df_selection["Total"].sum())
average_rating = round(df_selection["Rating"].mean(), 1)

# semantic_retail_dashboard.py (runtime)
metric = metric_matches[0][0] if metric_matches else None   # e.g. "Total"
aggregation = 'sum'
result = df_filtered.groupby(dim)[metric].agg(aggregation).sort_values(ascending=False)
```

---

## 9.5 What the Evaluator will Look For

- ✅ A complete 17-section academic report (Abstract → Bibliography) that maps
  1:1 to the supplied structure.
- ✅ Code snippets that are `python` blocks, each annotated with a caption.
- ✅ Screenshots showing the **7-panel** rendering pipeline for ≥ 3 distinct queries.
- ✅ At least 3 LaTeX proof examples (sum, mean, correlation/Scatter).
- ✅ Comparison table of `app_1.py`/`app_2.py` (hard-coded) vs
  `semantic_retail_dashboard.py` (runtime schema discovery).
- ✅ Clear demonstration that the report file lives at the project root alongside
  the code and the CSV dataset.
