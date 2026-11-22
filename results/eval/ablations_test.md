| variant | nDCG@10_foundational | nDCG@10_recent | nDCG@10_all | P@10_all | median_year@10 |
|---|---|---|---|---|---|
| S1 tf-idf lnc.ltc (flat) | 0.659 | 0.671 | 0.665 | 0.785 | 2019.750 |
| S2 BM25 (flat) | 0.781 | 0.784 | 0.783 | 0.890 | 2020.000 |
| S3 BM25F (zones) | 0.849 | 0.861 | 0.855 | 0.910 | 2019.500 |
| S3L linear zone BM25 | 0.772 | 0.769 | 0.771 | 0.830 | 2020.000 |
| S4 BM25F + raw citations | 0.842 | 0.854 | 0.848 | 0.910 | 2019.000 |
| S5 BM25F + per-year citations | 0.838 | 0.861 | 0.850 | 0.910 | 2019.250 |
| S6 BM25F + cohort percentile | 0.836 | 0.868 | 0.852 | 0.910 | 2018.750 |
| S7 BM25F + cohort + adaptive λ | 0.837 | 0.865 | 0.851 | 0.910 | 2018.500 |
| S8 S7 with PageRank authority | 0.814 | 0.848 | 0.831 | 0.890 | 2019.750 |
| S9 BM25F + topic authority (HITS) | 0.845 | 0.856 | 0.851 | 0.915 | 2018.250 |
| S10 BM25F + topic HITS + adaptive λ | 0.837 | 0.870 | 0.853 | 0.920 | 2018.250 |
| S9 classic (unweighted) HITS | 0.840 | 0.856 | 0.848 | 0.905 | 2018.500 |
| S9 HITS only (no blend with A2) | 0.844 | 0.823 | 0.834 | 0.905 | 2018.000 |
| S9 classic HITS only | 0.837 | 0.825 | 0.831 | 0.885 | 2018.500 |
| S6 multiplicative net score | 0.844 | 0.865 | 0.854 | 0.910 | 2019.250 |
| S7 multiplicative net score | 0.845 | 0.868 | 0.857 | 0.915 | 2019.250 |
| BM25F + PageRank-cohort, fixed λ | 0.807 | 0.849 | 0.828 | 0.885 | 2020.000 |
| S3 champion lists (r=500) | 0.830 | 0.828 | 0.829 | 0.885 | 2019.500 |
| S3 index elimination | 0.735 | 0.813 | 0.774 | 0.820 | 2019.500 |
| S3 stem=False stopwords=True | 0.769 | 0.647 | 0.708 | 0.740 | 2019.000 |
| S3 stem=True stopwords=False | 0.837 | 0.856 | 0.846 | 0.915 | 2019.500 |
| S3 stem=False stopwords=False | 0.772 | 0.633 | 0.703 | 0.745 | 2019.000 |
