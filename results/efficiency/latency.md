| system | label | mean_ms | median_ms | p95_ms |
|---|---|---|---|---|
| RAND | random order of candidates | 7.390 | 7.810 | 12.970 |
| S1 | tf-idf lnc.ltc (flat) | 6.530 | 6.630 | 11.410 |
| S2 | BM25 (flat) | 7.390 | 7.760 | 12.620 |
| S3 | BM25F (zones) | 7.050 | 7.070 | 12.360 |
| S3L | linear zone BM25 | 7.270 | 7.260 | 12.790 |
| S4 | BM25F + raw citations | 10.080 | 9.300 | 18.590 |
| S5 | BM25F + per-year citations | 10.200 | 10.880 | 18.000 |
| S6 | BM25F + cohort percentile | 9.350 | 9.580 | 16.980 |
| S7 | BM25F + cohort + adaptive λ | 8.620 | 8.410 | 14.820 |
| S8 | S7 with PageRank authority | 8.290 | 8.090 | 14.490 |
| S9 | BM25F + topic authority (HITS) | 12.990 | 13.390 | 18.980 |
| S10 | BM25F + topic HITS + adaptive λ | 12.380 | 12.830 | 18.010 |
