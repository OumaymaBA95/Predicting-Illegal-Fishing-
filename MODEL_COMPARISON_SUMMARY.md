# Model Comparison Summary

## PU Ranking Comparison (Stricter Negatives)

| Metric | 2013-2024 (`vessel_features_2013-2024.csv`) | All Years (`vessel_features_all_years.csv`) |
|---|---:|---:|
| Rows | 192,092 | 192,584 |
| Known IUU labels | 11 | 11 |
| PR-AUC (test) | 0.5004 | 0.5016 |
| ROC-AUC (test) | 0.8557 | 0.9584 |
| MRR (test) | 0.5002 | 0.5008 |
| Top-25 IUU (full) | 2 | 2 |
| Top-100 IUU (full) | 3 | 3 |
| Top-500 IUU (full) | 6 | 9 |
| Top-1000 IUU (full) | 7 | 10 |

## Short Summary
Using the rebuilt all-years dataset improves recall at larger K without hurting top‑25 or top‑100: Top‑500 rises from 6 to 9 known IUU vessels and Top‑1000 rises from 7 to 10. Test‑split PR‑AUC and MRR are comparable, while ROC‑AUC improves noticeably, so the all‑years features are the stronger baseline for reporting results.
