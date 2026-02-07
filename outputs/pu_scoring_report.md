# PU Scoring Report

Dataset: `vessel_features_all_years.csv`
Rows: 192584
Known IUU labels: 11
Training mode: PU (positives + reliable negatives)

## Top-K Ranking (Full Dataset)
| K | TP | Precision | Recall | Lift |
|---:|---:|---:|---:|---:|
| 10 | 1 | 0.1000 | 0.0909 | 1750.76 |
| 25 | 2 | 0.0800 | 0.1818 | 1400.61 |
| 50 | 3 | 0.0600 | 0.2727 | 1050.46 |
| 100 | 3 | 0.0300 | 0.2727 | 525.23 |
| 250 | 6 | 0.0240 | 0.5455 | 420.18 |
| 500 | 9 | 0.0180 | 0.8182 | 315.14 |
| 1000 | 10 | 0.0100 | 0.9091 | 175.08 |

## Notes
- Risk scores are relative rankings, not probabilities.
- PU training treats unknowns as unlabeled, not true negatives.