# Model Evaluation Results

**Date**: February 3, 2026  
**Dataset**: `vessel_features_2013-2024.csv` (192,092 vessels, 11 IUU positives)

---

## 🔴 Critical Constraints (Small IUU List)

### 1. **Extreme Class Imbalance**

- **Example ratio**: ~17,462:1 (11 positives out of 192,092 vessels)
- The model will always struggle to predict all positives with this few labels
- Treat this as **Positive + Unlabeled (PU)**, not standard supervised learning

### 2. **Labels are Incomplete**

- The IUU list is a **seed list**, not full ground truth
- Many true IUU vessels are likely labeled as 0 (unknown)
- We should prioritize **ranking** over binary classification

### 3. **Feature Coverage is Sparse in Target Regions**

- Most vessels never enter the Gulf/Med region, so regional features are sparse
- This is expected and is handled by rank-based evaluation

---

## 📊 Dataset Characteristics

- **Total vessels**: 192,092
- **IUU vessels**: 11 (0.0057%)
- **Features**: 30+ (proxy risk signals; AIS gap fields unavailable in mmsi‑daily v3)
- **Missing values**: Region-specific features have 95%+ missing values (expected)

### Feature Statistics (IUU vs Non-IUU)

| Feature | IUU Mean | Non-IUU Mean | Difference |
|---------|----------|--------------|------------|
| `all_fishing_hours_sum` | 4,215 | 2,787 | IUU vessels fish MORE |
| `all_n_points` | 2,996 | 2,365 | IUU vessels have MORE data points |
| `all_lat_mean` | ~5.3° | ~25.1° | IUU vessels are in different latitudes |
| `all_lon_mean` | ~76.8° | ~48.9° | IUU vessels are in different longitudes |

**Note**: IUU vessels appear to have different geographic patterns, but the differences are subtle.

---

## 🎯 Root Causes

1. **Too Few Positive Examples**: Only 8 IUU vessels is insufficient for supervised learning
2. **Weak Features**: Current features don't capture IUU behavior patterns
3. **Class Imbalance**: 18,943:1 ratio overwhelms the model
4. **Model Defaults to Zero**: Logistic regression with extreme imbalance predicts everything as negative

---

## ✅ Implemented Improvements (Small-Label Strategy)

### 1. **AIS Gap Features (Behavioral Proxy)**

The pipeline supports AIS gap aggregates, but **mmsi‑daily v3 does not include `hours_gap`**, so these fields are empty in the current run.

### 2. **Rule‑Based Risk Baseline**

An interpretable score that does not rely on large labels:
- +2 if near EEZ boundary (`near_eez_boundary_50km`)
- +2 if SAR hits near vessel (`sar_manual_bin_hits_at_region_mean` > 0)
- +1 if flag risk score ≥ 4
- +1 if strong region focus (`region_hours_fraction` ≥ 0.2)
- +1 if AIS gaps are large (`gap_hours_mean` ≥ 6)

This provides a defensible baseline and lets us compare the ML model against a human‑interpretable method.

### 3. **PU‑Style Evaluation**

We now report ranking metrics as “known‑positive precision” (treating negatives as unlabeled):
- Top‑K precision, recall, lift
- Mean Reciprocal Rank (MRR)
- PR‑AUC for ranking quality

### 4. **Top‑K Explanation Report**

Generated report:  
`outputs/top_risky_vessels_explanations.csv`

Includes top‑100 vessels with model score + key proxy features so results are explainable.

---

## 🚀 How to Run the Updated Evaluation

1) Build features (now includes AIS gaps):
```
python -u "Senior Project.py"
```

2) Train and evaluate (includes rule baseline + PU metrics):
```
python -u "train_model.py" --model gbdt
```

---

## ✅ Latest GBDT Results (2013–2024, tuned)

- **Top‑25**: precision 0.1200 (TP=3), recall 0.2727, lift 2095.55  
- **Top‑100**: precision 0.0500 (TP=5), recall 0.4545, lift 873.15  
- **Top‑1000**: precision 0.0080 (TP=8), recall 0.7273, lift 139.70  
- **MRR**: 0.1237  

These metrics are based on full‑dataset ranking (not thresholded classification) and are stronger than the previous baseline.

---

## 📈 Success Metrics to Track

After implementing fixes, track:

1. **Top‑K Precision / Lift**: How many known IUU in top 25/50/100?
2. **MRR**: Are known IUU vessels ranked near the top?
3. **Model vs Rule Baseline**: Does ML beat the rule score?
4. **Explainability**: Are top‑risk vessels supported by proxy signals?

---

## 📊 Stability Check (GBDT)

Top‑K overlap across random seeds shows moderate stability at larger K:
- **Top‑25**: Jaccard 0.064–0.136  
- **Top‑100**: Jaccard 0.282–0.307  
- **Top‑1000**: Jaccard 0.322–0.431  

This supports using larger Top‑K lists (e.g., 100–1000) for triage.

---

## 🔍 Next Steps

1. ✅ **DONE**: Add AIS gap features + rule baseline + PU evaluation
2. ⏭️ **NEXT**: Re-run training and paste updated metrics
3. ⏭️ **OPTIONAL**: Add true EEZ point‑in‑polygon or AIS gap/SAR time matching

---

## 💡 Key Insight

**With a small IUU list, the right goal is ranking risk, not binary classification.**

The pipeline now:
- Uses proxy behavioral and geopolitical signals
- Compares ML ranking vs a transparent rule baseline
- Produces an explanation report for top‑risk vessels
