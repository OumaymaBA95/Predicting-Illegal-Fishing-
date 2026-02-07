# Next Steps for Senior Project

## Current Status ✅

Your project has a **working end-to-end pipeline**:

1. **Data Processing** (`Senior Project.py`)
   - ✅ Processes GFW fishing effort data (2020-2024)
   - ✅ Creates 28 features per vessel (fishing hours, regions, SAR, EEZ, metadata)
   - ✅ Labels vessels using IUU list (MMSI + IMO matching)
   - ✅ Outputs: `vessel_features_2020-2024.csv` (151,555 vessels, 8 IUU positives)

2. **Model Training** (`train_model.py`)
   - ✅ Trains logistic regression model
   - ✅ Evaluates performance (PR-AUC, ROC-AUC, precision, recall, F1)
   - ✅ Outputs risk scores and top risky vessels lists

3. **IMO Matching**
   - ✅ 50 MMSI-IMO pairs extracted and documented
   - ✅ 13 vessels without IMO numbers identified and verified

---

## Recommended Next Steps (Prioritized)

### 🔴 **HIGH PRIORITY** - Model Evaluation & Validation

**1. Check Model Performance**
```bash
python train_model.py --data outputs/vessel_features_2020-2024.csv
```
- Review the PR-AUC, ROC-AUC, precision, recall metrics
- **Question**: Is the model actually learning? (With only 8 positives out of 151K, this is challenging)
- **Action**: If metrics are poor, consider:
  - Class balancing techniques (oversampling, SMOTE, or class weights)
  - Feature importance analysis to see what the model is using
  - Cross-validation to ensure stability

**2. Feature Importance Analysis**
Create a script to analyze which features are most predictive:
- Which features correlate most with IUU labels?
- Are region-specific features helping?
- Are SAR/EEZ features contributing?

**3. Validate Risk Scores**
- Check if the top risky vessels make sense:
  - Do they have suspicious patterns (high fishing hours, near EEZ boundaries)?
  - Are any known IUU vessels appearing in the top risk list?
  - Are false positives explainable?

---

### 🟡 **MEDIUM PRIORITY** - Model Improvements

**4. Try Different Models**
Your current model is logistic regression. Consider:
- **Random Forest** (if you can resolve SSL issues or use alternative package source)
- **Gradient Boosting** (XGBoost/LightGBM - may need installation)
- **Ensemble methods** (combine multiple models)

**5. Address Class Imbalance**
With only 8 positives out of 151K vessels:
- **Oversampling**: Duplicate or synthesize IUU examples
- **Class weights**: Weight IUU vessels more heavily during training
- **Threshold tuning**: Adjust decision threshold (currently 0.5) to optimize precision/recall tradeoff

**6. Feature Engineering Enhancements**
- **Temporal features**: Fishing patterns over time (seasonality, consistency)
- **Behavioral features**: Speed patterns, port visits, AIS gaps
- **Network features**: Vessel associations (same owner, same port, etc.)
- **Risk score combinations**: Create interaction features

---

### 🟢 **LOW PRIORITY** - Analysis & Presentation

**7. Visualization & Reporting**
Create visualizations:
- **Risk score distribution**: Histogram of risk scores
- **Feature distributions**: Compare IUU vs non-IUU vessels
- **Geographic maps**: Plot high-risk vessels on a map
- **Time series**: Show fishing patterns over time for risky vessels

**8. Case Studies**
Select a few high-risk vessels and create detailed case studies:
- What makes them suspicious?
- What features contribute to their high risk score?
- Are they actually IUU vessels (if verifiable)?

**9. Documentation & Presentation**
- **Write-up**: Document methodology, results, limitations
- **Slides**: Create presentation summarizing findings
- **Code comments**: Ensure code is well-documented for reviewers

**10. Sensitivity Analysis**
- Test model robustness:
  - How does performance change with different year ranges?
  - What if you exclude certain features?
  - How stable are risk rankings?

---

## Quick Wins (Do These First)

### ✅ **1. Run Model Evaluation** (5 minutes)
```bash
cd "/Users/momoba/Desktop/Senior Final Project"
source .venv/bin/activate
python train_model.py --data outputs/vessel_features_2020-2024.csv
```
**Look for**: PR-AUC > 0.5, ROC-AUC > 0.5 (better than random)

### ✅ **2. Check Top Risky Vessels** (5 minutes)
```bash
cat outputs/top_risky_vessels.csv
cat outputs/top_risky_vessels_gulf_med.csv
```
**Questions**:
- Do these MMSIs appear suspicious?
- Are any known IUU vessels in the top 25?
- Do the risk scores make sense (0.9+ = very risky)?

### ✅ **3. Feature Correlation Analysis** (10 minutes)
Create a quick script to see which features correlate with `is_iuu`:
```python
import pandas as pd
df = pd.read_csv('outputs/vessel_features_2020-2024.csv')
correlations = df.corr()['is_iuu'].sort_values(ascending=False)
print(correlations.head(10))
```

---

## Potential Challenges & Solutions

### Challenge: Very Few Positive Labels (8 IUU vessels)
**Problem**: Machine learning models struggle with extreme class imbalance.

**Solutions**:
1. **Expand IUU list**: Look for additional IUU vessel databases
2. **Semi-supervised learning**: Use unlabeled suspicious vessels
3. **Anomaly detection**: Treat as outlier detection problem instead
4. **Focus on ranking**: Don't try to classify, just rank by risk

### Challenge: Model May Not Be Learning
**Problem**: With so few positives, the model might just predict "all zeros"

**Solutions**:
1. **Class weights**: Make IUU examples count more
2. **Oversampling**: Create synthetic IUU examples
3. **Different evaluation**: Focus on top-K precision instead of overall accuracy

### Challenge: Features May Not Be Predictive
**Problem**: Current features might not capture IUU behavior well

**Solutions**:
1. **Domain knowledge**: Consult fishing experts about suspicious patterns
2. **Feature engineering**: Create more sophisticated behavioral features
3. **External data**: Incorporate port data, ownership data, etc.

---

## Questions to Answer

Before moving forward, consider:

1. **What is the goal?**
   - [ ] Identify specific high-risk vessels for investigation?
   - [ ] Understand patterns in IUU fishing?
   - [ ] Build a production-ready risk scoring system?
   - [ ] Academic research / publication?

2. **What is acceptable performance?**
   - [ ] High precision (few false positives)?
   - [ ] High recall (catch most IUU vessels)?
   - [ ] Good ranking (top vessels are actually risky)?

3. **What resources are available?**
   - [ ] More data sources?
   - [ ] Domain expert consultation?
   - [ ] Computational resources for complex models?

---

## Recommended Immediate Action

**Start with Step 1**: Run the model evaluation and check the metrics. This will tell you:
- Is the model working at all?
- What's the baseline performance?
- Where should you focus improvement efforts?

Then proceed based on the results!
