## Predicting Illegal Fishing (IUU) 

### What is this project about?
Illegal, Unreported, and Unregulated (IUU) fishing is a major global problem. It removes too many fish from the ocean, harms marine life, and hurts honest fishermen and coastal communities. This senior project builds a system to help spot suspicious fishing activity by combining five public datasets. The goal is to create a model that gives each vessel a **risk score** so authorities can focus inspections on the most likely violators.

The project is tailored to two important areas:
- **Houston / Gulf of Mexico**: illegal crossings by small boats targeting red snapper and other species.
- **Tunisia / Mediterranean Sea**: pressure on tuna, swordfish, and shark fisheries under GFCM rules.

### The five datasets used
Each source provides a different piece of the puzzle:

## Data sources (five datasets)
1. **Combined IUU Vessel List** (`combined_iuu_list.csv`)  
   Labels known illegal vessels; matched by MMSI with optional IMO fallback.
2. **GFW Fishing Effort** (`mmsi-daily-csvs-10-v3-YYYY/`)  
   Daily AIS‑based fishing activity and positions.
3. **IUU Fishing Risk Index** (`iuu_risk_index_2023.csv` or latest)  
   Country‑level risk score by flag.
4. **World EEZ Boundaries** (`eez_boundaries_v12.csv`)  
   Boundary‑distance proxy features (not full polygon containment).
5. **xView3 SAR Labels** (`xView3-SAR Ship Detection Dataset/*.csv`)  
   SAR detection bins as a proxy for dark‑vessel risk.


### How the data is combined (merges)
The process starts with the GFW dataset (daily boat tracks) and adds information step by step:
- Add known illegal labels from the IUU list (direct match on MMSI; optional IMO fallback).
- Add country risk score from the Risk Index (match on flag).
- Add EEZ boundary‑distance proxy features.
- Add SAR detection proxy features.


Thus, This project produces a **ranked list** of vessels for inspection. It does **not** make legal determinations. The ranking approach is necessary because only a **small confirmed IUU list** exists relative to the full AIS dataset.

### What the training data does 
We turn each vessel’s daily AIS activity into **one summary row per vessel** (fishing hours, AIS gaps, EEZ boundary proximity, SAR detections, region focus, and metadata). We label known IUU vessels and treat the rest as **unknown** rather than confirmed legal. The model learns patterns from known illegal vessels versus **very low‑risk vessels**, then produces a **risk ranking** of all vessels.

### How to interpret results
- The **risk score is a ranking signal**, not a legal decision.
- Higher score means **more suspicious relative to others**.
- The output is meant to **prioritize inspections**, not replace them.

### Key output files
- `outputs/top_risky_vessels.csv` → global ranked list
- `outputs/top_risky_vessels_pu.csv` → PU‑only ranked list (recommended)
- `outputs/top_risky_vessels_gulf_med.csv` → ranked list for Gulf + Med
- `outputs/top_risky_vessels_explanations.csv` → top‑100 explanations
- `outputs/top_risky_vessels_ensemble.csv` → optional ensemble ranking


## Modeling approach (detailed)
### PU learning (Positive + Unlabeled)
Only a small set of vessels is confirmed IUU. We treat those as **positives** and treat everyone else as **unlabeled**, not guaranteed legal. We then sample **reliable negatives** (very low‑risk vessels) and train the model to separate **known IUU** from **reliable low‑risk** vessels. This reduces label bias and improves ranking quality.

### Train/test split (what “stratified” means)
The train/test split is **stratified**, which keeps the same IUU vs non‑IUU ratio in both sets. With `--test-size 0.2`, about 80% goes to train and 20% to test, and both sets still contain known IUU vessels. When PU training is enabled, the split is applied **only** to the PU subset (positives + reliable negatives), not the full unlabeled pool.

### Gradient‑Boosted Trees (GBDT)
The primary model uses GBDT, which builds many small decision trees in sequence. Each tree corrects errors from the previous one, allowing the model to learn **non‑linear patterns** such as “high risk only when multiple signals align.” The output is a **risk score** used to rank vessels.

### Baselines and support signals
- **Weighted logistic regression**: a simple linear baseline for comparison.
- **Rule‑based score**: interpretable signals (EEZ proximity, AIS gaps, SAR hits, flag risk).
- **Anomaly scoring**: Isolation Forest for unlabeled outliers when labels are extremely sparse.

### Evaluation
We evaluate with **Top‑K precision/recall** rather than accuracy. The goal is to place known IUU vessels high in the ranked list (Top‑25/100/500), not to make binary legal decisions.

## What runs in this repo
- `Senior Project.py` → builds `outputs/vessel_features_*.csv`
- `train_model.py` → trains models and writes ranked lists
- `anomaly_model.py` → unsupervised anomaly ranking

## Setup
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Data inputs (expected locations)
Required:
- `combined_iuu_list.csv` (preferred) or `IUUList-20251108.csv`
- `mmsi-daily-csvs-10-v3-2013/` … `mmsi-daily-csvs-10-v3-2024/`

Optional (adds features):
- `Updated CSV/ World EEZ Boundaries (v12)/World_EEZ_v12_20231025_gpkg/eez_boundaries_v12.csv`
- `xView3-SAR Ship Detection Dataset/public.csv` and `validation.csv`
- `CSV/fa615300-b670-11f0-b282-dfbcfd65324c.zip` (fishing‑vessels metadata)

## Build the dataset
All years:
```bash
. .venv/bin/activate
python -u "Senior Project.py"
```

Specific range:
```bash
. .venv/bin/activate
GFW_START_YEAR=2013 GFW_END_YEAR=2024 python -u "Senior Project.py"
```

Outputs:
- `outputs/vessel_features_YYYY-YYYY.csv`
- `outputs/vessel_features_all_years.csv`

## Train + rank vessels
```bash
. .venv/bin/activate
python "train_model.py" --model gbdt --data "outputs/vessel_features_all_years.csv"
```

Outputs:
- `outputs/top_risky_vessels.csv`
- `outputs/top_risky_vessels_pu.csv`
- `outputs/top_risky_vessels_gulf_med.csv`
- `outputs/top_risky_vessels_explanations.csv`
- `outputs/top_risky_vessels_ensemble.csv`

## Live dashboard (interactive)
After you have outputs (features + ranked lists), you can browse them in a live UI:

```bash
. .venv/bin/activate
streamlit run dashboard.py
```

Then open the app in your browser:
- Local URL: `http://localhost:8501`

What it shows:
- Dataset stats (rows, known IUU count, base rate)
- Top‑K ranked lists (PU / ensemble / anomaly) joined to key features
- Optional Gulf+Med-only filter (`in_target_region`)
- A simple map of mean vessel locations
- Embedded evaluation write-ups (`MODEL_EVALUATION_RESULTS.md`, `MODEL_COMPARISON_SUMMARY.md`, `Cross validation.md`, `Cross-validation.html`, `outputs/pu_scoring_report.md`)

### Optional: deploy it (make it “live” on the web)
- Push this repo to GitHub
- Create a Streamlit Community Cloud app pointing at this repo ([Streamlit Community Cloud](https://streamlit.io/cloud))
- Set the app entrypoint to `dashboard.py`
- Ensure `requirements.txt` is present (it is in this repo)
After deployment, Streamlit will give you a public URL like `https://<your-app-name>.streamlit.app`.

## Recommended workflow (tiny label list)
Use **PU‑only ranking** and review a fixed **Top‑K** list. Compare against the rule‑based baseline for interpretability.

## Run: anomaly detection (no labels required)
```bash
. .venv/bin/activate
python "anomaly_model.py" --data "outputs/vessel_features_all_years.csv"
```

Outputs:
- `outputs/top_risky_vessels_anomaly.csv`
- `outputs/top_risky_vessels_gulf_med_anomaly.csv`

## Feature examples
- GFW aggregates: `all_fishing_hours_sum`, `all_lat_mean`, `all_lon_mean`
- Region flags: `in_gulf`, `in_mediterranean`, `in_target_region`
- Behavioral: `fishing_hours_per_point`, `log_all_n_points`
- Spatial coverage: `cell_unique_count_sum`, `cell_unique_count_mean_per_day`
- Dispersion: `all_lat_var`, `all_lon_var`
- Activity: `active_days`, `fishing_days_by_date`, `fishing_days_fraction_by_date`
- Fishing intensity: `fishing_hours_max`, `fishing_hours_median`, `fishing_hours_p95`
- AIS gaps: `gap_hours_mean`, `gap_hours_p95`, `gap_over_6h_fraction`, `gap_over_12h_fraction`
- Seasonality: `fishing_month_entropy`, `fishing_peak_month_fraction`, `fishing_months_active`
- SAR proxy: `sar_manual_bin_hits_at_all_mean`, `sar_manual_bin_hits_at_region_mean`
- EEZ proxy: `eez_boundary_dist_km_region_mean`, `near_eez_boundary_50km`
- Vessel metadata (fv_*): `fv_flag_gfw`, `fv_vessel_class_gfw`, `fv_length_m_gfw`, etc.
- Identity diversity: `flag_unique_count`, `flag_entropy`, `flag_top_fraction`, `geartype_unique_count`, `geartype_entropy`

## Notes / limitations
- **IUU labeling is MMSI‑based**; IMO fallback requires a registry file.
- **EEZ features are boundary‑distance proxies**, not full polygon checks.
- **SAR feature is spatial‑only**, not AIS‑gap time matching.
- AIS gaps require `hours_gap` in the AIS source (not always present).


### Terminology / glossary (quick)
- **IUU**: *Illegal, Unreported, and Unregulated* fishing. In this project it’s the “illegal” label we try to learn/predict.
- **MMSI**: *Maritime Mobile Service Identity* (a 9-digit vessel identifier broadcast by AIS). This project uses MMSI as the main vessel ID.
- **AIS**: *Automatic Identification System* radio broadcasts from vessels (position/time/ID). GFW derives “apparent fishing effort” from AIS.
- **Risk score**: the model output probability-like number (0–1) used to rank vessels by risk.
- **EEZ**: *Exclusive Economic Zone* (maritime zone). We use an EEZ **boundary distance** proxy feature (not full polygon containment).


### Notes / limitations (current implementation)
- **IUU labeling is MMSI-based**: the label is `is_iuu=1` if a vessel’s MMSI appears in the IUU list. 
- **IMO fallback**: “IMO matching” means using the vessel’s IMO number as an alternate ID, but that requires a separate MMSI→IMO registry file.
- **EEZ is a boundary-distance proxy**: we estimate distance to the nearest EEZ boundary line; we do not compute “inside foreign EEZ”.
- **SAR feature is spatial-only**: we count nearby xView3 “manual” detections; we do not do full AIS-gap ±3 hour matching.
- **AIS gap fields are unavailable**: mmsi‑daily v3 does not include `hours_gap`, so gap features are empty unless you have a different AIS dataset.

### Environment variables (optional)
- **`GFW_START_YEAR` / `GFW_END_YEAR`**: limit the year range
- **`IUU_LIST_PATH`**: override IUU list path
- **`EEZ_BOUNDARIES_PATH`**: override EEZ boundary CSV path
- **`EEZ_POINT_STRIDE`**: sample EEZ boundary vertices (speed/accuracy tradeoff; default `20`)
- **`EEZ_MAX_RADIUS_DEG`**: search radius for EEZ boundary distance (default `8`)
- **`FISHING_VESSELS_PATH` / `FISHING_VESSELS_ZIP`**: override fishing-vessels metadata source


#### Optional (adds features)
- **World EEZ boundaries (v12)**:
  - `Updated CSV/ World EEZ Boundaries (v12)/World_EEZ_v12_20231025_gpkg/eez_boundaries_v12.csv`
  - Used to compute **approx distance to nearest EEZ boundary** (a proxy feature).
- **xView3 SAR labels**:
  - `xView3-SAR Ship Detection Dataset/public.csv`
  - `xView3-SAR Ship Detection Dataset/validation.csv`
  - Used to compute a lightweight **“manual SAR detections near vessel mean location”** feature.
- **GFW fishing-vessels metadata**:
  - If you have `CSV/fa615300-b670-11f0-b282-dfbcfd65324c.zip`, the script reads `fishing-vessels-v3.csv` from inside and joins `fv_*` metadata columns.





