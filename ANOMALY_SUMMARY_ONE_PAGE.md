# One‑Page Summary: Anomaly Detection Results

## Goal
Identify vessels that look **unusual** compared to typical fishing behavior, given very few confirmed IUU labels.

## Method 
- For each numeric feature, compare a vessel to the “normal” vessel using **median** and **MAD**.
- Compute a **robust z‑score** for each feature (how far from normal).
- The **anomaly score** is the average of the **top 5 most extreme** feature z‑scores.
- Higher score = more unusual behavior across multiple features.

## Why this approach
We only have **8 confirmed IUU vessels** out of **151,555**, so supervised learning is unreliable.  
This anomaly method does **not** need labels and still produces a ranked risk list.

## Data used (examples)
- Total fishing activity: `all_fishing_hours_sum`
- AIS density: `all_n_points`
- Location patterns: `all_lat_mean`, `all_lon_mean`
- Regional intensity: `gulf_*`, `med_*`, `region_fishing_hours_sum`
- Optional: SAR evidence, EEZ distance, vessel metadata (length, flag, engine)

## Top results (see full list)
Files created:
- `outputs/top_risky_vessels_anomaly.csv`
- `outputs/top_risky_vessels_gulf_med_anomaly.csv`

## Interpretation
These are **not confirmed illegal vessels** — they are **statistical outliers**.  
Use the list as a **risk ranking** to investigate further.
