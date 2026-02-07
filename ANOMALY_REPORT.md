# Anomaly Detection Report (Plain Language)

This report explains **why these vessels look unusual** compared to most vessels.

## Methodology (simple but thorough)

### What data is used
- Every **numeric feature** in the dataset (except IDs like `mmsi`, `imo`, and the label `is_iuu`).
- Examples: total fishing hours, number of AIS points, regional activity, SAR counts, EEZ distance, vessel metadata.

### Step 1: Make each feature comparable (robust z‑score)
For each feature, I compare a vessel’s value to what’s typical:
- **Median** = normal value
- **MAD** (median absolute deviation) = typical spread

Formula used:
- **z = 0.6745 × (x − median) / MAD**

Why this math?
- Regular mean/std is distorted by extreme vessels.
- Median/MAD is **robust**, so outliers don’t “break” the scale.

What does **robust** mean?
- It means the method is **not easily thrown off by extreme values**.
- Example: if most vessels fish 10–100 hours but one has 100,000 hours, the mean shifts a lot.
- The median/MAD stay stable, so the “unusual” score stays reliable.

### Step 2: Find each vessel’s most unusual features
For each vessel:
- Compute **absolute z‑scores** (high or low both count as unusual).
- Pick the **top 5 most extreme features**.

### Step 3: Final anomaly score
The final score is the **average of those top 5 extreme z‑scores**.

So a high score means:
- The vessel is far from normal on **several different features**, not just one.

## What the anomaly score means
- The model looks at each vessel’s numeric features.
- It asks: **“Which features are unusually large or small compared to most vessels?”**
- The **anomaly score** is higher when a vessel is **far from normal** on several features.

## How to read the features
- **`gulf_*` / `med_*`**: activity in Gulf of Mexico or Mediterranean Sea.
- **`*_lat_sum` / `*_lon_sum`**: total of lat/lon points in that region (very large = lots of points there).
- **`*_n_points`**: number of data points (very large = many observations).
- **`*_fishing_hours_sum`**: total fishing hours in that region.

In simple terms: **the vessels below stand out because they show extremely large activity in a region**, compared to most vessels.

## Top 10 Anomalous Vessels (and why)

### MMSI 238117140 (IUU label: 0)
Anomaly score: 170270.5014  
Why it stands out:
- Extremely large Mediterranean activity (very high `med_lat_sum`, `med_lon_sum`)
- Very high regional fishing hours

### MMSI 366986160 (IUU label: 0)
Anomaly score: 142591.0111  
Why it stands out:
- Extremely large Gulf activity (very high `gulf_lat_sum`, `gulf_lon_sum`)
- Very high number of Gulf data points

### MMSI 238987940 (IUU label: 0)
Anomaly score: 141579.5568  
Why it stands out:
- Very large Mediterranean activity
- Very high Mediterranean fishing hours

### MMSI 367553360 (IUU label: 0)
Anomaly score: 140848.9318  
Why it stands out:
- Extremely large Gulf activity
- High regional fishing hours

### MMSI 238988040 (IUU label: 0)
Anomaly score: 138870.3325  
Why it stands out:
- Very large Mediterranean activity
- Very high Mediterranean fishing hours

### MMSI 367088130 (IUU label: 0)
Anomaly score: 137346.5522  
Why it stands out:
- Extremely large Gulf activity
- Very high number of Gulf data points

### MMSI 367496950 (IUU label: 0)
Anomaly score: 135217.8729  
Why it stands out:
- Extremely large Gulf activity
- High regional fishing hours

### MMSI 367414380 (IUU label: 0)
Anomaly score: 134805.1437  
Why it stands out:
- Extremely large Gulf activity
- High regional fishing hours

### MMSI 366986130 (IUU label: 0)
Anomaly score: 133481.9895  
Why it stands out:
- Extremely large Gulf activity
- Very high number of Gulf data points

### MMSI 367735690 (IUU label: 0)
Anomaly score: 133041.9081  
Why it stands out:
- Extremely large Gulf activity
- High regional fishing hours

## Important note
- **Anomaly ≠ Illegal.**  
  These are just the vessels that look **most unusual** compared to typical patterns.
- Use this as a **risk list** to investigate further, not a final verdict.