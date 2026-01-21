# Predictin Illegal Fishing 

## What Is This Project About?

Illegal, Unreported, and Unregulated (IUU) fishing is a major global problem. It removes too many fish from the ocean, harms marine life, and hurts honest fishermen and coastal communities. This senior project builds a system to help spot suspicious fishing activity by combining five public datasets. The goal is to create a machine learning model that gives each vessel a risk score (for example, "98% chance of illegal activity") so authorities can focus inspections on the most likely violators.
The project is tailored to two important areas:

Houston and the Gulf of Mexico: where Mexican boats sometimes cross into U.S. waters illegally to catch red snapper and other species.
Tunisia and the Mediterranean Sea: where overfishing of tuna, swordfish, and sharks is a serious issue, even with rules from the General Fisheries Commission for the Mediterranean (GFCM).


## The Five Datasets Used

The project combines information from five main sources. Each provides a different piece of the puzzle:

### 1. Combined IUU Vessel List

File: combined_iuu_list.csv
Purpose: Lists known illegal vessels (the "gold standard" truth labels).
Key columns: is_iuu (1 = confirmed illegal), vessel_name, flag, imo.
How it connects: Match on MMSI (vessel ID). If no exact match (~80% success), use fuzzy matching on name or IMO number.

### 2. GFW Fishing Effort

File: mmsi-daily-10-v3.csv
Purpose: Daily tracks of fishing boats from AIS signals (shows position, fishing hours, gaps when AIS is off).
Key columns: fishing_hours, lat, lon, flag, geartype, hours_gap.
How it connects: This is the main dataset. Labels from #1 are added to every row of matching vessels.

### 3. IUU Fishing Risk Index

File: iuu_risk_index_2023.csv (or latest 2025 update)
Purpose: Gives a risk score (1 = low risk, 5 = high risk) for each country's fishing fleet.
Key columns: flag (country code), risk_score.
How it connects: Match on flag to add the country's risk level to all its vessels.

### 4. World EEZ Boundaries

File: eez_v12.geojson converted to CSV
Purpose: Maps legal fishing zones (Exclusive Economic Zones). Helps detect boats fishing illegally in another country's waters.
Key columns: eez_id, sovereign (country), geometry (map shapes).
How it connects: For each boat position, check if it is inside a foreign EEZ (in_foreign_eez = 1) or far from any EEZ boundary (dist_to_eez_km).

### 5. xView3-SAR Labels

File: train.csv
Purpose: Satellite radar images detect boats that turn off AIS ("dark vessels").
Key columns: detected (1 = boat seen), scene_id, sar_timestamp.
How it connects: Match radar detections to gaps in AIS data (within 50 km and ±3 hours). If matched, mark dark_risk = 1.


## How the Data Is Combined (Merges)

The process starts with the GFW dataset (daily boat tracks) and adds information step by step:
1. Add known illegal labels from the IUU list (direct match on MMSI, fuzzy fallback if needed).
2. Add country risk score from the Risk Index (match on flag).
3. Check locations against EEZ maps to find illegal zone entries or distance to boundaries.
4. Look for "dark" behavior by matching satellite radar hits to times when boats disappear from AIS.


## Why XGBoost and Random Forest?

These two machine learning models work best here because:

1. The data is in tables (numbers like fishing hours, categories like gear type, yes/no flags like in_foreign_eez).
2. IUU cases are rare, so the models handle imbalance well.
3. Combining them (stacking) gives even better accuracy.


## Why Focus on Houston (Gulf of Mexico/America) and Tunisia (Mediterranean)?

Gulf of Mexico (near Houston): Mexican boats often enter U.S. waters illegally. The U.S. Coast Guard catches dozens of these "lanchas" each year, but enforcement is challenging due to resources and priorities. Recent reports show fewer interdictions in 2025 despite continued issues.
Tunisia (Mediterranean): Overfishing affects tuna, swordfish, and sharks. Tunisia scores medium-high on the IUU Risk Index due to vulnerability and compliance challenges. Regional rules (GFCM) exist, but enforcement needs better tools.