import pandas as pd
import os
import glob
import numpy as np
# ML libraries imported for future use (not used in current data processing step)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# Load Combined IUU List (known bad vessels – gold labels)
iuu = pd.read_csv('IUUList-20251108.csv')
# Convert MMSI to lowercase 'mmsi' and create 'is_iuu' column
iuu['mmsi'] = iuu['MMSI'].astype(str).str.strip()
iuu['is_iuu'] = iuu['CurrentlyListed'].astype(int)  # Already boolean, convert to int
# Filter out rows with missing/invalid MMSI
iuu = iuu[iuu['mmsi'].notna() & (iuu['mmsi'] != '') & (iuu['mmsi'] != 'nan')]
# Create lookup dictionary for faster matching (much faster than merge for large datasets)
iuu_dict = dict(zip(iuu['mmsi'], iuu['is_iuu']))
print(f"Loaded {len(iuu_dict)} IUU vessels from the list")


# Point to the folder with all the daily CSVs
data_folder = "/Users/momoba/Desktop/Senior Final Project/Updated CSV/GFW Fishing Effort /mmsi-daily-csvs-10-v3-2024/"

# Find all 2024 daily files
csv_files_2024 = glob.glob(os.path.join(data_folder, "mmsi-daily-csvs-10-v3-2024-*.csv"))


agg_list = []
for file_path in sorted(csv_files_2024):
    chunks = pd.read_csv(file_path, chunksize=100000)
    for chunk in chunks:
        # Filter for Gulf of Mexico OR Mediterranean Sea in one combined condition
        # Gulf: lat 20-30, lon -98 to -80 | Mediterranean: lat 30-46, lon -6 to 36
        filtered = chunk[
            ((chunk['cell_ll_lat'].between(20, 30)) & (chunk['cell_ll_lon'].between(-98, -80))) |
            ((chunk['cell_ll_lat'].between(30, 46)) & (chunk['cell_ll_lon'].between(-6, 36)))
        ]
        if not filtered.empty:                     
            agg = filtered.groupby('mmsi').agg({
                'fishing_hours': 'sum',
                'cell_ll_lat': 'mean',           
                'cell_ll_lon': 'mean'              
            })
            agg_list.append(agg)
            print(f"Added {len(agg)} vessels from {os.path.basename(file_path)}")

# Aggregate all data after processing all files
if agg_list:
    gfw_agg = pd.concat(agg_list)
    # Re-aggregate: sum fishing_hours, recalculate mean for lat/lon
    gfw_agg = gfw_agg.groupby('mmsi').agg({
        'fishing_hours': 'sum',
        'cell_ll_lat': 'mean',
        'cell_ll_lon': 'mean'
    }).reset_index()
    
    # Convert mmsi to string and add IUU labels using dictionary lookup (faster than merge)
    gfw_agg['mmsi'] = gfw_agg['mmsi'].astype(str)
    gfw_agg['is_iuu'] = gfw_agg['mmsi'].map(iuu_dict).fillna(0).astype(int)
    
    print(f"\nTotal vessels processed: {len(gfw_agg)}")
    print(f"IUU vessels: {gfw_agg['is_iuu'].sum()}")
else:
    print("No data found matching the criteria!")



