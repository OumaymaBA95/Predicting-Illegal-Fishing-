#!/usr/bin/env python3
"""
Helper script to identify IUU list entries that need manual IMO lookups.

This script finds entries in combined_iuu_list.csv that have MMSI but are missing IMO,
so you can look them up on Equasis and add them to mmsi_imo_registry.csv.
"""

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IUU_CSV = os.path.join(SCRIPT_DIR, "combined_iuu_list.csv")
REGISTRY_CSV = os.path.join(SCRIPT_DIR, "mmsi_imo_registry.csv")

def normalize_mmsi(mmsi_str):
    """Extract digits only from MMSI string."""
    if pd.isna(mmsi_str):
        return None
    mmsi_clean = str(mmsi_str).replace(".", "").replace(",", "").strip()
    digits = "".join(c for c in mmsi_clean if c.isdigit())
    return digits if len(digits) >= 6 else None  # MMSI should be 6-9 digits

def main():
    print("=" * 70)
    print("Finding IUU entries that need manual IMO lookups")
    print("=" * 70)
    
    # Read IUU list
    if not os.path.exists(IUU_CSV):
        print(f"ERROR: {IUU_CSV} not found!")
        return
    
    df = pd.read_csv(IUU_CSV)
    print(f"\nTotal entries in IUU list: {len(df)}")
    
    # Check which entries have MMSI but missing IMO
    has_mmsi = df["MMSI"].notna()
    has_imo = df["IMO"].notna()
    
    missing_imo = df[has_mmsi & ~has_imo].copy()
    
    print(f"\nEntries with MMSI: {has_mmsi.sum()}")
    print(f"Entries with IMO: {has_imo.sum()}")
    print(f"Entries with BOTH: {(has_mmsi & has_imo).sum()}")
    print(f"\nEntries with MMSI but MISSING IMO: {len(missing_imo)}")
    
    if len(missing_imo) == 0:
        print("\n✅ All entries with MMSI already have IMO!")
        return
    
    # Normalize MMSIs
    missing_imo["mmsi_normalized"] = missing_imo["MMSI"].apply(normalize_mmsi)
    missing_imo = missing_imo[missing_imo["mmsi_normalized"].notna()].copy()
    
    # Check which ones are already in the registry
    already_in_registry = set()
    if os.path.exists(REGISTRY_CSV):
        registry_df = pd.read_csv(REGISTRY_CSV)
        already_in_registry = set(registry_df["mmsi"].astype(str))
        print(f"\nAlready in registry: {len(already_in_registry)} MMSIs")
    
    # Filter out ones already in registry
    needs_lookup = missing_imo[
        ~missing_imo["mmsi_normalized"].astype(str).isin(already_in_registry)
    ].copy()
    
    print(f"Entries needing manual lookup: {len(needs_lookup)}")
    
    if len(needs_lookup) == 0:
        print("\n✅ All entries are already in the registry!")
        return
    
    # Display entries that need lookup
    print("\n" + "=" * 70)
    print("ENTRIES NEEDING MANUAL IMO LOOKUP:")
    print("=" * 70)
    print("\nLook these up on Equasis and add to mmsi_imo_registry.csv\n")
    
    output_cols = ["Name", "MMSI", "mmsi_normalized", "Flag", "VesselType", "GearType"]
    available_cols = [col for col in output_cols if col in needs_lookup.columns]
    
    for idx, row in needs_lookup.iterrows():
        print(f"\n--- Entry {idx + 1} ---")
        print(f"Name: {row.get('Name', 'N/A')}")
        print(f"MMSI (raw): {row.get('MMSI', 'N/A')}")
        print(f"MMSI (normalized): {row.get('mmsi_normalized', 'N/A')}")
        if "Flag" in row and pd.notna(row["Flag"]):
            print(f"Flag: {row['Flag']}")
        if "VesselType" in row and pd.notna(row["VesselType"]):
            print(f"Vessel Type: {row['VesselType']}")
        if "GearType" in row and pd.notna(row["GearType"]):
            print(f"Gear Type: {row['GearType']}")
        print(f"→ Look up MMSI {row['mmsi_normalized']} on Equasis")
    
    # Create a summary CSV for easy reference
    summary_file = os.path.join(SCRIPT_DIR, "needs_imo_lookup.csv")
    summary_df = needs_lookup[["Name", "MMSI", "mmsi_normalized"]].copy()
    summary_df.columns = ["name", "mmsi_raw", "mmsi"]
    summary_df["imo"] = ""  # Empty column for user to fill in
    summary_df = summary_df.sort_values("mmsi")
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "=" * 70)
    print(f"✅ Created summary file: {summary_file}")
    print("   Fill in the 'imo' column with values from Equasis,")
    print("   then append those rows to mmsi_imo_registry.csv")
    print("=" * 70)
    
    # Also show entries with IMO but missing MMSI (less useful but informative)
    missing_mmsi = df[has_imo & ~has_mmsi].copy()
    if len(missing_mmsi) > 0:
        print(f"\nNote: {len(missing_mmsi)} entries have IMO but are missing MMSI.")
        print("   These are less useful for matching since we match by MMSI first.")

if __name__ == "__main__":
    main()
