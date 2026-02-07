#!/usr/bin/env python3
"""
Feature Engineering Ideas and Helper Functions

This script provides ideas and helper functions for creating better features
that might be more predictive of IUU fishing behavior.
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_temporal_features(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add temporal features that might indicate IUU behavior.
    
    Ideas:
    - Fishing consistency (variance in fishing hours over time)
    - Seasonal patterns (more fishing in certain months)
    - Day/night fishing ratios
    - Weekend vs weekday patterns
    """
    df = df.copy()
    
    # Example: If you have date information, you could add:
    # - month, day_of_week, is_weekend
    # - fishing_hours_variance (consistency)
    # - peak_season_flag
    
    # For now, this is a placeholder showing the structure
    print("Note: Temporal features require date information in the dataset")
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add behavioral features that might indicate IUU behavior.
    
    Ideas:
    - AIS gap patterns (vessels that turn off AIS frequently)
    - Speed patterns (suspicious speed changes)
    - Port visit frequency
    - Time spent in high-risk areas
    - Distance traveled vs fishing hours ratio
    """
    df = df.copy()
    
    # Example: Fishing efficiency (fishing hours per data point)
    if "all_fishing_hours_sum" in df.columns and "all_n_points" in df.columns:
        df["fishing_efficiency"] = (
            df["all_fishing_hours_sum"] / (df["all_n_points"] + 1)
        )
    
    # Example: Region concentration (how focused is fishing in target regions?)
    if "region_hours_fraction" in df.columns:
        df["region_concentration"] = df["region_hours_fraction"].fillna(0)
    
    # Example: Geographic spread (how spread out is fishing?)
    # This would require lat/lon variance, which we don't currently compute
    
    return df


def add_risk_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add risk indicator features based on vessel characteristics.
    
    Ideas:
    - Flag state risk (some flags are higher risk)
    - Vessel age/condition
    - Historical violations
    - Vessel class risk (certain gear types are higher risk)
    """
    df = df.copy()
    
    # Example: Flag state risk (placeholder - would need a risk lookup table)
    if "fv_flag_gfw" in df.columns:
        # High-risk flags (example - would need actual data)
        high_risk_flags = {"Unknown", "XXX", ""}  # Placeholder
        df["high_risk_flag"] = df["fv_flag_gfw"].isin(high_risk_flags).astype(int)
    
    # Example: Vessel class risk (certain gear types might be higher risk)
    if "fv_vessel_class_gfw" in df.columns:
        # High-risk gear types (example - would need domain knowledge)
        high_risk_gear = {"Purse seine", "Longline"}  # Placeholder
        df["high_risk_gear"] = df["fv_vessel_class_gfw"].isin(high_risk_gear).astype(int)
    
    # Example: Vessel size risk (very small or very large might be suspicious)
    if "fv_length_m_gfw" in df.columns:
        df["suspicious_size"] = (
            (df["fv_length_m_gfw"] < 10) | (df["fv_length_m_gfw"] > 100)
        ).astype(int)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features (combinations of existing features).
    
    Ideas:
    - Fishing hours × region concentration
    - EEZ distance × fishing hours
    - SAR hits × fishing hours
    """
    df = df.copy()
    
    # Example: High fishing + near EEZ boundary = suspicious
    if "all_fishing_hours_sum" in df.columns and "near_eez_boundary_50km" in df.columns:
        df["high_fishing_near_eez"] = (
            (df["all_fishing_hours_sum"] > df["all_fishing_hours_sum"].quantile(0.75)) &
            (df["near_eez_boundary_50km"] == 1)
        ).astype(int)
    
    # Example: High fishing + SAR detections = suspicious
    if "all_fishing_hours_sum" in df.columns and "sar_manual_bin_hits_at_all_mean" in df.columns:
        df["high_fishing_with_sar"] = (
            (df["all_fishing_hours_sum"] > df["all_fishing_hours_sum"].quantile(0.75)) &
            (df["sar_manual_bin_hits_at_all_mean"] > 0)
        ).astype(int)
    
    # Example: Region fishing but not in target region = suspicious
    if "region_fishing_hours_sum" in df.columns and "in_target_region" in df.columns:
        df["region_fishing_outside_target"] = (
            (df["region_fishing_hours_sum"] > 0) &
            (df["in_target_region"] == 0)
        ).astype(int)
    
    return df


def analyze_feature_candidates(df: pd.DataFrame, target_col: str = "is_iuu") -> pd.DataFrame:
    """
    Analyze potential new features to see which would be most predictive.
    
    Returns a dataframe with feature names and their correlations with the target.
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found")
        return pd.DataFrame()
    
    correlations = {}
    for col in df.columns:
        if col != target_col and df[col].dtype in [np.float64, np.int64]:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations[col] = {
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "mean_iuu": df[df[target_col] == 1][col].mean() if (df[target_col] == 1).any() else np.nan,
                    "mean_non_iuu": df[df[target_col] == 0][col].mean() if (df[target_col] == 0).any() else np.nan,
                }
    
    result_df = pd.DataFrame(correlations).T
    result_df = result_df.sort_values("abs_correlation", ascending=False)
    
    return result_df


def main():
    """
    Example usage of feature engineering functions.
    """
    import os
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "outputs", "vessel_features_2020-2024.csv")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Run 'Senior Project.py' first to generate the dataset.")
        return
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} vessels")
    
    print("\n" + "=" * 70)
    print("CURRENT FEATURE ANALYSIS")
    print("=" * 70)
    feature_analysis = analyze_feature_candidates(df)
    print(feature_analysis.head(15).to_string())
    
    print("\n" + "=" * 70)
    print("ADDING NEW FEATURES")
    print("=" * 70)
    
    # Add behavioral features
    df = add_behavioral_features(df)
    print("✓ Added behavioral features")
    
    # Add risk indicator features
    df = add_risk_indicator_features(df)
    print("✓ Added risk indicator features")
    
    # Add interaction features
    df = add_interaction_features(df)
    print("✓ Added interaction features")
    
    print("\n" + "=" * 70)
    print("NEW FEATURE ANALYSIS")
    print("=" * 70)
    
    # Analyze new features
    new_features = [col for col in df.columns if col not in pd.read_csv(data_path).columns]
    if new_features:
        print(f"\nNew features created: {len(new_features)}")
        print(f"New feature names: {new_features}")
        
        new_analysis = analyze_feature_candidates(df[new_features + ["is_iuu"]])
        if not new_analysis.empty:
            print("\nNew feature correlations:")
            print(new_analysis.to_string())
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print("\nTo use these features, modify 'Senior Project.py' to include them,")
    print("or save this enhanced dataset and use it for training.")


if __name__ == "__main__":
    main()
