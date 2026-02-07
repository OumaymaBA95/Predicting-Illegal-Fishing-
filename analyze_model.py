#!/usr/bin/env python3
"""
Quick analysis script to understand model performance and feature importance.

Usage:
    python analyze_model.py --data outputs/vessel_features_2020-2024.csv
"""

import argparse
import os
import pandas as pd
import numpy as np


def analyze_dataset(df: pd.DataFrame):
    """Analyze the dataset characteristics."""
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    print(f"\nTotal vessels: {len(df):,}")
    print(f"IUU vessels (is_iuu=1): {df['is_iuu'].sum()}")
    print(f"Non-IUU vessels: {(df['is_iuu']==0).sum():,}")
    print(f"Class imbalance ratio: {(df['is_iuu']==0).sum() / max(df['is_iuu'].sum(), 1):.0f}:1")
    
    # Feature columns
    feature_cols = [c for c in df.columns if c not in ["mmsi", "is_iuu", "imo"]]
    print(f"\nTotal features: {len(feature_cols)}")
    
    # Missing values
    missing = df[feature_cols].isna().sum()
    if missing.any():
        print(f"\nFeatures with missing values:")
        for col, count in missing[missing > 0].items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print("\n✅ No missing values in features")
    
    # Feature statistics
    print(f"\nFeature statistics (IUU vs non-IUU):")
    iuu_mask = df["is_iuu"] == 1
    if iuu_mask.any():
        for col in feature_cols[:10]:  # Show first 10 features
            if df[col].dtype in [np.float64, np.int64]:
                iuu_mean = df.loc[iuu_mask, col].mean()
                non_iuu_mean = df.loc[~iuu_mask, col].mean()
                print(f"  {col}:")
                print(f"    IUU mean: {iuu_mean:.4f}, Non-IUU mean: {non_iuu_mean:.4f}")


def analyze_feature_importance(df: pd.DataFrame):
    """Analyze which features correlate most with IUU labels."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Correlation with is_iuu)")
    print("=" * 70)
    
    feature_cols = [c for c in df.columns if c not in ["mmsi", "is_iuu", "imo"]]
    
    # Calculate correlations
    correlations = {}
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.int64]:
            corr = df[col].corr(df["is_iuu"])
            if not np.isnan(corr):
                correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop 15 features by absolute correlation:")
    print(f"{'Feature':<40} {'Correlation':>12}")
    print("-" * 54)
    for col, corr in sorted_corr[:15]:
        direction = "↑" if corr > 0 else "↓"
        print(f"{col:<40} {corr:>11.4f} {direction}")
    
    return sorted_corr


def analyze_risk_scores(df: pd.DataFrame, risk_scores: np.ndarray):
    """Analyze the distribution of risk scores."""
    print("\n" + "=" * 70)
    print("RISK SCORE ANALYSIS")
    print("=" * 70)
    
    df_risk = df.copy()
    df_risk["risk_score"] = risk_scores
    
    print(f"\nRisk score statistics:")
    print(f"  Mean: {risk_scores.mean():.4f}")
    print(f"  Median: {np.median(risk_scores):.4f}")
    print(f"  Min: {risk_scores.min():.4f}")
    print(f"  Max: {risk_scores.max():.4f}")
    print(f"  Std: {risk_scores.std():.4f}")
    
    # Risk score distribution by label
    iuu_scores = risk_scores[df["is_iuu"] == 1]
    non_iuu_scores = risk_scores[df["is_iuu"] == 0]
    
    print(f"\nRisk scores by label:")
    if len(iuu_scores) > 0:
        print(f"  IUU vessels: mean={iuu_scores.mean():.4f}, median={np.median(iuu_scores):.4f}")
    print(f"  Non-IUU vessels: mean={non_iuu_scores.mean():.4f}, median={np.median(non_iuu_scores):.4f}")
    
    # Top risky vessels
    top_risky = df_risk.nlargest(25, "risk_score")[["mmsi", "is_iuu", "risk_score"]]
    print(f"\nTop 25 risky vessels:")
    print(top_risky.to_string(index=False))
    
    # How many IUU vessels in top 25?
    iuu_in_top25 = top_risky["is_iuu"].sum()
    print(f"\nIUU vessels in top 25: {iuu_in_top25} / {df['is_iuu'].sum()} total IUU vessels")


def main():
    parser = argparse.ArgumentParser(description="Analyze model performance and features")
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/vessel_features_2020-2024.csv",
        help="Path to vessel features CSV",
    )
    parser.add_argument(
        "--risk-scores",
        type=str,
        help="Path to CSV with risk scores (mmsi,risk_score). If not provided, will try to compute from model.",
    )
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.data):
        print(f"ERROR: {args.data} not found!")
        return 1
    
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Analyze feature importance
    correlations = analyze_feature_importance(df)
    
    # Analyze risk scores if available
    if args.risk_scores and os.path.exists(args.risk_scores):
        risk_df = pd.read_csv(args.risk_scores)
        df_merged = df.merge(risk_df, on="mmsi", how="left")
        if "risk_score" in df_merged.columns:
            analyze_risk_scores(df, df_merged["risk_score"].fillna(0).values)
    else:
        print("\n" + "=" * 70)
        print("NOTE: Risk scores not provided.")
        print("Run 'python train_model.py' first to generate risk scores,")
        print("then run: python analyze_model.py --data <data> --risk-scores outputs/top_risky_vessels.csv")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
