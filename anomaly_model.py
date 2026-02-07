import argparse
import os
import glob

import numpy as np
import pandas as pd


def pick_default_dataset_path(project_dir: str) -> str:
    outputs_dir = os.path.join(project_dir, "outputs")
    candidates = sorted(
        glob.glob(os.path.join(outputs_dir, "vessel_features_*.csv")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return os.path.join(outputs_dir, "vessel_features_all_years.csv")


def robust_zscore(X: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    z = 0.6745 * (x - median) / MAD
    """
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad = np.where(mad == 0, 1.0, mad)
    return 0.6745 * (X - med) / mad


def compute_anomaly_score(X: np.ndarray, top_k: int = 5) -> np.ndarray:
    """
    Anomaly score = mean of top_k absolute robust z-scores per row.
    """
    z = robust_zscore(X)
    z_abs = np.abs(z)
    if top_k <= 0:
        return np.nanmean(z_abs, axis=1)
    k = min(top_k, z_abs.shape[1])
    top_k_vals = np.partition(z_abs, -k, axis=1)[:, -k:]
    return np.nanmean(top_k_vals, axis=1)


def rank_report(y_true: np.ndarray, y_score: np.ndarray, ks: list[int]) -> list[dict]:
    """
    Build precision/recall/lift stats for given K values.
    """
    y_true = y_true.astype(int)
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    order = np.argsort(-y_score)
    results = []
    for k in ks:
        if k <= 0:
            continue
        top_k = order[:k]
        tp = int(y_true[top_k].sum())
        precision = tp / min(k, len(y_true))
        recall = tp / int(y_true.sum()) if y_true.sum() else 0.0
        lift = (precision / base_rate) if base_rate > 0 else 0.0
        results.append(
            {
                "k": k,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }
        )
    return results


def main() -> int:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = pick_default_dataset_path(project_dir)

    parser = argparse.ArgumentParser(description="Unsupervised anomaly scoring for IUU risk.")
    parser.add_argument(
        "--data",
        default=default_data,
        help="Path to vessel_features_*.csv (default: most recent in ./outputs/).",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=5,
        help="Number of most extreme features to average for anomaly score.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Run the builder first:\n"
            "  . .venv/bin/activate\n"
            '  python -u "Senior Project.py"\n'
        )

    df = pd.read_csv(args.data, low_memory=False)
    if "mmsi" not in df.columns:
        raise ValueError("Expected column 'mmsi' in the dataset.")

    # Features: all numeric columns except IDs and labels
    feature_df = df.drop(columns=[c for c in ["mmsi", "is_iuu", "imo"] if c in df.columns]).copy()
    X = feature_df.select_dtypes(include="number")
    if X.shape[1] > 0:
        non_all_nan = ~X.isna().all(axis=0)
        X = X.loc[:, non_all_nan]

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found to score.")

    # Median impute
    X_np = X.to_numpy(dtype=float)
    med = np.nanmedian(X_np, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_np = np.where(np.isnan(X_np), med, X_np)

    # Compute anomaly scores
    scores = compute_anomaly_score(X_np, top_k=args.top_k_features)

    risk_df = pd.DataFrame(
        {
            "mmsi": df["mmsi"],
            "anomaly_score": scores,
            "is_iuu": df["is_iuu"] if "is_iuu" in df.columns else 0,
        }
    )

    # Save top risky vessels (global)
    outputs_dir = os.path.join(project_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    top = risk_df.nlargest(25, "anomaly_score")
    top_path = os.path.join(outputs_dir, "top_risky_vessels_anomaly.csv")
    top[["mmsi", "anomaly_score"]].to_csv(top_path, index=False)

    print(f"Saved anomaly top-risk list to: {top_path}")
    print(top[["mmsi", "is_iuu", "anomaly_score"]].to_string(index=False))

    # Ranking metrics on full dataset
    if "is_iuu" in df.columns:
        print("\n## Anomaly ranking metrics (Full dataset)")
        print(f"Base rate: {df['is_iuu'].mean():.6f}")
        for row in rank_report(df["is_iuu"].values, scores, [10, 25, 50, 100, 250, 500, 1000]):
            print(
                f"Top-{row['k']} Precision: {row['precision']:.4f} "
                f"(TP={row['tp']}), Recall: {row['recall']:.4f}, Lift: {row['lift']:.2f}"
            )

    # Gulf+Med top-risk list if available
    if "in_target_region" in df.columns:
        region_mask = df["in_target_region"].fillna(0).astype(int) == 1
        if region_mask.any():
            top_region = risk_df.loc[region_mask].nlargest(25, "anomaly_score")
            top_region_path = os.path.join(outputs_dir, "top_risky_vessels_gulf_med_anomaly.csv")
            top_region[["mmsi", "anomaly_score"]].to_csv(top_region_path, index=False)
            print(f"\nSaved Gulf+Med anomaly top-risk list to: {top_region_path}")
            print(top_region[["mmsi", "is_iuu", "anomaly_score"]].to_string(index=False))
        else:
            print("\nNo rows flagged as in_target_region; skipping Gulf+Med anomaly list.")

    # Simple diagnostic: how many IUU vessels in top 25?
    if "is_iuu" in df.columns:
        iuu_in_top = int(top["is_iuu"].sum())
        total_iuu = int(df["is_iuu"].sum())
        print(f"\nIUU vessels in top 25: {iuu_in_top} / {total_iuu} total IUU vessels")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
