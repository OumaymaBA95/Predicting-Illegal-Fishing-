import argparse
import os
import glob

import pandas as pd
import numpy as np
try:
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
except Exception:
    HistGradientBoostingClassifier = None
    IsolationForest = None


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

def stratified_split(y: np.ndarray, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    y = y.astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.arange(len(y)), np.array([], dtype=int)

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_test = max(1, int(round(len(pos_idx) * test_size)))
    n_neg_test = max(1, int(round(len(neg_idx) * test_size)))

    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def stratified_kfold_indices(y: np.ndarray, k: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-fold indices for binary labels.
    Returns list of (train_idx, test_idx).
    """
    if k <= 1:
        return []
    rng = np.random.default_rng(random_state)
    y = y.astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    all_idx = np.arange(len(y))
    folds = []
    for i in range(k):
        test_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        rng.shuffle(test_idx)
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        folds.append((train_idx, test_idx))
    return folds


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)  # numerical stability
    return 1.0 / (1.0 + np.exp(-z))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # AP = mean precision at ranks where y_true==1
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return 0.0
    tp_cum = np.cumsum(y_sorted)
    precision_at_k = tp_cum / (np.arange(len(y_sorted)) + 1)
    return float(precision_at_k[y_sorted == 1].mean())


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Rank-based AUC with average ranks for ties.
    y_true = y_true.astype(int)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()  # ascending ranks starting at 1
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def top_k_precision(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Calculate precision at top K (how many positives in top K predictions)."""
    if k <= 0:
        return 0.0
    order = np.argsort(-y_score)  # descending order
    top_k = order[:k]
    y_top_k = y_true[top_k]
    return float(y_top_k.sum() / min(k, len(y_true)))


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


def format_rank_table(rows: list[dict]) -> str:
    header = "| K | TP | Precision | Recall | Lift |\n|---:|---:|---:|---:|---:|"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['tp']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['lift']:.2f} |"
        )
    return "\n".join(lines)


def mean_reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for positive examples."""
    y_true = y_true.astype(int)
    pos_idx = np.where(y_true == 1)[0]
    if len(pos_idx) == 0:
        return 0.0
    order = np.argsort(-y_score)  # descending order
    ranks = {val: rank + 1 for rank, val in enumerate(order)}
    reciprocal_ranks = [1.0 / ranks[idx] for idx in pos_idx]
    return float(np.mean(reciprocal_ranks))


def compute_rule_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based risk score baseline (interpretable).
    Uses proxy signals that do not require large label coverage.
    """
    idx = df.index
    def _col(name: str, default=0):
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=idx)

    # Tightened thresholds to keep the rule-score more selective
    near_eez = _col("near_eez_boundary_50km", 0).fillna(0).astype(int)
    sar_hits = (_col("sar_manual_bin_hits_at_region_mean", 0).fillna(0) >= 2).astype(int)
    flag_risk = (pd.to_numeric(_col("flag_risk_score", 0), errors="coerce").fillna(0) >= 4.5).astype(int)
    if "region_hours_fraction" in df.columns:
        region_focus = (pd.to_numeric(df["region_hours_fraction"], errors="coerce").fillna(0) >= 0.3).astype(int)
    else:
        region_focus = (pd.to_numeric(_col("region_fishing_hours_sum", 0), errors="coerce").fillna(0) >= 50).astype(int)
    gap_risk = (pd.to_numeric(_col("gap_hours_mean", 0), errors="coerce").fillna(0) >= 12).astype(int)

    rule_score = (
        (2 * near_eez)
        + (3 * sar_hits)
        + (2 * flag_risk)
        + (1 * region_focus)
        + (2 * gap_risk)
    )

    return pd.DataFrame(
        {
            "rule_near_eez": near_eez,
            "rule_sar_hits": sar_hits,
            "rule_flag_risk_high": flag_risk,
            "rule_region_focus": region_focus,
            "rule_gap_risk": gap_risk,
            "rule_risk_score": rule_score,
        },
        index=idx,
    )


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """
    Rank positions (1 = highest score).
    """
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def ensemble_from_ranks(score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
    """
    Combine two score vectors by averaging their rank positions.
    Returns a score in [0,1] where higher is better.
    """
    ra = rank_positions(score_a)
    rb = rank_positions(score_b)
    combined_rank = (ra + rb) / 2.0
    if len(combined_rank) <= 1:
        return np.ones_like(combined_rank, dtype=float)
    return 1.0 - (combined_rank - 1.0) / (len(combined_rank) - 1.0)


def ensemble_from_ranks_multi(scores: list[np.ndarray]) -> np.ndarray:
    """
    Combine multiple score vectors by averaging their rank positions.
    Returns a score in [0,1] where higher is better.
    """
    if not scores:
        return np.array([])
    ranks = [rank_positions(s) for s in scores]
    combined_rank = np.mean(np.stack(ranks, axis=0), axis=0)
    if len(combined_rank) <= 1:
        return np.ones_like(combined_rank, dtype=float)
    return 1.0 - (combined_rank - 1.0) / (len(combined_rank) - 1.0)


def weighted_ensemble_from_ranks(scores: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Weighted rank-averaging ensemble. Weights are normalized to sum to 1.
    """
    if not scores:
        return np.array([])
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length.")
    weights = np.array(weights, dtype=float)
    if np.all(weights <= 0):
        weights = np.ones_like(weights, dtype=float)
    weights = weights / weights.sum()
    ranks = [rank_positions(s) for s in scores]
    combined_rank = np.average(np.stack(ranks, axis=0), axis=0, weights=weights)
    if len(combined_rank) <= 1:
        return np.ones_like(combined_rank, dtype=float)
    return 1.0 - (combined_rank - 1.0) / (len(combined_rank) - 1.0)


def build_reliable_negative_mask(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic filter for "reliable negatives" (low-risk vessels).
    This supports PU-style learning when positives are sparse.
    """
    idx = df.index
    def _num(name: str, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series([default] * len(df), index=idx)

    rule_score = _num("rule_risk_score", 0.0)
    flag_risk = _num("flag_risk_score", 0.0)
    gap_hours = _num("gap_hours_mean", 0.0)
    near_eez = _num("near_eez_boundary_50km", 0.0)
    dist_eez = _num("dist_to_eez_km", 0.0)
    sar_hits = _num("sar_manual_bin_hits_at_region_mean", 0.0)
    region_frac = _num("region_hours_fraction", 0.0)
    region_hours = _num("region_fishing_hours_sum", 0.0)
    all_hours = _num("all_fishing_hours_sum", 0.0)
    in_target = _num("in_target_region", 0.0)

    # Conservative "definitely low-risk" filter, tuned for Gulf/Mediterranean focus.
    region_focus = (region_frac >= 0.03) | (region_hours >= 10)
    far_from_eez = (dist_eez > 100) if "dist_to_eez_km" in df.columns else (near_eez <= 0)

    reliable = (
        (rule_score <= 1)
        & (flag_risk <= 2.5)
        & (gap_hours < 2)
        & (far_from_eez)
        & (sar_hits <= 0)
        & (~region_focus)
        & (in_target <= 0)
        & (all_hours > 0)
    )
    return reliable.astype(bool)

def train_weighted_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 3000,
    lr: float = 0.1,
    l2: float = 1.0,
    random_state: int = 42,
    pos_weight_ratio: float = 0.95,
) -> np.ndarray:
    """
    Train a weighted logistic regression (L2) with gradient descent.
    Weights are set to heavily favor positive class to handle extreme imbalance.
    pos_weight_ratio: fraction of total loss from positive class (default 0.95 for 95/5 split).
    Returns weight vector including bias term as w[0].
    """
    rng = np.random.default_rng(random_state)
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    # Add bias column
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    w = rng.normal(scale=0.01, size=(Xb.shape[1],))

    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    # Heavily weight positive class to handle extreme imbalance
    w_pos = pos_weight_ratio / n_pos
    w_neg = (1.0 - pos_weight_ratio) / n_neg
    sample_w = np.where(y == 1, w_pos, w_neg)

    for _ in range(steps):
        p = sigmoid(Xb @ w)
        err = (p - y) * sample_w
        grad = (Xb.T @ err)
        # L2 (exclude bias)
        grad[1:] += l2 * w[1:]
        w -= lr * grad

    return w


def compute_sample_weights(y: np.ndarray, pos_weight_ratio: float) -> np.ndarray:
    """
    Compute per-row weights so the positive class contributes pos_weight_ratio of total loss.
    """
    y = y.astype(int)
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    w_pos = pos_weight_ratio / n_pos
    w_neg = (1.0 - pos_weight_ratio) / n_neg
    return np.where(y == 1, w_pos, w_neg)


def _collapse_rare_categories(series: pd.Series, *, top_k: int = 50) -> pd.Series:
    """
    Keep only the top_k most common categories; collapse others to 'other'.
    """
    vc = series.value_counts(dropna=False)
    keep = set(vc.head(top_k).index.astype(str))
    return series.apply(lambda v: v if str(v) in keep else "other")


def _fit_model(
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    *,
    args,
    sample_w: np.ndarray,
):
    if args.model == "logreg":
        w = train_weighted_logreg(
            X_train_z,
            y_train,
            steps=4000,
            lr=0.05,
            l2=0.5,
            random_state=args.random_state,
            pos_weight_ratio=args.pos_weight_ratio,
        )

        def predict_proba(Xz: np.ndarray) -> np.ndarray:
            Xb = np.concatenate([np.ones((Xz.shape[0], 1)), Xz], axis=1)
            return sigmoid(Xb @ w)
    else:
        if HistGradientBoostingClassifier is None:
            raise ImportError(
                "scikit-learn is required for --model gbdt. "
                "Install it with: pip install scikit-learn"
            )
        clf = HistGradientBoostingClassifier(
            max_depth=args.gbdt_max_depth,
            max_iter=args.gbdt_max_iter,
            learning_rate=args.gbdt_learning_rate,
            l2_regularization=args.gbdt_l2,
            random_state=args.random_state,
        )
        clf.fit(X_train_z, y_train, sample_weight=sample_w)

        def predict_proba(Xz: np.ndarray) -> np.ndarray:
            return clf.predict_proba(Xz)[:, 1]

    return predict_proba


def main() -> int:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = pick_default_dataset_path(project_dir)

    parser = argparse.ArgumentParser(description="Train a baseline IUU risk model.")
    parser.add_argument(
        "--data",
        default=default_data,
        help="Path to vessel_features_*.csv (default: most recent in ./outputs/).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--pos-weight-ratio",
        type=float,
        default=0.95,
        help="Fraction of loss from positive class (0.95 = 95%% positive, 5%% negative). Higher values help with extreme imbalance.",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "gbdt"],
        default="logreg",
        help="Model type: logreg (weighted logistic regression) or gbdt (sklearn HistGradientBoosting).",
    )
    parser.add_argument(
        "--pu-train",
        action="store_true",
        help="Enable PU-style training using reliable negatives (recommended for small IUU labels).",
    )
    parser.add_argument(
        "--min-reliable-negatives",
        type=int,
        default=200,
        help="Minimum reliable negatives required for PU training; otherwise fallback to standard training.",
    )
    parser.add_argument(
        "--use-anomaly",
        action="store_true",
        help="Add an IsolationForest anomaly score to the ensemble (unsupervised).",
    )
    parser.add_argument(
        "--anomaly-contamination",
        type=float,
        default=0.01,
        help="Estimated anomaly fraction for IsolationForest (default 0.01).",
    )
    parser.add_argument(
        "--ensemble-weights",
        default="0.7,0.2,0.1",
        help="Comma-separated weights for [supervised, anomaly, rule] scores.",
    )
    parser.add_argument("--gbdt-max-depth", type=int, default=8)
    parser.add_argument("--gbdt-max-iter", type=int, default=500)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.03)
    parser.add_argument("--gbdt-l2", type=float, default=0.0)
    parser.add_argument("--cv-folds", type=int, default=0, help="Run stratified K-fold CV on PU subset.")
    parser.add_argument("--cv-seed", type=int, default=42, help="Random seed for CV splits.")
    parser.add_argument(
        "--stability-seeds",
        default="42,7,13",
        help="Comma-separated seeds for stability check (GBDT only).",
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
    if "mmsi" not in df.columns or "is_iuu" not in df.columns:
        raise ValueError("Expected columns 'mmsi' and 'is_iuu' in the dataset.")

    y = df["is_iuu"].astype(int)
    counts = y.value_counts(dropna=False).to_dict()
    print(f"Loaded dataset: {args.data}")
    print(f"Rows: {len(df)}")
    print(f"Label counts: {counts}")

    # Rule-based baseline score (interpretable risk)
    rule_df = compute_rule_score(df)
    df = pd.concat([df, rule_df], axis=1)

    if y.nunique() < 2:
        print(
            "\nCannot train a classifier because only one class is present.\n"
            "This usually means your IUU labels didn't match any MMSI in your GFW data.\n"
            "Next things to try:\n"
            "- Verify MMSI normalization (digits-only) is consistent across datasets\n"
            "- Confirm the IUU list contains MMSIs for vessels in your chosen regions/years\n"
            "- Expand regions/years temporarily to validate label matching\n"
        )
        return 0

    # Features: all numeric columns except the target.
    feature_df = df.drop(columns=["is_iuu"]).copy()
    if "mmsi" in feature_df.columns:
        feature_df = feature_df.drop(columns=["mmsi"])
    if "imo" in feature_df.columns:
        feature_df = feature_df.drop(columns=["imo"])

    # Optional categorical features (one-hot)
    cat_cols = [
        c for c in [
            "flag_mode",
            "geartype_mode",
            "flag_best",
            "fv_flag_gfw",
            "fv_flag_registry",
            "fv_flag_ais",
            "fv_vessel_class_gfw",
        ] if c in feature_df.columns
    ]

    cat_df = pd.DataFrame(index=feature_df.index)
    for col in cat_cols:
        s = feature_df[col].fillna("unknown").astype(str).str.strip()
        s = _collapse_rare_categories(s, top_k=50)
        dummies = pd.get_dummies(s, prefix=col, dtype=float)
        cat_df = pd.concat([cat_df, dummies], axis=1)

    X_num = feature_df.select_dtypes(include="number")
    X = pd.concat([X_num, cat_df], axis=1)

    # Drop columns that are entirely NaN (can happen with optional features)
    if X.shape[1] > 0:
        non_all_nan = ~X.isna().all(axis=0)
        X = X.loc[:, non_all_nan]

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found to train on.")

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)

    # PU-style training subset (positives + reliable negatives)
    use_pu = False
    pu_mask = None
    if args.pu_train:
        reliable_neg = build_reliable_negative_mask(df) & (y == 0)
        n_pos = int((y == 1).sum())
        n_rel_neg = int(reliable_neg.sum())
        print(f"PU training requested. Positives: {n_pos}, reliable negatives: {n_rel_neg}")
        if n_rel_neg >= args.min_reliable_negatives and n_pos > 0:
            use_pu = True
            pu_mask = (y == 1) | reliable_neg
        else:
            print(
                "Not enough reliable negatives for PU training; falling back to standard training."
            )

    if use_pu and pu_mask is not None:
        X_np_train = X_np[pu_mask.to_numpy()]
        y_np_train = y_np[pu_mask.to_numpy()]
    else:
        X_np_train = X_np
        y_np_train = y_np

    if args.cv_folds and args.cv_folds > 1:
        print(f"\n## Stratified K-fold CV (k={args.cv_folds}) on PU subset")
        fold_metrics = []
        for fold_i, (train_idx, test_idx) in enumerate(
            stratified_kfold_indices(y_np_train, k=args.cv_folds, random_state=args.cv_seed), start=1
        ):
            if len(test_idx) == 0:
                continue
            X_train = X_np_train[train_idx]
            X_test = X_np_train[test_idx]
            y_train = y_np_train[train_idx]
            y_test = y_np_train[test_idx]

            med = np.nanmedian(X_train, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            X_train = np.where(np.isnan(X_train), med, X_train)
            X_test = np.where(np.isnan(X_test), med, X_test)

            mu = X_train.mean(axis=0)
            sigma = X_train.std(axis=0)
            sigma = np.where(sigma == 0, 1.0, sigma)
            X_train_z = (X_train - mu) / sigma
            X_test_z = (X_test - mu) / sigma

            sample_w = compute_sample_weights(y_train, args.pos_weight_ratio)
            predict_proba = _fit_model(X_train_z, y_train, args=args, sample_w=sample_w)

            proba_test = predict_proba(X_test_z)
            pr_auc = average_precision(y_test, proba_test)
            ra = roc_auc(y_test, proba_test)
            mrr = mean_reciprocal_rank(y_test, proba_test)
            topk = rank_report(y_test, proba_test, [25, 100, 500])
            top25 = next((r for r in topk if r["k"] == 25), {"precision": 0.0, "recall": 0.0})
            top100 = next((r for r in topk if r["k"] == 100), {"precision": 0.0, "recall": 0.0})
            top500 = next((r for r in topk if r["k"] == 500), {"precision": 0.0, "recall": 0.0})
            fold_metrics.append(
                {
                    "pr_auc": pr_auc,
                    "roc_auc": ra,
                    "mrr": mrr,
                    "top25_p": top25["precision"],
                    "top25_r": top25["recall"],
                    "top100_p": top100["precision"],
                    "top100_r": top100["recall"],
                    "top500_p": top500["precision"],
                    "top500_r": top500["recall"],
                }
            )
            print(
                f"Fold {fold_i}: PR-AUC={pr_auc:.4f}, ROC-AUC={ra:.4f}, "
                f"MRR={mrr:.4f}, Top-25 P/R={top25['precision']:.4f}/{top25['recall']:.4f}"
            )

        if fold_metrics:
            df_cv = pd.DataFrame(fold_metrics)
            avg = df_cv.mean(numeric_only=True)
            print(
                "\nCV mean: "
                f"PR-AUC={avg['pr_auc']:.4f}, ROC-AUC={avg['roc_auc']:.4f}, "
                f"MRR={avg['mrr']:.4f}, "
                f"Top-25 P/R={avg['top25_p']:.4f}/{avg['top25_r']:.4f}, "
                f"Top-100 P/R={avg['top100_p']:.4f}/{avg['top100_r']:.4f}, "
                f"Top-500 P/R={avg['top500_p']:.4f}/{avg['top500_r']:.4f}"
            )

    train_idx, test_idx = stratified_split(y_np_train, test_size=args.test_size, random_state=args.random_state)
    if len(test_idx) == 0:
        print("Could not create a stratified test split; skipping metrics.")
        return 0

    X_train = X_np_train[train_idx]
    X_test = X_np_train[test_idx]
    y_train = y_np_train[train_idx]
    y_test = y_np_train[test_idx]

    # Median impute using train stats
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_train = np.where(np.isnan(X_train), med, X_train)
    X_test = np.where(np.isnan(X_test), med, X_test)

    # Standardize using train stats
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    X_train_z = (X_train - mu) / sigma
    X_test_z = (X_test - mu) / sigma

    sample_w = compute_sample_weights(y_train, args.pos_weight_ratio)
    predict_proba = _fit_model(X_train_z, y_train, args=args, sample_w=sample_w)

    proba_test = predict_proba(X_test_z)
    pred_test = (proba_test >= 0.5).astype(int)

    pr_auc = average_precision(y_test, proba_test)
    ra = roc_auc(y_test, proba_test)
    mrr = mean_reciprocal_rank(y_test, proba_test)

    tp = int(((pred_test == 1) & (y_test == 1)).sum())
    fp = int(((pred_test == 1) & (y_test == 0)).sum())
    fn = int(((pred_test == 0) & (y_test == 1)).sum())
    tn = int(((pred_test == 0) & (y_test == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print(f"\nModel: {args.model}")
    if use_pu:
        print("Training mode: PU (positives + reliable negatives)")
    else:
        print("Training mode: Standard (labeled positives/negatives)")
    print("## Test metrics (threshold=0.5)")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {ra:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"MRR:       {mrr:.4f}")
    print("Confusion matrix [[tn fp]\n [fn tp]]:")
    print(np.array([[tn, fp], [fn, tp]]))
    
    print("\n## Ranking metrics (Test split)")
    print(f"Base rate: {y_test.mean():.6f}")
    for row in rank_report(y_test, proba_test, [10, 25, 50, 100, 250, 500]):
        print(
            f"Top-{row['k']} Precision: {row['precision']:.4f} "
            f"(TP={row['tp']}), Recall: {row['recall']:.4f}, Lift: {row['lift']:.2f}"
        )

    # PU-style evaluation (known positives only; negatives treated as unlabeled)
    print("\n## PU-style evaluation (known positives only)")
    for row in rank_report(y_test, proba_test, [25, 50, 100, 250, 500]):
        print(
            f"Top-{row['k']} Known-Positive Precision: {row['precision']:.4f} "
            f"(TP={row['tp']}), Recall: {row['recall']:.4f}, Lift: {row['lift']:.2f}"
        )

    # Produce a top-risk list across the whole dataset.
    X_all = np.where(np.isnan(X_np), med, X_np)
    X_all_z = (X_all - mu) / sigma
    proba_all = predict_proba(X_all_z)
    
    # Create full risk score dataframe with labels
    risk_df = pd.DataFrame({
        "mmsi": df["mmsi"],
        "risk_score": proba_all,
        "is_iuu": df["is_iuu"],
        "rule_risk_score": df["rule_risk_score"],
    })
    
    # Optional anomaly score (unsupervised)
    anomaly_score = None
    if args.use_anomaly:
        if IsolationForest is None:
            print("IsolationForest unavailable (scikit-learn not installed); skipping anomaly score.")
        else:
            iso = IsolationForest(
                n_estimators=300,
                contamination=args.anomaly_contamination,
                random_state=args.random_state,
            )
            iso.fit(X_all_z)
            anomaly_score = -iso.score_samples(X_all_z)
            risk_df["anomaly_score"] = anomaly_score

    # Ensemble (supervised + rule + optional anomaly) using weighted rank averaging
    def _parse_weights(s: str) -> list[float]:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 3:
            return [0.7, 0.2, 0.1]
        try:
            return [float(p) for p in parts]
        except Exception:
            return [0.7, 0.2, 0.1]

    w_super, w_anom, w_rule = _parse_weights(args.ensemble_weights)
    ensemble_inputs = [
        pd.to_numeric(risk_df["risk_score"], errors="coerce").fillna(0).to_numpy(),
    ]
    ensemble_weights = [w_super]
    if anomaly_score is not None:
        ensemble_inputs.append(pd.to_numeric(risk_df["anomaly_score"], errors="coerce").fillna(0).to_numpy())
        ensemble_weights.append(w_anom)
    else:
        w_rule += w_anom  # reassign anomaly weight to rule if not used
    ensemble_inputs.append(pd.to_numeric(risk_df["rule_risk_score"], errors="coerce").fillna(0).to_numpy())
    ensemble_weights.append(w_rule)
    ensemble_score = weighted_ensemble_from_ranks(ensemble_inputs, ensemble_weights)
    risk_df["ensemble_score"] = ensemble_score

    top = risk_df.nlargest(25, "risk_score")
    
    mrr_full = mean_reciprocal_rank(df["is_iuu"].values, proba_all)
    
    print("\n## Full dataset ranking metrics")
    print(f"Base rate: {df['is_iuu'].mean():.6f}")
    for row in rank_report(df["is_iuu"].values, proba_all, [10, 25, 50, 100, 250, 500, 1000]):
        print(
            f"Top-{row['k']} Precision: {row['precision']:.4f} "
            f"(TP={row['tp']}), Recall: {row['recall']:.4f}, Lift: {row['lift']:.2f}"
        )
    print(f"MRR: {mrr_full:.4f}")
    print(f"IUU vessels in top 25: {top['is_iuu'].sum()} / {df['is_iuu'].sum()} total IUU vessels")

    outputs_dir = os.path.join(project_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    top_path = os.path.join(outputs_dir, "top_risky_vessels.csv")
    top[["mmsi", "risk_score"]].to_csv(top_path, index=False)
    print(f"\nSaved top-risk list to: {top_path}")
    print(top[["mmsi", "is_iuu", "risk_score"]].to_string(index=False))

    # PU-only ranking list (main deliverable for small-label setting)
    if use_pu:
        pu_path = os.path.join(outputs_dir, "top_risky_vessels_pu.csv")
        risk_df[["mmsi", "risk_score"]].sort_values("risk_score", ascending=False).head(1000).to_csv(
            pu_path, index=False
        )
        print(f"Saved PU-only top-risk list to: {pu_path}")

    # Ensemble ranking (full dataset)
    print("\n## Ensemble ranking (Full dataset)")
    for row in rank_report(df["is_iuu"].values, ensemble_score, [10, 25, 50, 100, 250, 500, 1000]):
        print(
            f"Top-{row['k']} Precision: {row['precision']:.4f} "
            f"(TP={row['tp']}), Recall: {row['recall']:.4f}, Lift: {row['lift']:.2f}"
        )
    ensemble_path = os.path.join(outputs_dir, "top_risky_vessels_ensemble.csv")
    risk_df[["mmsi", "ensemble_score"]].sort_values("ensemble_score", ascending=False).head(1000).to_csv(
        ensemble_path, index=False
    )
    print(f"Saved ensemble top-risk list to: {ensemble_path}")

    # Baseline rule-score ranking (full dataset)
    print("\n## Baseline rule-score ranking (Full dataset)")
    rule_scores = pd.to_numeric(df["rule_risk_score"], errors="coerce").fillna(0).to_numpy()
    for row in rank_report(df["is_iuu"].values, rule_scores, [10, 25, 50, 100, 250, 500, 1000]):
        print(
            f"Top-{row['k']} Precision: {row['precision']:.4f} "
            f"(TP={row['tp']}), Recall: {row['recall']:.4f}, Lift: {row['lift']:.2f}"
        )

    # Produce a top-risk list restricted to the target regions (Gulf+Mediterranean),
    # if the dataset provides the flag.
    if "in_target_region" in df.columns:
        region_mask = df["in_target_region"].fillna(0).astype(int) == 1
        if region_mask.any():
            top_region = (
                pd.DataFrame(
                    {
                        "mmsi": df.loc[region_mask, "mmsi"],
                        "risk_score": proba_all[region_mask.to_numpy()],
                    }
                )
                .sort_values("risk_score", ascending=False)
                .head(25)
            )
            top_region_path = os.path.join(outputs_dir, "top_risky_vessels_gulf_med.csv")
            top_region.to_csv(top_region_path, index=False)
            print(f"\nSaved Gulf+Med top-risk list to: {top_region_path}")
            print(top_region.to_string(index=False))
        else:
            print("\nNo rows flagged as in_target_region; skipping Gulf+Med top-risk list.")

    # Stability check (GBDT only)
    if args.model == "gbdt" and args.stability_seeds:
        seeds = [int(s.strip()) for s in args.stability_seeds.split(",") if s.strip()]
        if len(seeds) >= 2:
            print("\n## Stability check (GBDT)")
            topk_sets: dict[int, dict[int, set]] = {}
            for seed in seeds:
                clf_s = HistGradientBoostingClassifier(
                    max_depth=args.gbdt_max_depth,
                    max_iter=args.gbdt_max_iter,
                    learning_rate=args.gbdt_learning_rate,
                    l2_regularization=args.gbdt_l2,
                    random_state=seed,
                )
                clf_s.fit(X_train_z, y_train, sample_weight=sample_w)
                proba_all_s = clf_s.predict_proba(X_all_z)[:, 1]
                order = np.argsort(-proba_all_s)
                topk_sets[seed] = {
                    25: set(order[:25]),
                    100: set(order[:100]),
                    1000: set(order[:1000]),
                }
            base_seed = seeds[0]
            for k in [25, 100, 1000]:
                base_set = topk_sets[base_seed][k]
                for seed in seeds[1:]:
                    other = topk_sets[seed][k]
                    inter = len(base_set & other)
                    union = len(base_set | other)
                    jaccard = inter / union if union else 0.0
                    print(f"Top-{k} overlap seed {base_seed} vs {seed}: {inter}/{k}, Jaccard={jaccard:.3f}")

    # Top-K explanation report (interpretable features)
    explain_cols = [
        "mmsi",
        "risk_score",
        "rule_risk_score",
        "anomaly_score",
        "ensemble_score",
        "is_iuu",
        "in_target_region",
        "flag_risk_score",
        "near_eez_boundary_50km",
        "sar_manual_bin_hits_at_region_mean",
        "gap_hours_mean",
        "gap_days_fraction",
        "region_hours_fraction",
        "region_fishing_hours_sum",
        "all_fishing_hours_sum",
    ]
    base_df = df.drop(columns=["mmsi", "is_iuu", "rule_risk_score"], errors="ignore")
    explain_base_cols = ["mmsi", "risk_score", "rule_risk_score", "is_iuu"]
    if "anomaly_score" in risk_df.columns:
        explain_base_cols.append("anomaly_score")
    explain_df = pd.concat([risk_df[explain_base_cols], base_df], axis=1)
    explain_cols = [c for c in explain_cols if c in explain_df.columns]
    explain_df = explain_df.loc[:, explain_cols]
    top_explain = explain_df.sort_values("risk_score", ascending=False).head(100)
    explain_path = os.path.join(outputs_dir, "top_risky_vessels_explanations.csv")
    top_explain.to_csv(explain_path, index=False)
    print(f"\nSaved top-K explanation report to: {explain_path}")

    # PU scoring report + thesis summary table
    report_rows = rank_report(df["is_iuu"].values, proba_all, [10, 25, 50, 100, 250, 500, 1000])
    pu_report = [
        "# PU Scoring Report",
        "",
        f"Dataset: `{os.path.basename(args.data)}`",
        f"Rows: {len(df)}",
        f"Known IUU labels: {int(df['is_iuu'].sum())}",
        f"Training mode: {'PU (positives + reliable negatives)' if use_pu else 'Standard'}",
        "",
        "## Top-K Ranking (Full Dataset)",
        format_rank_table(report_rows),
        "",
        "## Notes",
        "- Risk scores are relative rankings, not probabilities.",
        "- PU training treats unknowns as unlabeled, not true negatives.",
    ]
    pu_report_path = os.path.join(outputs_dir, "pu_scoring_report.md")
    with open(pu_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pu_report))
    print(f"Saved PU scoring report to: {pu_report_path}")

    thesis_table = [
        "# Thesis Summary Table (PU Ranking)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Dataset | {os.path.basename(args.data)} |",
        f"| Rows | {len(df)} |",
        f"| Known IUU labels | {int(df['is_iuu'].sum())} |",
        f"| PR-AUC (test split) | {pr_auc:.4f} |",
        f"| ROC-AUC (test split) | {ra:.4f} |",
        f"| MRR (test split) | {mrr:.4f} |",
        f"| Top-25 IUU (full dataset) | {report_rows[1]['tp']} |",
        f"| Top-100 IUU (full dataset) | {report_rows[3]['tp']} |",
        f"| Top-500 IUU (full dataset) | {report_rows[5]['tp']} |",
        "",
        "These metrics are reported for ranking performance with sparse labels.",
    ]
    thesis_path = os.path.join(project_dir, "THESIS_SUMMARY_TABLE.md")
    with open(thesis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(thesis_table))
    print(f"Saved thesis summary table to: {thesis_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

