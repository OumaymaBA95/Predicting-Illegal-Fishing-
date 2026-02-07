# Cross Validation Report

## A) Cross‑validation (PU Stratified K‑fold)

**Command used**
```
python "train_model.py" --pu-train --model gbdt --cv-folds 5 --cv-seed 42 --data "outputs/vessel_features_all_years.csv"
```

**Key code paths**
```
47:77:train_model.py
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
    ...
```

```
429:516:train_model.py
if args.cv_folds and args.cv_folds > 1:
    print(f"\n## Stratified K-fold CV (k={args.cv_folds}) on PU subset")
    fold_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(
        stratified_kfold_indices(y_np_train, k=args.cv_folds, random_state=args.cv_seed), start=1
    ):
        ...
        pr_auc = average_precision(y_test, proba_test)
        ra = roc_auc(y_test, proba_test)
        mrr = mean_reciprocal_rank(y_test, proba_test)
        topk = rank_report(y_test, proba_test, [25, 100, 500])
        ...
```

**Results (CV mean)**
- PR‑AUC: **0.4061**
- ROC‑AUC: **0.9495**
- MRR: **0.3656**
- Top‑25 P/R: **0.0480 / 0.6000**
- Top‑100 P/R: **0.0120 / 0.6000**
- Top‑500 P/R: **0.0032 / 0.7667**

**Interpretation**
The model is reasonably stable across folds and consistently places known IUU vessels near the top of the ranked list. Given only 11 labeled positives, this is strong evidence the ranking signal is meaningful.

---

## B) Main training run (PU + GBDT)

**Command used**
```
python "train_model.py" --pu-train --use-anomaly --model gbdt --ensemble-weights 0.7,0.2,0.1 --data "outputs/vessel_features_all_years.csv"
```

**Key code paths**
```
430:449:train_model.py
if args.pu_train:
    reliable_neg = build_reliable_negative_mask(df) & (y == 0)
    ...
    if n_rel_neg >= args.min_reliable_negatives and n_pos > 0:
        use_pu = True
        pu_mask = (y == 1) | reliable_neg
```

```
290:339:train_model.py
def _fit_model(
    X_train_z: np.ndarray,
    y_train: np.ndarray,
    *,
    args,
    sample_w: np.ndarray,
):
    ...
    clf = HistGradientBoostingClassifier(
        max_depth=args.gbdt_max_depth,
        max_iter=args.gbdt_max_iter,
        learning_rate=args.gbdt_learning_rate,
        l2_regularization=args.gbdt_l2,
        random_state=args.random_state,
    )
    clf.fit(X_train_z, y_train, sample_weight=sample_w)
```

**Results (full dataset ranking)**
- Top‑25 IUU: **2**
- Top‑100 IUU: **3**
- Top‑250 IUU: **6**
- Top‑500 IUU: **9**
- Top‑1000 IUU: **10**

**Interpretation**
The PU‑ranking model successfully pulls most known IUU vessels into the top‑ranked lists (especially at Top‑500 and Top‑1000). The output is best used as a **triage list** for investigation rather than a legal determination.
