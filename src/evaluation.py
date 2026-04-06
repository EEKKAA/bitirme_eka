"""
Evaluation module with Stratified Group K-fold cross-validation.
Handles MinMax scaling, feature selection, and threshold optimization
inside each fold to prevent data leakage.

Uses StratifiedGroupKFold to ensure all years of a company stay
in the same fold (prevents company-level information leakage).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

from sklearn.model_selection import train_test_split
from sklearn.base import clone as sklearn_clone

from config import CV_FOLDS, RANDOM_STATE, FEATURE_SELECTION_K, RANDOMIZED_N_ITER
from src.feature_selection import select_features_freg
from src.preprocessing import winsorize_from_train, impute_from_train


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Computes all evaluation metrics for a single fold."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = np.nan

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def _optimize_threshold(y_true, y_prob) -> tuple:
    """Sweep thresholds to find F1-optimal decision boundary."""
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.30, 0.71, 0.02):
        y_pred_t = (y_prob >= t).astype(int)
        f1_t = f1_score(y_true, y_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_t, best_f1 = t, f1_t
    return best_t, best_f1


def cross_validate_model(model, param_grid, X: pd.DataFrame,
                         y: pd.Series, groups: pd.Series = None,
                         model_name: str = "") -> dict:
    """
    Performs Stratified Group K-fold cross-validation with proper data
    handling inside each fold (winsorization, feature selection, scaling).

    Args:
        model: Base estimator instance.
        param_grid: Hyperparameter grid for RandomizedSearchCV.
        X: Feature matrix (all candidate features).
        y: Target vector (bankruptcy_label).
        groups: Company identifiers for group-aware splitting.
        model_name: Display name for logging.

    Returns:
        Dict with keys: 'fold_metrics', 'mean_metrics', 'std_metrics',
                        'best_params', 'best_model', 'selected_features',
                        'optimal_threshold'
    """
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)
        split_iter = skf.split(X, y, groups=groups)
        print(f"    StratifiedGroupKFold (company-level splits)")
    else:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        split_iter = skf.split(X, y)
        print(f"    StratifiedKFold (observation-level splits)")

    fold_metrics = []
    fold_metrics_opt = []  # with optimized threshold
    best_model_overall = None
    best_auc_overall = -1
    best_params_overall = {}
    selected_features_overall = []
    optimal_thresholds = []

    # Collect OOF (out-of-fold) predictions for global threshold optimization
    oof_y_true = []
    oof_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, 1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # 0. Impute remaining NaN with train-only global median (no leakage)
        X_train, X_test = impute_from_train(X_train, X_test)

        # 1. Winsorize inside fold (bounds from train only)
        X_train, X_test = winsorize_from_train(X_train, X_test)

        # 2. Feature selection on training data
        sel_features, f_scores, _ = select_features_freg(
            X_train, y_train, k=FEATURE_SELECTION_K
        )
        X_train = X_train[sel_features]
        X_test = X_test[sel_features]

        # 3. MinMax scaling (fit on train, transform both)
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=sel_features, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=sel_features, index=X_test.index
        )

        # 4. Hyperparameter tuning with RandomizedSearchCV
        from sklearn.base import clone
        model_clone = clone(model)

        if param_grid:
            # Count total combinations to decide search strategy
            from functools import reduce
            import operator
            n_combos = reduce(operator.mul,
                              [len(v) for v in param_grid.values()], 1)
            n_iter = min(RANDOMIZED_N_ITER, n_combos)

            if n_iter >= n_combos:
                # Exhaustive search if grid is small
                from sklearn.model_selection import GridSearchCV
                search = GridSearchCV(
                    model_clone, param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=RANDOM_STATE),
                    scoring="roc_auc",
                    n_jobs=-1,
                    refit=True,
                )
            else:
                search = RandomizedSearchCV(
                    model_clone, param_grid,
                    n_iter=n_iter,
                    cv=StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=RANDOM_STATE),
                    scoring="roc_auc",
                    n_jobs=-1,
                    refit=True,
                    random_state=RANDOM_STATE,
                )
            search.fit(X_train_scaled, y_train)
            fold_model = search.best_estimator_
            fold_params = search.best_params_
        else:
            model_clone.fit(X_train_scaled, y_train)
            fold_model = model_clone
            fold_params = {}

        # 5. Predict on test set
        y_pred = fold_model.predict(X_test_scaled)
        if hasattr(fold_model, "predict_proba"):
            y_prob = fold_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        # 6. Compute metrics (default threshold = 0.5)
        fold_result = _compute_metrics(y_test, y_pred, y_prob)
        fold_result["fold"] = fold_idx
        fold_result["best_params"] = fold_params
        fold_metrics.append(fold_result)

        # Collect OOF predictions
        oof_y_true.extend(y_test.values)
        oof_y_prob.extend(y_prob)

        # 7. Threshold optimization on VALIDATION split (unbiased)
        #    Train a separate model on 80% of training, find optimal threshold
        #    on the held-out 20% validation, then apply to test set.
        try:
            X_tr_inner, X_val_th, y_tr_inner, y_val_th = train_test_split(
                X_train_scaled, y_train, test_size=0.2,
                stratify=y_train, random_state=RANDOM_STATE
            )
            threshold_model = sklearn_clone(model)
            threshold_model.set_params(**fold_params)
            threshold_model.fit(X_tr_inner, y_tr_inner)
            y_val_prob = threshold_model.predict_proba(X_val_th)[:, 1]
            opt_t, _ = _optimize_threshold(y_val_th, y_val_prob)
        except (ValueError, Exception):
            # Fallback if validation split fails (e.g., too few samples)
            opt_t = 0.5

        y_pred_opt = (y_prob >= opt_t).astype(int)
        fold_result_opt = _compute_metrics(y_test, y_pred_opt, y_prob)
        fold_result_opt["fold"] = fold_idx
        fold_result_opt["threshold"] = opt_t
        fold_metrics_opt.append(fold_result_opt)
        optimal_thresholds.append(opt_t)

        # Track best model across folds
        fold_auc = fold_result.get("auc", 0)
        if fold_auc > best_auc_overall:
            best_auc_overall = fold_auc
            best_model_overall = fold_model
            best_params_overall = fold_params
            selected_features_overall = sel_features

        print(f"    Fold {fold_idx:2d}: AUC={fold_result['auc']:.4f}  "
              f"F1={fold_result['f1']:.4f}  MCC={fold_result['mcc']:.4f}  "
              f"(opt_t={opt_t:.2f} -> F1={fold_result_opt['f1']:.4f})")

    # Aggregate metrics (default threshold)
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
    mean_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics if not np.isnan(m.get(key, np.nan))]
        mean_metrics[key] = np.mean(values) if values else np.nan
        std_metrics[key] = np.std(values) if values else np.nan

    # Aggregate metrics (optimized threshold)
    mean_metrics_opt = {}
    std_metrics_opt = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics_opt if not np.isnan(m.get(key, np.nan))]
        mean_metrics_opt[key] = np.mean(values) if values else np.nan
        std_metrics_opt[key] = np.std(values) if values else np.nan

    avg_threshold = np.mean(optimal_thresholds)

    # ── Global OOF threshold optimization ─────────────────────────────────
    # Uses ALL out-of-fold predictions (unbiased, full dataset coverage)
    oof_y_true = np.array(oof_y_true)
    oof_y_prob = np.array(oof_y_prob)
    oof_threshold, oof_f1 = _optimize_threshold(oof_y_true, oof_y_prob)
    oof_y_pred = (oof_y_prob >= oof_threshold).astype(int)
    oof_metrics = _compute_metrics(oof_y_true, oof_y_pred, oof_y_prob)

    print(f"\n    OOF Global Threshold: {oof_threshold:.2f} "
          f"(F1={oof_metrics['f1']:.4f}, AUC={oof_metrics['auc']:.4f}, "
          f"MCC={oof_metrics['mcc']:.4f})")

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_metrics_opt": fold_metrics_opt,
        "mean_metrics_opt": mean_metrics_opt,
        "std_metrics_opt": std_metrics_opt,
        "optimal_threshold": avg_threshold,
        "oof_threshold": oof_threshold,
        "oof_metrics": oof_metrics,
        "oof_predictions": {"y_true": oof_y_true, "y_prob": oof_y_prob},
        "best_params": best_params_overall,
        "best_model": best_model_overall,
        "selected_features": selected_features_overall,
    }
