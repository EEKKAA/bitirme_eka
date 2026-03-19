"""
Evaluation module with Stratified 10-fold cross-validation.
Handles SMOTE, MinMax scaling, and feature selection inside each fold
to prevent data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from config import CV_FOLDS, RANDOM_STATE, SMOTE_THRESHOLD, FEATURE_SELECTION_K
from src.feature_selection import select_features_freg


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


def cross_validate_model(model, param_grid, X: pd.DataFrame,
                         y: pd.Series, model_name: str = "",
                         force_smote: bool = None) -> dict:
    """
    Performs Stratified K-fold cross-validation with proper data handling
    inside each fold (scaling, SMOTE, feature selection).

    Args:
        model: Base estimator instance.
        param_grid: Hyperparameter grid for GridSearchCV.
        X: Feature matrix (all candidate ratios, preprocessed).
        y: Target vector (bankruptcy_label).
        model_name: Display name for logging.
        force_smote: If True, always apply SMOTE. If False, never apply.
                     If None, use threshold-based logic.

    Returns:
        Dict with keys: 'fold_metrics', 'mean_metrics', 'std_metrics',
                        'best_params', 'best_model', 'selected_features'
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    fold_metrics = []
    best_model_overall = None
    best_auc_overall = -1
    best_params_overall = {}
    selected_features_overall = []

    # Determine SMOTE application
    distress_ratio = y.mean()
    if force_smote is True:
        apply_smote = SMOTE_AVAILABLE
        if apply_smote:
            print(f"    SMOTE forced ON (distress ratio: {distress_ratio:.2%})")
        else:
            print(f"    Warning: SMOTE forced but imblearn not installed.")
    elif force_smote is False:
        apply_smote = False
        print(f"    SMOTE OFF (distress ratio: {distress_ratio:.2%})")
    else:
        apply_smote = distress_ratio > SMOTE_THRESHOLD and SMOTE_AVAILABLE
        if apply_smote:
            print(f"    SMOTE applied (distress ratio: {distress_ratio:.2%})")


    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # 1. Feature selection on training data
        sel_features, f_scores, _ = select_features_freg(
            X_train, y_train, k=FEATURE_SELECTION_K
        )
        X_train = X_train[sel_features]
        X_test = X_test[sel_features]

        # 2. MinMax scaling (fit on train, transform both)
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=sel_features, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=sel_features, index=X_test.index
        )

        # 3. SMOTE oversampling (training set only)
        if apply_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_final, y_train_final = smote.fit_resample(
                X_train_scaled, y_train
            )
        else:
            X_train_final, y_train_final = X_train_scaled, y_train

        # 4. Hyperparameter tuning with inner CV
        from sklearn.base import clone
        model_clone = clone(model)

        if param_grid:
            grid = GridSearchCV(
                model_clone, param_grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True,
                                   random_state=RANDOM_STATE),
                scoring="roc_auc",
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_train_final, y_train_final)
            fold_model = grid.best_estimator_
            fold_params = grid.best_params_
        else:
            model_clone.fit(X_train_final, y_train_final)
            fold_model = model_clone
            fold_params = {}

        # 5. Predict on test set
        y_pred = fold_model.predict(X_test_scaled)
        if hasattr(fold_model, "predict_proba"):
            y_prob = fold_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        # 6. Compute metrics
        fold_result = _compute_metrics(y_test, y_pred, y_prob)
        fold_result["fold"] = fold_idx
        fold_result["best_params"] = fold_params
        fold_metrics.append(fold_result)

        # Track best model across folds
        fold_auc = fold_result.get("auc", 0)
        if fold_auc > best_auc_overall:
            best_auc_overall = fold_auc
            best_model_overall = fold_model
            best_params_overall = fold_params
            selected_features_overall = sel_features

        print(f"    Fold {fold_idx:2d}: AUC={fold_result['auc']:.4f}  "
              f"F1={fold_result['f1']:.4f}  MCC={fold_result['mcc']:.4f}")

    # Aggregate metrics
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
    mean_metrics = {}
    std_metrics = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics if not np.isnan(m.get(key, np.nan))]
        mean_metrics[key] = np.mean(values) if values else np.nan
        std_metrics[key] = np.std(values) if values else np.nan

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "best_params": best_params_overall,
        "best_model": best_model_overall,
        "selected_features": selected_features_overall,
    }
