"""
Evaluation module with Stratified 10-fold cross-validation.
Handles SMOTE, MinMax scaling, and feature selection inside each fold
to prevent data leakage.

v2: Threshold optimization added.
  - _find_optimal_threshold() finds the F1-maximizing decision threshold
    using training-fold probabilities (no test leakage).
  - Each fold reports metrics at both threshold=0.5 and threshold=optimal.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

try:
    from optuna_integration import OptunaSearchCV
    import optuna
    import warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
    precision_recall_curve,
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from config import (
    CV_FOLDS, RANDOM_STATE, SMOTE_THRESHOLD, FEATURE_SELECTION_K,
    OPTUNA_PARAM_SPACES, N_OPTUNA_TRIALS, TEMPORAL_MIN_TRAIN_YEARS,
)
from src.feature_selection import select_features_freg
from src.preprocessing import fit_winsorize_bounds, apply_winsorize_bounds


def _temporal_folds(year_series: pd.Series, min_train_years: int = TEMPORAL_MIN_TRAIN_YEARS):
    """
    Expanding-window temporal fold generator for financial panel data.

    Each fold trains on all years < test_year and tests on test_year.
    This prevents future data from leaking into training — unlike StratifiedKFold
    which shuffles observations across years randomly.

    Example with years [2018..2024] and min_train_years=2:
      Fold 1: Train 2018-2019, Test 2020
      Fold 2: Train 2018-2020, Test 2021
      Fold 3: Train 2018-2021, Test 2022
      Fold 4: Train 2018-2022, Test 2023
      Fold 5: Train 2018-2023, Test 2024

    Args:
        year_series: pd.Series of year values aligned with X/y index.
        min_train_years: Minimum number of distinct training years before
                         the first test fold is created.

    Yields:
        (train_idx, test_idx) as numpy integer arrays.
    """
    years = sorted(year_series.unique())
    for test_year in years[min_train_years:]:
        train_idx = np.where(year_series.values < test_year)[0]
        test_idx  = np.where(year_series.values == test_year)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


def _find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds the decision threshold that maximises F1 score.

    Uses the precision-recall curve so every unique probability value is
    tested — no grid search needed and no test-set data is touched.

    Args:
        y_true: True binary labels (training fold only).
        y_prob: Predicted probabilities for the positive class (training fold).

    Returns:
        Optimal threshold in [0, 1].  Falls back to 0.5 if the curve is
        degenerate (e.g. all predictions identical).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds has one fewer element than precisions/recalls
    if len(thresholds) == 0:
        return 0.5

    denom = precisions[:-1] + recalls[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_scores = np.where(
            denom > 0,
            2 * precisions[:-1] * recalls[:-1] / denom,
            0.0,
        )

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


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
                         force_smote: bool = None,
                         year_series: pd.Series = None) -> dict:
    """
    Cross-validation with proper data handling inside each fold
    (scaling, SMOTE, feature selection, threshold tuning).

    CV strategy:
      - If year_series is provided: expanding-window temporal CV (walk-forward).
        Each fold trains on all years < test_year, tests on test_year only.
        This is the correct approach for financial panel data — prevents future
        information from leaking into training via random shuffling.
      - If year_series is None: falls back to StratifiedKFold (legacy).

    Args:
        model: Base estimator instance.
        param_grid: Hyperparameter grid for GridSearchCV.
        X: Feature matrix (all candidate ratios, preprocessed).
        y: Target vector (bankruptcy_label).
        model_name: Display name for logging.
        force_smote: If True, always apply SMOTE. If False, never apply.
                     If None, use threshold-based logic.
        year_series: pd.Series of year values (same index as X/y).
                     When provided, temporal walk-forward CV is used.

    Returns:
        Dict with keys: 'fold_metrics', 'fold_metrics_opt',
                        'mean_metrics', 'std_metrics',
                        'mean_metrics_opt', 'std_metrics_opt',
                        'best_params', 'best_model', 'selected_features',
                        'thresholds'
    """
    if year_series is not None:
        fold_iter = list(_temporal_folds(year_series))
        cv_label = "Temporal walk-forward"
    else:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        fold_iter = list(skf.split(X, y))
        cv_label = f"StratifiedKFold (k={CV_FOLDS})"

    fold_metrics = []
    fold_metrics_opt = []
    thresholds = []
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

    # ── Pass 1: run all folds, collect OOF probabilities ─────────────────
    oof_probs  = np.zeros(len(y))
    oof_labels = np.zeros(len(y))
    fold_data  = []   # store per-fold info for Pass 2

    print(f"    CV: {cv_label} ({len(fold_iter)} folds)")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter, 1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # 1. Feature selection on training data
        sel_features, f_scores, _ = select_features_freg(
            X_train, y_train, k=FEATURE_SELECTION_K
        )
        X_train = X_train[sel_features]
        X_test  = X_test[sel_features]

        # 2. Winsorization — fit on train, apply to both (no test leakage)
        w_bounds = fit_winsorize_bounds(X_train, sel_features)
        X_train  = apply_winsorize_bounds(X_train, w_bounds)
        X_test   = apply_winsorize_bounds(X_test,  w_bounds)

        # 4. MinMax scaling (fit on train, transform both)
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=sel_features, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=sel_features, index=X_test.index
        )

        # 5. SMOTE oversampling (training set only)
        if apply_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_final, y_train_final = smote.fit_resample(
                X_train_scaled, y_train
            )
        else:
            X_train_final, y_train_final = X_train_scaled, y_train

        # 6. Hyperparameter tuning with inner CV
        from sklearn.base import clone
        model_clone = clone(model)

        optuna_space = OPTUNA_PARAM_SPACES.get(model_name, {})
        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )

        if optuna_space and OPTUNA_AVAILABLE:
            # Optuna TPE search — wider continuous distributions
            # average_precision (AUC-PR) is used instead of roc_auc because
            # it is more informative under class imbalance: it focuses on the
            # minority (distress) class and penalises false negatives heavily.
            search = OptunaSearchCV(
                model_clone,
                optuna_space,
                cv=inner_cv,
                scoring="average_precision",
                n_trials=N_OPTUNA_TRIALS,
                random_state=RANDOM_STATE,
                refit=True,
                verbose=0,
            )
            search.fit(X_train_final, y_train_final)
            fold_model  = search.best_estimator_
            fold_params = search.best_params_
        elif param_grid:
            # Fallback: GridSearchCV
            grid = GridSearchCV(
                model_clone, param_grid,
                cv=inner_cv,
                scoring="average_precision",
                n_jobs=1,
                refit=True,
            )
            grid.fit(X_train_final, y_train_final)
            fold_model  = grid.best_estimator_
            fold_params = grid.best_params_
        else:
            model_clone.fit(X_train_final, y_train_final)
            fold_model  = model_clone
            fold_params = {}

        # 7. Predict probabilities on test set
        if hasattr(fold_model, "predict_proba"):
            y_prob_test = fold_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob_test = fold_model.predict(X_test_scaled).astype(float)

        # Accumulate OOF predictions
        oof_probs[test_idx]  = y_prob_test
        oof_labels[test_idx] = y_test.values

        # Default threshold=0.5 metrics (logged immediately)
        y_pred_default = (y_prob_test >= 0.5).astype(int)
        fold_result = _compute_metrics(y_test, y_pred_default, y_prob_test)
        fold_result["fold"]        = fold_idx
        fold_result["best_params"] = fold_params
        fold_result["threshold"]   = 0.5
        fold_metrics.append(fold_result)

        # Track best model across folds (AUC — threshold-independent)
        fold_auc = fold_result.get("auc", 0)
        if fold_auc > best_auc_overall:
            best_auc_overall       = fold_auc
            best_model_overall     = fold_model
            best_params_overall    = fold_params
            selected_features_overall = sel_features

        fold_data.append({
            "test_idx":    test_idx,
            "y_test":      y_test,
            "y_prob_test": y_prob_test,
            "fold_params": fold_params,
            "fold_idx":    fold_idx,
        })

        print(f"    Fold {fold_idx:2d}: AUC={fold_result['auc']:.4f}  "
              f"F1@0.50={fold_result['f1']:.4f}  "
              f"MCC={fold_result['mcc']:.4f}")

    # ── Pass 2: find OOF-optimal threshold, recompute metrics ────────────
    #
    # The threshold is found on the full set of OOF predictions.
    # Every single prediction was made on a held-out fold, so there is
    # zero leakage from the test set into the threshold decision.
    oof_threshold = _find_optimal_threshold(oof_labels, oof_probs)
    thresholds.append(oof_threshold)   # single global value

    for fd in fold_data:
        y_pred_opt = (fd["y_prob_test"] >= oof_threshold).astype(int)
        fold_result_opt = _compute_metrics(
            fd["y_test"], y_pred_opt, fd["y_prob_test"]
        )
        fold_result_opt["fold"]        = fd["fold_idx"]
        fold_result_opt["best_params"] = fd["fold_params"]
        fold_result_opt["threshold"]   = oof_threshold
        fold_metrics_opt.append(fold_result_opt)

    # Aggregate metrics — both default and optimised
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]

    def _aggregate(metrics_list):
        mean_m, std_m = {}, {}
        for key in metric_keys:
            vals = [m[key] for m in metrics_list
                    if not np.isnan(m.get(key, np.nan))]
            mean_m[key] = np.mean(vals) if vals else np.nan
            std_m[key]  = np.std(vals)  if vals else np.nan
        return mean_m, std_m

    mean_metrics,     std_metrics     = _aggregate(fold_metrics)
    mean_metrics_opt, std_metrics_opt = _aggregate(fold_metrics_opt)

    print(
        f"\n    --- {model_name} Summary ---\n"
        f"    Default   (thr=0.50)     : AUC={mean_metrics['auc']:.4f}  "
        f"F1={mean_metrics['f1']:.4f}  Recall={mean_metrics['recall']:.4f}\n"
        f"    Optimised (thr={oof_threshold:.2f})     : AUC={mean_metrics_opt['auc']:.4f}  "
        f"F1={mean_metrics_opt['f1']:.4f}  Recall={mean_metrics_opt['recall']:.4f}"
    )

    return {
        "fold_metrics":      fold_metrics,
        "fold_metrics_opt":  fold_metrics_opt,
        "mean_metrics":      mean_metrics,
        "std_metrics":       std_metrics,
        "mean_metrics_opt":  mean_metrics_opt,
        "std_metrics_opt":   std_metrics_opt,
        "best_params":       best_params_overall,
        "best_model":        best_model_overall,
        "selected_features": selected_features_overall,
        "thresholds":        thresholds,   # [oof_threshold]
        "oof_threshold":     oof_threshold,
    }
