"""
Stacking Ensemble for Financial Distress Prediction.

Architecture:
  Level-0 (base learners): CatBoost, LightGBM, XGBoost
  Level-1 (meta-learner):  Logistic Regression

Training uses out-of-fold (OOF) predictions to build the meta-feature
matrix, so the meta-learner never sees predictions made on the same data
the base models were trained on.  This is the standard stacking protocol
that prevents leakage between levels.

The full stacking pipeline mirrors the existing cross_validate_model():
  - Feature selection inside each fold (training data only)
  - Winsorization inside each fold  (fit on train, apply to both)
  - MinMax scaling inside each fold (fit on train, apply to both)
  - SMOTE on training fold only (optional)
  - OOF threshold optimisation (same logic as evaluation.py)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from config import CV_FOLDS, RANDOM_STATE, SMOTE_THRESHOLD, FEATURE_SELECTION_K, TEMPORAL_MIN_TRAIN_YEARS
from src.feature_selection import select_features_freg
from src.preprocessing import fit_winsorize_bounds, apply_winsorize_bounds
from src.evaluation import _compute_metrics, _find_optimal_threshold, _temporal_folds


def _build_base_models(scale_pos_weight: float = 4.0):
    """Returns the three base learner instances."""
    import xgboost as xgb

    base_models = {}

    try:
        from catboost import CatBoostClassifier
        base_models["CatBoost"] = CatBoostClassifier(
            random_seed=RANDOM_STATE,
            verbose=0,
            auto_class_weights="Balanced",
            iterations=500,
            depth=6,
            learning_rate=0.1,
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        base_models["LightGBM"] = lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
            n_jobs=1,
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=63,
            min_child_samples=10,
            subsample=0.8,
        )
    except ImportError:
        pass

    base_models["XGBoost"] = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    return base_models


def run_stacking_cv(
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float = 4.0,
    force_smote: bool = False,
    year_series: pd.Series = None,
) -> dict:
    """
    Runs stacking ensemble with 10-fold stratified CV.

    For each outer fold:
      1. An inner 10-fold CV produces OOF meta-features for the training fold.
      2. Base models are retrained on the full training fold to predict the
         test fold — these become the test meta-features.
      3. The meta-learner (Logistic Regression) is trained on the OOF
         meta-features and evaluated on the test meta-features.

    Returns:
        Dict with 'fold_metrics', 'fold_metrics_opt', 'mean_metrics',
        'std_metrics', 'mean_metrics_opt', 'std_metrics_opt', 'oof_threshold'.
    """
    if year_series is not None:
        outer_folds = list(_temporal_folds(year_series))
        cv_label = "Temporal walk-forward"
    else:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        outer_folds = list(skf.split(X, y))
        cv_label = f"StratifiedKFold (k={CV_FOLDS})"

    distress_ratio = y.mean()
    apply_smote = force_smote and SMOTE_AVAILABLE

    fold_metrics = []
    oof_probs    = np.zeros(len(y))
    oof_labels   = np.zeros(len(y))
    fold_data    = []
    best_auc_fold          = -1
    selected_features_best = []

    base_model_defs = _build_base_models(scale_pos_weight)
    n_base = len(base_model_defs)
    base_names = list(base_model_defs.keys())

    print(f"    Base learners: {base_names}")
    print(f"    Meta-learner:  Logistic Regression")
    print(f"    CV: {cv_label} ({len(outer_folds)} folds)")
    print(f"    SMOTE: {'ON' if apply_smote else 'OFF'} "
          f"(distress ratio: {distress_ratio:.2%})")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds, 1):
        X_train_raw = X.iloc[train_idx].copy()
        X_test_raw  = X.iloc[test_idx].copy()
        y_train     = y.iloc[train_idx].copy()
        y_test      = y.iloc[test_idx].copy()

        # --- Feature selection (train only) ---
        sel_features, _, _ = select_features_freg(
            X_train_raw, y_train, k=FEATURE_SELECTION_K
        )
        X_train_raw = X_train_raw[sel_features]
        X_test_raw  = X_test_raw[sel_features]

        # --- Winsorization (fit on train) ---
        w_bounds    = fit_winsorize_bounds(X_train_raw, sel_features)
        X_train_raw = apply_winsorize_bounds(X_train_raw, w_bounds)
        X_test_raw  = apply_winsorize_bounds(X_test_raw,  w_bounds)

        # --- MinMax scaling (fit on train) ---
        scaler      = MinMaxScaler()
        X_train_sc  = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=sel_features, index=X_train_raw.index,
        )
        X_test_sc   = pd.DataFrame(
            scaler.transform(X_test_raw),
            columns=sel_features, index=X_test_raw.index,
        )

        # --- SMOTE (train only) ---
        if apply_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_final, y_train_final = smote.fit_resample(X_train_sc, y_train)
        else:
            X_train_final, y_train_final = X_train_sc, y_train

        # ── Build OOF meta-features for the training fold ────────────────
        # Inner CV produces one OOF probability column per base model.
        inner_skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                    random_state=RANDOM_STATE + 1)

        # meta_train shape: (len(X_train_final), n_base)
        # Use original (pre-SMOTE) indices for proper OOF mapping
        meta_train = np.zeros((len(X_train_sc), n_base))

        for inner_train_idx, inner_val_idx in inner_skf.split(
            X_train_sc, y_train
        ):
            X_in_tr = X_train_sc.iloc[inner_train_idx]
            y_in_tr = y_train.iloc[inner_train_idx]
            X_in_val = X_train_sc.iloc[inner_val_idx]

            if apply_smote:
                sm = SMOTE(random_state=RANDOM_STATE)
                X_in_tr, y_in_tr = sm.fit_resample(X_in_tr, y_in_tr)

            for col_idx, (bname, bmodel) in enumerate(base_model_defs.items()):
                m = clone(bmodel)
                m.fit(X_in_tr, y_in_tr)
                meta_train[inner_val_idx, col_idx] = (
                    m.predict_proba(X_in_val)[:, 1]
                )

        # ── Build test meta-features: retrain on full training fold ──────
        meta_test = np.zeros((len(X_test_sc), n_base))
        for col_idx, (bname, bmodel) in enumerate(base_model_defs.items()):
            m = clone(bmodel)
            m.fit(X_train_final, y_train_final)
            meta_test[:, col_idx] = m.predict_proba(X_test_sc)[:, 1]

        # ── Train meta-learner ────────────────────────────────────────────
        meta_lr = LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
        )
        meta_lr.fit(meta_train, y_train)

        # ── Evaluate on test fold ─────────────────────────────────────────
        y_prob_test = meta_lr.predict_proba(meta_test)[:, 1]
        y_pred_default = (y_prob_test >= 0.5).astype(int)

        fold_result = _compute_metrics(y_test, y_pred_default, y_prob_test)
        fold_result["fold"] = fold_idx
        fold_metrics.append(fold_result)

        if fold_result.get("auc", 0) > best_auc_fold:
            best_auc_fold          = fold_result.get("auc", 0)
            selected_features_best = sel_features

        oof_probs[test_idx]  = y_prob_test
        oof_labels[test_idx] = y_test.values
        fold_data.append({
            "test_idx":    test_idx,
            "y_test":      y_test,
            "y_prob_test": y_prob_test,
            "fold_idx":    fold_idx,
        })

        print(f"    Fold {fold_idx:2d}: AUC={fold_result['auc']:.4f}  "
              f"F1@0.50={fold_result['f1']:.4f}  "
              f"MCC={fold_result['mcc']:.4f}")

    # ── OOF threshold optimisation ────────────────────────────────────────
    oof_threshold = _find_optimal_threshold(oof_labels, oof_probs)

    fold_metrics_opt = []
    for fd in fold_data:
        y_pred_opt = (fd["y_prob_test"] >= oof_threshold).astype(int)
        fold_result_opt = _compute_metrics(
            fd["y_test"], y_pred_opt, fd["y_prob_test"]
        )
        fold_result_opt["fold"]      = fd["fold_idx"]
        fold_result_opt["threshold"] = oof_threshold
        fold_metrics_opt.append(fold_result_opt)

    # ── Aggregate ─────────────────────────────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]

    def _agg(mlist):
        mean_m, std_m = {}, {}
        for k in metric_keys:
            vals = [m[k] for m in mlist if not np.isnan(m.get(k, np.nan))]
            mean_m[k] = np.mean(vals) if vals else np.nan
            std_m[k]  = np.std(vals)  if vals else np.nan
        return mean_m, std_m

    mean_metrics,     std_metrics     = _agg(fold_metrics)
    mean_metrics_opt, std_metrics_opt = _agg(fold_metrics_opt)

    print(
        f"\n    --- Stacking Ensemble Summary ---\n"
        f"    Default   (thr=0.50)    : AUC={mean_metrics['auc']:.4f}  "
        f"F1={mean_metrics['f1']:.4f}  Recall={mean_metrics['recall']:.4f}\n"
        f"    Optimised (thr={oof_threshold:.2f})    : AUC={mean_metrics_opt['auc']:.4f}  "
        f"F1={mean_metrics_opt['f1']:.4f}  Recall={mean_metrics_opt['recall']:.4f}"
    )

    return {
        "fold_metrics":      fold_metrics,
        "fold_metrics_opt":  fold_metrics_opt,
        "mean_metrics":      mean_metrics,
        "std_metrics":       std_metrics,
        "mean_metrics_opt":  mean_metrics_opt,
        "std_metrics_opt":   std_metrics_opt,
        "oof_threshold":     oof_threshold,
        "selected_features": selected_features_best,
        "best_params":       {},
    }


class StackingEnsemble:
    """
    Fitted stacking ensemble that accepts pre-scaled, feature-selected input.
    Compatible with joblib.dump / predict_proba interface.
    """
    def __init__(self, base_models: dict, meta_lr):
        self.base_models = base_models   # {name: fitted_estimator}
        self.meta_lr = meta_lr

    def _meta_features(self, X):
        return np.column_stack([
            m.predict_proba(X)[:, 1] for m in self.base_models.values()
        ])

    def predict_proba(self, X):
        return self.meta_lr.predict_proba(self._meta_features(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_stacking_final(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float = 4.0,
) -> "StackingEnsemble":
    """
    Trains the stacking ensemble on the full (already scaled) dataset.

    Steps:
      1. Inner CV produces OOF meta-features from base learners.
      2. Base learners are retrained on the full dataset.
      3. Meta-learner (Logistic Regression) is trained on OOF meta-features.

    Args:
        X_scaled: Pre-scaled feature matrix (shape: n_samples × n_features).
        y: Target vector.
        scale_pos_weight: Passed to XGBoost base learner.

    Returns:
        Fitted StackingEnsemble instance.
    """
    base_model_defs = _build_base_models(scale_pos_weight)
    n_base = len(base_model_defs)

    inner_skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # Build OOF meta-features
    meta_train = np.zeros((len(X_scaled), n_base))
    for inner_train_idx, inner_val_idx in inner_skf.split(X_scaled, y):
        X_in_tr  = X_scaled.iloc[inner_train_idx]
        y_in_tr  = y.iloc[inner_train_idx]
        X_in_val = X_scaled.iloc[inner_val_idx]
        for col_idx, (_, bmodel) in enumerate(base_model_defs.items()):
            m = clone(bmodel)
            m.fit(X_in_tr, y_in_tr)
            meta_train[inner_val_idx, col_idx] = m.predict_proba(X_in_val)[:, 1]

    # Retrain base learners on full data
    fitted_base = {}
    for bname, bmodel in base_model_defs.items():
        m = clone(bmodel)
        m.fit(X_scaled, y)
        fitted_base[bname] = m

    # Train meta-learner on OOF meta-features
    meta_lr = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
    )
    meta_lr.fit(meta_train, y)

    return StackingEnsemble(fitted_base, meta_lr)
