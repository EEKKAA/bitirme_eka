"""
Machine learning models for financial distress prediction.
Logistic Regression, XGBoost, CatBoost, LightGBM -- with hyperparameter grid configs.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from config import RANDOM_STATE, PARAM_GRIDS


def build_model_configs(scale_pos_weight: float = 4.0) -> dict:
    """
    Returns a dictionary of {model_name: (model_instance, param_grid)}.

    Each model is configured with class_weight / scale_pos_weight
    to handle imbalanced classes.

    Args:
        scale_pos_weight: Ratio of negative to positive samples in the
            training set (n_neg / n_pos).  Used by XGBoost, which does not
            support sklearn's class_weight='balanced'.  Defaults to 4.0
            (≈ 417/104 for the BIST dataset); pass the value computed from
            the actual training labels for correctness.
    """
    configs = {}

    # Logistic Regression
    configs["Logistic Regression"] = (
        LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
        PARAM_GRIDS["Logistic Regression"],
    )

    # Random Forest
    configs["Random Forest"] = (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=1,
        ),
        PARAM_GRIDS["Random Forest"],
    )

    # XGBoost — scale_pos_weight replaces class_weight for gradient boosting
    configs["XGBoost"] = (
        xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
            scale_pos_weight=scale_pos_weight,
        ),
        PARAM_GRIDS["XGBoost"],
    )

    # CatBoost
    if CATBOOST_AVAILABLE:
        configs["CatBoost"] = (
            CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=0,
                auto_class_weights="Balanced",
            ),
            PARAM_GRIDS["CatBoost"],
        )
    else:
        print("Warning: CatBoost not installed. Skipping CatBoost model.")

    # LightGBM
    if LIGHTGBM_AVAILABLE:
        configs["LightGBM"] = (
            lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
                n_jobs=1,
            ),
            PARAM_GRIDS["LightGBM"],
        )
    else:
        print("Warning: LightGBM not installed. Skipping LightGBM model.")

    return configs
