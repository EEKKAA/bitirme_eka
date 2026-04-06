"""
Feature selection module — combined F-classif + Mutual Information scoring.

v2 change: Replaced univariate F-classif with a 50/50 combination of
  F-classif (linear associations) and mutual_info_classif (non-linear
  associations). This better captures interaction features whose relationship
  with the target is non-linear, which pure ANOVA F-test misses.

Applied on training data only inside each CV fold to prevent leakage.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif


def select_features_freg(X_train: pd.DataFrame, y_train: pd.Series,
                         k: int = 13) -> tuple:
    """
    Selects the top-k features using a combined F-classif + Mutual Information
    score (equal 0.5 / 0.5 weighting). Both scores are min-max normalised
    before combining so their scales are comparable.

    F-classif captures linear relationships (fast, deterministic).
    mutual_info_classif captures non-linear relationships (slower, random seed
    fixed for reproducibility) — especially important for interaction features.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector (binary).
        k: Number of top features to select.

    Returns:
        Tuple of (selected_feature_names, combined_scores_dict, None)
        combined_scores_dict is sorted descending by combined score.
    """
    k = min(k, X_train.shape[1])

    # ── F-classif (ANOVA) scores ──────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_scores, _ = f_classif(X_train, y_train)

    f_scores = pd.Series(
        np.nan_to_num(f_scores, nan=0.0), index=X_train.columns
    )
    f_min, f_max = f_scores.min(), f_scores.max()
    f_norm = (f_scores - f_min) / (f_max - f_min + 1e-9)

    # ── Mutual Information scores ─────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    mi_scores = pd.Series(mi_scores, index=X_train.columns)
    mi_min, mi_max = mi_scores.min(), mi_scores.max()
    mi_norm = (mi_scores - mi_min) / (mi_max - mi_min + 1e-9)

    # ── Combined score (equal weight) ─────────────────────────────────────
    combined = 0.5 * f_norm + 0.5 * mi_norm

    # ── Select top-k ─────────────────────────────────────────────────────
    selected_features = combined.nlargest(k).index.tolist()

    combined_scores_dict = combined.sort_values(ascending=False).to_dict()

    return selected_features, combined_scores_dict, None
