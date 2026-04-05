"""
Feature selection module using F-classification scoring.
Applied on training data only inside each CV fold.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest


def select_features_freg(X_train: pd.DataFrame, y_train: pd.Series,
                         k: int = 13) -> tuple:
    """
    Selects the top-k features using ANOVA F-test for classification.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector (binary).
        k: Number of features to select.

    Returns:
        Tuple of (selected_feature_names, f_scores_dict, selector_object)
    """
    # Ensure k does not exceed available features
    k = min(k, X_train.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    # Get selected feature names
    mask = selector.get_support()
    selected_features = X_train.columns[mask].tolist()

    # Build F-score ranking
    f_scores = dict(zip(X_train.columns, selector.scores_))
    f_scores = {feat: score for feat, score in
                sorted(f_scores.items(), key=lambda x: x[1], reverse=True)
                if not np.isnan(score)}

    return selected_features, f_scores, selector
