"""
Preprocessing module for the financial distress prediction pipeline.

Handles missing values and winsorization of extreme ratio values.
MinMax scaling is applied inside each CV fold (see evaluation.py).
"""
import pandas as pd
import numpy as np
from config import CANDIDATE_FEATURES, CANDIDATE_RATIOS


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Imputes missing values using a two-stage approach:

    Stage 1 — Company-level median: for each company, fill NaNs with the
              median of that company's other years (financial logic: same
              firm's historical behavior is the best proxy).

    Stage 2 — Global median fallback: fill remaining NaNs (e.g. company
              has only 1 year of data) with the global median across all
              companies.

    Args:
        df: Dataset with ratio and feature columns.
        strategy: Fallback strategy ('median', 'mean').

    Returns:
        Dataset with missing values imputed.
    """
    df_out = df.copy()
    features_present = [f for f in CANDIDATE_FEATURES if f in df.columns]

    if not features_present:
        return df_out

    # Stage 1: Company-level median imputation
    if "company" in df_out.columns:
        for col in features_present:
            company_medians = df_out.groupby("company")[col].transform("median")
            df_out[col] = df_out[col].fillna(company_medians)

    # Stage 2: Global median/mean fallback
    for col in features_present:
        if df_out[col].isna().any():
            if strategy == "median":
                fill_val = df_out[col].median()
            else:
                fill_val = df_out[col].mean()
            df_out[col] = df_out[col].fillna(fill_val)

    return df_out


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Winsorizes (clips) outliers in the feature columns.

    Args:
        df: Dataset with feature columns.
        lower: Lower percentile for clipping.
        upper: Upper percentile for clipping.

    Returns:
        Dataset with outliers clipped.
    """
    df_out = df.copy()
    features_present = [f for f in CANDIDATE_FEATURES if f in df.columns]

    for col in features_present:
        lo = df_out[col].quantile(lower)
        hi = df_out[col].quantile(upper)
        df_out[col] = df_out[col].clip(lo, hi)

    return df_out


def winsorize_from_train(df_train: pd.DataFrame, df_test: pd.DataFrame,
                         lower: float = 0.01, upper: float = 0.99):
    """
    Winsorizes both train and test using bounds computed from train only.
    Prevents data leakage by never peeking at test quantiles.

    Returns:
        (df_train_clipped, df_test_clipped)
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    features_present = [f for f in CANDIDATE_FEATURES if f in df_train.columns]

    for col in features_present:
        lo = df_train[col].quantile(lower)
        hi = df_train[col].quantile(upper)
        df_train[col] = df_train[col].clip(lo, hi)
        df_test[col] = df_test[col].clip(lo, hi)

    return df_train, df_test


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the global preprocessing pipeline: imputation only.
    Winsorization is handled inside CV folds to prevent data leakage.

    Args:
        df: Raw dataset containing feature columns.

    Returns:
        Preprocessed dataset (imputed, NOT winsorized).
    """
    print("  Handling missing values (company-median -> global-median)...")
    df = handle_missing_values(df)

    # Report remaining missing
    features_present = [f for f in CANDIDATE_FEATURES if f in df.columns]
    remaining = sum(df[col].isna().sum() for col in features_present)
    if remaining > 0:
        print(f"  Warning: {remaining} missing values remain after imputation")
    else:
        print(f"  OK: All {len(features_present)} features have zero missing values")

    return df
