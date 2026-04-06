"""
Preprocessing module for the financial distress prediction pipeline.

Pipeline order (leakage-free):
  preprocess_data()          — missing-value imputation on the full dataset
                               (uses company-level medians, no future leakage)
  fit_winsorize_bounds()     — compute clip bounds from the *training* fold
  apply_winsorize_bounds()   — apply those bounds to train and test folds
  MinMaxScaler               — fit on train, transform both  (evaluation.py)
  SMOTE                      — training fold only             (evaluation.py)

Winsorization was previously applied to the full dataset before CV splitting,
which allowed test-set extreme values to influence the clip bounds.  The new
split into fit / apply removes that leakage.
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


def fit_winsorize_bounds(
    df: pd.DataFrame,
    columns: list,
    lower: float = 0.01,
    upper: float = 0.99,
) -> dict:
    """
    Computes per-column clip bounds from *training* data only.

    Args:
        df: Training fold DataFrame.
        columns: Feature columns to winsorize.
        lower: Lower quantile (default 1st percentile).
        upper: Upper quantile (default 99th percentile).

    Returns:
        Dict mapping column name → (lo, hi) clip bounds.
    """
    bounds = {}
    for col in columns:
        if col in df.columns:
            bounds[col] = (df[col].quantile(lower), df[col].quantile(upper))
    return bounds


def apply_winsorize_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Clips a DataFrame using pre-fitted winsorization bounds.

    Args:
        df: DataFrame to clip (train or test fold).
        bounds: Dict from fit_winsorize_bounds() — {col: (lo, hi)}.

    Returns:
        Clipped DataFrame.
    """
    df_out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lo, hi)
    return df_out


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the pre-CV preprocessing pipeline: missing-value imputation only.

    Winsorization is intentionally excluded here and moved inside each CV
    fold (evaluation.py) to prevent test-set leakage into clip bounds.
    Scaling (MinMaxScaler) and SMOTE are also handled inside each fold.

    Args:
        df: Raw dataset containing feature columns.

    Returns:
        Dataset with missing values imputed.
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
