"""
KMV (Merton) Distance-to-Default model for credit risk assessment.

Implements a simplified Merton structural model using balance-sheet proxies
when market equity data is unavailable. This is the standard approach in
academic literature for comparing structural models with ML approaches.

Methodology:
    DD = (ln(V_A / D) + (r - 0.5 * sigma_A^2) * T) / (sigma_A * sqrt(T))
    PD = Phi(-DD)  (Normal CDF)

Where:
    V_A     = Book value of Total Assets (proxy for asset value)
    D       = Total Liabilities (default point)
    r       = Risk-free rate (TCMB policy rate / 100)
    sigma_A = Asset volatility (estimated from cross-sectional variation)
    T       = Time horizon (1 year)
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve,
)


def compute_distance_to_default(
    total_assets: pd.Series,
    total_liabilities: pd.Series,
    risk_free_rate: pd.Series,
    asset_volatility: float,
    T: float = 1.0,
) -> pd.Series:
    """
    Computes Merton Distance-to-Default (DD).

    Args:
        total_assets: Book value of total assets (V_A proxy).
        total_liabilities: Book value of total liabilities (D).
        risk_free_rate: Annual risk-free rate as decimal (e.g. 0.175 for 17.5%).
        asset_volatility: Estimated annual asset volatility (sigma_A).
        T: Time horizon in years (default: 1 year).

    Returns:
        Distance-to-Default (DD) for each observation.
    """
    # Avoid division by zero
    V = total_assets.replace(0, np.nan)
    D = total_liabilities.replace(0, np.nan)

    # Ensure V > D for log computation; where V <= D, DD should be very low
    ratio = V / D
    ratio = ratio.clip(lower=1e-6)

    r = risk_free_rate

    dd = (np.log(ratio) + (r - 0.5 * asset_volatility**2) * T) / \
         (asset_volatility * np.sqrt(T))

    return dd


def compute_default_probability(dd: pd.Series) -> pd.Series:
    """
    Converts Distance-to-Default to Probability of Default using Normal CDF.

    PD = Phi(-DD)

    Higher DD → Lower PD (healthier firm)
    Lower DD  → Higher PD (distressed firm)
    """
    return pd.Series(norm.cdf(-dd), index=dd.index)


def estimate_asset_volatility(df: pd.DataFrame) -> float:
    """
    Estimates asset volatility from cross-sectional variation of
    equity-to-assets ratio, which is a standard proxy when market
    data is unavailable.

    Uses the within-company time-series standard deviation of equity/assets,
    then takes the median across companies.
    """
    if "equity_to_assets" in df.columns:
        col = "equity_to_assets"
    elif "Total Assets" in df.columns and "Total Liabilities" in df.columns:
        df = df.copy()
        df["_eq_ratio"] = 1 - (df["Total Liabilities"] / df["Total Assets"].replace(0, np.nan))
        col = "_eq_ratio"
    else:
        return 0.30  # Default fallback

    if "company" in df.columns:
        company_vol = df.groupby("company")[col].std()
        sigma = company_vol.median()
        if pd.isna(sigma) or sigma <= 0:
            sigma = df[col].std()
    else:
        sigma = df[col].std()

    # Ensure reasonable bounds (10% - 80%)
    sigma = np.clip(sigma, 0.10, 0.80)
    return float(sigma)


def run_kmv_model(df: pd.DataFrame, target_col: str = "bankruptcy_label") -> dict:
    """
    Runs the full KMV (Merton) model pipeline:
      1. Estimate asset volatility
      2. Compute Distance-to-Default for all observations
      3. Convert to Probability of Default
      4. Evaluate prediction performance against actual labels

    Args:
        df: Dataset with Total Assets, Total Liabilities, interest_rate columns.
        target_col: Target variable column name.

    Returns:
        Dictionary with DD, PD, metrics, and ROC curve data.
    """
    result = {
        "model_name": "KMV (Merton)",
        "dd": None,
        "pd": None,
        "metrics": {},
        "roc_curve": None,
        "confusion_matrix": None,
    }

    # Compute Total Liabilities if not present
    if "Total Liabilities" not in df.columns:
        if "Current Liabilities" in df.columns and "Long-Term Liabilities" in df.columns:
            df = df.copy()
            df["Total Liabilities"] = df["Current Liabilities"] + df["Long-Term Liabilities"]
            print("  Computed Total Liabilities = Current + Long-Term Liabilities")
        elif "Total Assets" in df.columns and "Equity" in df.columns:
            df = df.copy()
            df["Total Liabilities"] = df["Total Assets"] - df["Equity"]
            print("  Computed Total Liabilities = Total Assets - Equity")
        else:
            print("  KMV Error: Cannot compute Total Liabilities")
            return result

    # Check required columns
    if "Total Assets" not in df.columns:
        print("  KMV Error: Missing column 'Total Assets'")
        return result

    if target_col not in df.columns:
        print(f"  KMV Error: Missing target column '{target_col}'")
        return result

    # Step 1: Estimate asset volatility
    sigma_A = estimate_asset_volatility(df)
    print(f"  Estimated asset volatility (sigma_A): {sigma_A:.4f}")

    # Step 2: Get risk-free rate
    if "interest_rate" in df.columns:
        r = df["interest_rate"] / 100.0  # Convert percentage to decimal
    else:
        r = pd.Series(0.15, index=df.index)  # Default 15% for Turkey
        print("  Using default risk-free rate: 15%")

    # Step 3: Compute DD and PD
    dd = compute_distance_to_default(
        total_assets=df["Total Assets"],
        total_liabilities=df["Total Liabilities"],
        risk_free_rate=r,
        asset_volatility=sigma_A,
    )
    pd_score = compute_default_probability(dd)

    # Handle NaN
    valid_mask = dd.notna() & pd_score.notna() & df[target_col].notna()
    dd_valid = dd[valid_mask]
    pd_valid = pd_score[valid_mask]
    y_true = df.loc[valid_mask, target_col].astype(int)

    if len(y_true) == 0 or y_true.nunique() < 2:
        print("  KMV Error: Not enough valid data for evaluation")
        return result

    result["dd"] = dd
    result["pd"] = pd_score

    # Step 4: Evaluate
    # Optimal threshold from ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, pd_valid)
    auc = roc_auc_score(y_true, pd_valid)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    y_pred = (pd_valid >= optimal_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "optimal_threshold": float(optimal_threshold),
        "sigma_A": float(sigma_A),
    }

    result["metrics"] = metrics
    result["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    print(f"  KMV Results:")
    print(f"    AUC:       {metrics['auc']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    MCC:       {metrics['mcc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    Threshold: {metrics['optimal_threshold']:.4f}")

    return result
