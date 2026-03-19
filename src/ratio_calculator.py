"""
Ratio calculator for selected financial distress prediction variables.
Büyükarıkan & Büyükarıkan (2025) methodology — 13 ratios in 4 categories.

Each ratio is safe against division by zero (returns np.nan).
"""
import pandas as pd
import numpy as np


def _sd(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe divide – returns NaN where denominator is 0 or NaN."""
    return pd.Series(
        np.where((den == 0) | den.isna(), np.nan, num / den),
        index=num.index,
    )


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    """Get a column or a zero-filled series if it is missing."""
    if col in df.columns:
        return df[col].fillna(0)
    return pd.Series(0.0, index=df.index)


def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 13 selected financial ratios and appends them to *df*.

    Categories
    ──────────
    Capital Structure (5):  debt_ratio, equity_to_assets,
                            equity_to_short_term_liabilities,
                            equity_to_long_term_liabilities,
                            fixed_assets_to_total_liabilities
    Liquidity (3):          current_assets_to_total_liabilities,
                            quick_ratio, working_capital_to_total_assets
    Profitability (3):      return_on_assets, ebit_to_current_liabilities,
                            net_operating_profit_margin
    Activity (2):           sales_to_current_assets, asset_turnover
    """
    df = df.copy()

    # ── helper columns ────────────────────────────────────────────────────
    CA  = _get(df, "Current Assets")
    CL  = _get(df, "Current Liabilities")
    TA  = _get(df, "Total Assets")
    EQ  = _get(df, "Equity")
    INV = _get(df, "Inventory")
    NCA = _get(df, "Non-Current Assets")
    LTL = _get(df, "Long-Term Liabilities")

    REV  = _get(df, "Revenue")
    OI   = _get(df, "Operating Income")
    EBIT = _get(df, "EBIT")
    NI   = _get(df, "Net Income")
    GP   = _get(df, "Gross Profit")

    TL   = TA - EQ                          # Total Liabilities
    WC   = CA - CL                          # Working Capital
    QA   = CA - INV                         # Quick Assets

    # ── CAPITAL STRUCTURE (5) ─────────────────────────────────────────────
    df["short_term_liabilities_to_assets"] = _sd(CL, TA)
    df["equity_to_assets"]                 = _sd(EQ, TA)
    df["equity_to_short_term_liabilities"] = _sd(EQ, CL)
    df["equity_to_long_term_liabilities"]  = _sd(EQ, LTL)
    df["fixed_assets_to_total_liabilities"]= _sd(NCA, TL)
    df["gross_profit_to_long_term_liabilities"] = _sd(GP, LTL)

    # ── LIQUIDITY (3) ────────────────────────────────────────────────────
    df["current_assets_to_total_liabilities"] = _sd(CA, TL)
    df["quick_ratio"]                         = _sd(QA, CL)
    df["working_capital_to_total_assets"]     = _sd(WC, TA)

    # ── PROFITABILITY (3) ─────────────────────────────────────────────────
    df["operating_income_to_assets"]   = _sd(OI, TA)
    df["ebit_to_current_liabilities"]  = _sd(EBIT, CL)
    df["net_operating_profit_margin"]  = _sd(OI, REV)

    # ── ACTIVITY (2) ──────────────────────────────────────────────────────
    df["sales_to_current_assets"] = _sd(REV, CA)
    df["asset_turnover"]          = _sd(REV, TA)

    # ── TREND (MOMENTUM) FEATURES (4) ─────────────────────────────────────
    # Calculate 1-year change (Delta) for the most critical ratios
    trend_cols = [
        "operating_income_to_assets",             # OI/TA momentum
        "short_term_liabilities_to_assets",       # Debt buildup speed
        "asset_turnover",                         # Activity momentum
        "gross_profit_to_long_term_liabilities"   # Debt service momentum
    ]
    
    # Sort securely to ensure diff() works chronologically per company
    df = df.sort_values(by=["company", "year"])
    
    for col in trend_cols:
        if col in df.columns:
            df[f"{col}_trend_1yr"] = df.groupby("company")[col].diff(periods=1)

    return df
