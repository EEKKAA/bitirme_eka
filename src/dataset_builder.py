"""
Module for building the final dataset from raw company data and distress labels.

Pipeline:
  1. Load all company financials from Excel files
  2. Calculate financial ratios
  3. Assign distress labels (bankruptcy year + all preceding years)
  4. Merge macroeconomic indicators (current year only — no lags)
  5. Create 11 interaction features (micro × macro)
  6. Remove excluded companies with no usable data
  7. Generate data profiling report
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_all_companies
from src.ratio_calculator import calculate_ratios
from src.interaction_features import create_interaction_features
from config import (
    LABELS_FILENAME, MACRO_DATA_FILENAME,
    MACRO_FEATURES, MACRO_LAG2_VARS, EXCLUDED_COMPANIES,
    CANDIDATE_RATIOS, CANDIDATE_FEATURES, TARGET,
    OUTPUTS_DIR,
)


def get_bankruptcy_labels() -> pd.DataFrame:
    """
    Loads distress labels from the external CSV file.

    Expected columns: company, bankruptcy_year
    - If bankruptcy_year is filled, the company experienced distress.
    - If bankruptcy_year is empty, the company is healthy.

    Returns:
        pd.DataFrame with columns ['company', 'bankruptcy_year']
    """
    if LABELS_FILENAME.exists():
        try:
            return pd.read_csv(LABELS_FILENAME)
        except Exception as e:
            print(f"Error reading labels file: {e}")

    return pd.DataFrame(columns=["company", "bankruptcy_year"])


def get_macro_data() -> pd.DataFrame:
    """
    Loads macroeconomic data and enriches it with year-over-year trend
    (delta) features for key macro indicators.

    Trend features (Δt = t − t-1) capture the *direction* of macro change,
    which Campbell, Hilscher & Szilagyi (2008) show is as informative as
    the level itself.  usdtry_change is already a raw change variable, so
    it is excluded from the diff computation.

    Returns:
        pd.DataFrame with original macro columns + trend columns.
    """
    if not MACRO_DATA_FILENAME.exists():
        print("No macro data found.")
        return pd.DataFrame(columns=["year"])

    try:
        macro_df = pd.read_csv(MACRO_DATA_FILENAME)
    except Exception as e:
        print(f"Error reading macro data file: {e}")
        return pd.DataFrame(columns=["year"])

    macro_df = macro_df.sort_values("year").reset_index(drop=True)

    # ── Year-over-year change (trend) features ────────────────────────────
    # usdtry_change already represents a change variable — skip it.
    trend_vars = [
        "gdp_growth",
        "inflation_rate",
        "interest_rate",
        "credit_growth",
        "unemployment_rate",
    ]
    for var in trend_vars:
        if var in macro_df.columns:
            macro_df[f"{var}_change"] = macro_df[var].diff()

    return macro_df


def generate_data_profile(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generates a data profiling report and saves it to a text file.

    Includes:
      - Overall dataset shape and class distribution
      - Missing value counts per feature
      - Company-level observation summary
      - Basic descriptive statistics for financial ratios
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  FINANCIAL DISTRESS PREDICTION — DATA PROFILING REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Dataset overview
    lines.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    n_companies = df["company"].nunique() if "company" in df.columns else 0
    lines.append(f"Number of companies: {n_companies}")
    years = sorted(df["year"].unique()) if "year" in df.columns else []
    lines.append(f"Years covered: {years}")
    lines.append("")

    # Class distribution
    if TARGET in df.columns:
        dist = df[TARGET].value_counts()
        total = len(df)
        lines.append("─── Class Distribution ───")
        for label, count in dist.items():
            label_name = "Distressed" if label == 1 else "Healthy"
            lines.append(f"  {label_name} (label={label}): {count} ({count/total:.1%})")
        lines.append("")

    # Missing values
    lines.append("─── Missing Values per Feature ───")
    features_to_check = [c for c in CANDIDATE_FEATURES if c in df.columns]
    for col in features_to_check:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        if n_missing > 0:
            lines.append(f"  {col:45s}: {n_missing:4d} missing ({pct:.1f}%)")
    n_complete = sum(1 for c in features_to_check if df[c].isna().sum() == 0)
    lines.append(f"  → {n_complete}/{len(features_to_check)} features have zero missing values")
    lines.append("")

    # Company-level summary
    if "company" in df.columns:
        lines.append("─── Company Observation Counts ───")
        company_counts = df.groupby("company").size().sort_values(ascending=False)
        for company, count in company_counts.items():
            label_info = ""
            if TARGET in df.columns:
                dist_count = df[df["company"] == company][TARGET].sum()
                if dist_count > 0:
                    label_info = f"  [DISTRESSED: {int(dist_count)} yrs]"
            lines.append(f"  {company:12s}: {count} observations{label_info}")
        lines.append("")

    # Ratio statistics
    lines.append("─── Financial Ratio Summary Statistics ───")
    ratio_cols = [c for c in CANDIDATE_RATIOS if c in df.columns]
    if ratio_cols:
        stats = df[ratio_cols].describe().round(4)
        lines.append(stats.to_string())
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    print(f"  Data profile saved to: {output_path}")


def build_dataset(root_folder: Path) -> pd.DataFrame:
    """
    Orchestrates the full data pipeline:

    1. Load raw company financials
    2. Calculate financial ratios
    3. Remove excluded companies (BIMEKS, EGELYH, MENSA)
    4. Assign distress labels
    5. Merge macroeconomic indicators (current year)
    6. Create 11 interaction features (micro × macro)
    7. Remove rows where Total Assets = 0 (ratios undefined)
    8. Generate data profiling report

    Args:
        root_folder: Directory containing raw company data folders.

    Returns:
        Panel dataset ready for preprocessing and modeling.
    """
    # ── Step 1: Load raw data ─────────────────────────────────────────────
    print(f"Loading companies from {root_folder}...")
    df = load_all_companies(root_folder)

    if df.empty:
        print("No data found!")
        return df

    # ── Step 2: Calculate financial ratios ────────────────────────────────
    print("Calculating financial ratios...")
    df = calculate_ratios(df)

    # ── Step 3: Remove excluded companies ─────────────────────────────────
    n_before = len(df)
    df = df[~df["company"].isin(EXCLUDED_COMPANIES)].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Removed {n_removed} rows from excluded companies: "
              f"{EXCLUDED_COMPANIES}")

    # ── Step 4: Assign distress labels ────────────────────────────────────
    labels_df = get_bankruptcy_labels()

    if not labels_df.empty and "company" in labels_df.columns:
        print("Merging distress labels...")
        df = df.merge(labels_df, on="company", how="left")

        if "bankruptcy_year" in df.columns:
            # Label all years up to and including bankruptcy_year as distressed (1).
            # No T-3 window — full distress history retained per company.
            is_distressed = df["bankruptcy_year"].notna()
            in_or_before = df["year"] <= df["bankruptcy_year"]
            df[TARGET] = (is_distressed & in_or_before).astype(int)

            n_labeled = int(df[TARGET].sum())
            print(f"  Distress labels assigned: {n_labeled} rows "
                  f"({n_labeled / len(df):.1%})")

            df = df.drop(columns=["bankruptcy_year"])
        else:
            df[TARGET] = df["company"].isin(
                labels_df["company"]
            ).astype(int)
    else:
        print("No external labels found. Assigning default 0 to bankruptcy_label.")
        df[TARGET] = 0

    # ── Step 5: Merge macro indicators (current, lag-1, lag-2) ──────────
    # Temporal enrichment per academic literature:
    #   Current (t)  : Shumway (2001), Campbell et al. (2008)
    #   Lag-1 (t-1)  : Shumway (2001), Duffie et al. (2007)
    #   Lag-2 (t-2)  : Duffie et al. (2007) — GDP/interest/inflation take
    #                  2 years to fully appear in company balance sheets
    #   Trend (Δt)   : Campbell et al. (2008) — level + direction together
    macro_df = get_macro_data()
    if not macro_df.empty and "year" in macro_df.columns:
        print("Merging macroeconomic indicators (current, lag-1, lag-2 + trends)...")

        # --- Lag-1: all macro columns (including trend features) ---
        macro_lag1 = macro_df.copy()
        macro_lag1["year"] = macro_lag1["year"] + 1
        lag1_rename = {col: f"{col}_lag1" for col in macro_lag1.columns if col != "year"}
        macro_lag1 = macro_lag1.rename(columns=lag1_rename)

        # --- Lag-2: key variables only (GDP, interest, inflation) ---
        lag2_cols = ["year"] + [v for v in MACRO_LAG2_VARS if v in macro_df.columns]
        macro_lag2 = macro_df[lag2_cols].copy()
        macro_lag2["year"] = macro_lag2["year"] + 2
        lag2_rename = {col: f"{col}_lag2" for col in macro_lag2.columns if col != "year"}
        macro_lag2 = macro_lag2.rename(columns=lag2_rename)

        df = df.merge(macro_df,   on="year", how="left")
        df = df.merge(macro_lag1, on="year", how="left")
        df = df.merge(macro_lag2, on="year", how="left")
    else:
        print("No macro data found. Proceeding without macroeconomic features.")

    # ── Step 6: Create 18 interaction features ─────────────────────────
    print("Creating 11 interaction features (micro x macro)...")
    df = create_interaction_features(df)

    # ── Step 7: Remove rows where Total Assets = 0 ───────────────────────
    if "Total Assets" in df.columns:
        bad_mask = (df["Total Assets"] == 0) | df["Total Assets"].isna()
        n_bad = bad_mask.sum()
        if n_bad > 0:
            print(f"  Removing {n_bad} rows with Total Assets = 0 or NaN")
            df = df[~bad_mask].reset_index(drop=True)

    # ── Step 8: Generate profiling report ─────────────────────────────────
    profile_path = OUTPUTS_DIR / "data_profile.txt"
    generate_data_profile(df, profile_path)

    return df
