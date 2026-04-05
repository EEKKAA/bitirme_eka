"""
Module for building the final dataset from raw company data and distress labels.

Pipeline:
  1. Load all company financials from Excel files
  2. Calculate financial ratios
  3. Assign distress labels (bankruptcy year + all preceding years)
  4. Merge macroeconomic indicators (current year only — no lags)
  5. Create 18 interaction features (micro × macro)
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
    MACRO_FEATURES, EXCLUDED_COMPANIES,
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
    Loads macroeconomic data (current year only — lag variables removed).

    Returns:
        pd.DataFrame with original macro columns.
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
    6. Create 18 interaction features (micro × macro)
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

            # ── (A) Olay penceresi: T, T-1, T-2 ─────────────────────────────
            is_distressed = df["bankruptcy_year"].notna()
            label_event = (
                is_distressed
                & (df["year"] >= df["bankruptcy_year"] - 2)
                & (df["year"] <= df["bankruptcy_year"])
            )

            # ── (B) TTK 376/2 ─────────────────────────────────────────────────
            # Şart: birikmiş zarar ≥ 2/3 × (ödenmiş sermaye + kanuni yedek)
            def _col(name):
                return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)

            paid_cap  = _col("Paid Capital")
            legal_res = _col("Legal Reserves")
            ret_earn  = _col("Retained Earnings")
            equity    = _col("Equity")

            capital_base     = paid_cap + legal_res          # ÖdenmişSermaye + KanuniYedek
            accumulated_loss = (-ret_earn).clip(lower=0)     # Birikmiş zarar (≥0)

            label_376_2 = (
                (ret_earn < 0)
                & (capital_base > 0)
                & (accumulated_loss >= (2 / 3) * capital_base)
            )

            # ── (C) TTK 376/3: borca batıklık (özkaynak < 0) ─────────────────
            label_376_3 = equity < 0

            # ── Birleşik etiket ───────────────────────────────────────────────
            # T-2 öncesi geçmiş yıllar silinmez, 0 olarak kalır.
            df[TARGET] = (label_event | label_376_2 | label_376_3).astype(int)

            df = df.drop(columns=["bankruptcy_year"])

            total_1 = int(df[TARGET].sum())
            print(f"  Distress labels assigned:")
            print(f"    (A) Event window (T to T-2) : {int(label_event.sum()):>5} obs")
            print(f"    (B) TTK 376/2               : {int(label_376_2.sum()):>5} obs")
            print(f"    (C) TTK 376/3               : {int(label_376_3.sum()):>5} obs")
            print(f"    Total label=1 (union)       : {total_1:>5} obs")
            print(f"    Total label=0               : {len(df) - total_1:>5} obs")
        else:
            df[TARGET] = df["company"].isin(
                labels_df["company"]
            ).astype(int)
    else:
        print("No external labels found. Assigning default 0 to bankruptcy_label.")
        df[TARGET] = 0

    # ── Step 5: Merge macro indicators (current year and lag-1) ─────────
    macro_df = get_macro_data()
    if not macro_df.empty and "year" in macro_df.columns:
        print("Merging macroeconomic indicators (current and lag-1)...")
        # Create lag-1 dataset by shifting year forward by 1
        macro_lag1 = macro_df.copy()
        macro_lag1["year"] = macro_lag1["year"] + 1
        
        # Rename columns to avoid collisions
        lag_rename = {col: f"{col}_lag1" for col in macro_lag1.columns if col != "year"}
        macro_lag1 = macro_lag1.rename(columns=lag_rename)
        
        df = df.merge(macro_df, on="year", how="left")
        df = df.merge(macro_lag1, on="year", how="left")
    else:
        print("No macro data found. Proceeding without macroeconomic features.")

    # ── Step 6: Create 18 interaction features ─────────────────────────
    print("Creating 18 interaction features (micro x macro)...")
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
