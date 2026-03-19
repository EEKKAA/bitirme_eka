"""
Data loader module for reading company financial data.

Supported filename formats inside a company folder:
  - COMPANY_YEAR.xlsx   (e.g. FROTO_2018.xlsx)   ← KAP / BIST format
  - YEAR.xlsx           (e.g. 2018.xlsx)           ← simple format

The year is always parsed from the numeric part of the stem.
"""
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from src.financial_items import extract_financial_items
from config import REQUIRED_FINANCIAL_ITEMS

# Regex that finds an isolated 4-digit year (2000-2099) anywhere in the filename stem
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _parse_year(stem: str) -> Optional[int]:
    """Return the year embedded in a filename stem, or None if not found."""
    m = _YEAR_RE.search(stem)
    return int(m.group(1)) if m else None


def load_company_year(file_path: Path) -> Dict[str, Any]:
    """
    Loads a single year's financial data for a company from an Excel file.

    Args:
        file_path: Path to the Excel file.

    Returns:
        Dict of extracted financial items.
    """
    return extract_financial_items(str(file_path), REQUIRED_FINANCIAL_ITEMS)


def load_company(folder: Path) -> pd.DataFrame:
    """
    Reads all years for a specific company folder.

    Accepts both  COMPANY_YEAR.xlsx  and  YEAR.xlsx  naming conventions.

    Args:
        folder: Directory containing the company's yearly Excel files.

    Returns:
        DataFrame with one row per year, or empty DataFrame if no files found.
    """
    data = []
    company_name = folder.name

    for file_path in sorted(folder.glob("*.xlsx")):
        year = _parse_year(file_path.stem)
        if year is None:
            print(f"  Skipping '{file_path.name}': cannot parse year from filename.")
            continue

        items = load_company_year(file_path)
        items["company"] = company_name
        items["year"] = year
        data.append(items)

    if data:
        return pd.DataFrame(data)
    return pd.DataFrame()


def load_all_companies(root_folder: Path) -> pd.DataFrame:
    """
    Reads all company sub-directories inside root_folder.

    Args:
        root_folder: Directory whose sub-directories each represent one company.

    Returns:
        Combined DataFrame for all companies and years.
    """
    all_dfs = []

    for company_dir in sorted(root_folder.iterdir()):
        if not company_dir.is_dir():
            continue  # skip files in root (labels.csv, macro.csv, …)

        print(f"  Loading {company_dir.name} …")
        company_df = load_company(company_dir)

        if not company_df.empty:
            all_dfs.append(company_df)
        else:
            print(f"  Warning: no data found for {company_dir.name}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()
