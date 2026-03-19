"""
Module to extract specific financial items from Turkish KAP/BIST formatted Excel files.

The Excel files contain a single 'Sheet1' with:
  - Column 0: Row label (Turkish financial statement item name)
  - Column 1: Yıllık (Annual) value  <-- we extract this column
  - Columns 2-4: Quarterly values    <-- ignored

Turkish item names are mapped to standard English keys for consistency.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

# ── Mapping: Turkish display name (lowercase, stripped) → English key ──────
TURKISH_TO_ENGLISH: Dict[str, str] = {
    # ---------- BALANCE SHEET ──────────────────────────────────────────────
    # Current Assets
    "dönen varlıklar":                             "Current Assets",

    # Current Liabilities
    "kısa vadeli yükümlülükler":                    "Current Liabilities",
    "kısa vadeli borçlar":                          "Current Liabilities",   # alt form

    # Total Assets
    "toplam varlıklar":                            "Total Assets",
    "toplam kaynaklar":                            "Total Assets",           # alt form

    # Total Debt (financial debt only – prefer long-term financial liabilities + short-term)
    "finansal borçlar":                            "Total Debt",             # both LT and ST

    # Equity
    "özkaynaklar":                                 "Equity",
    "ana ortaklığa ait özkaynaklar":               "Equity",                # alt form

    # Inventory
    "stoklar":                                     "Inventory",

    # ---------- INCOME STATEMENT ───────────────────────────────────────────
    # Revenue
    "satış gelirleri":                             "Revenue",
    "net satışlar":                                "Revenue",               # alt form
    "toplam gelirler":                             "Revenue",               # alt form

    # Net Income
    "dönem karı (zararı)":                         "Net Income",
    "dönem net kar/zararı":                        "Net Income",            # alt form
    "dönem net karı/zararı":                       "Net Income",            # alt form
    "sürdürülen faaliyetler dönem karı/zararı":    "Net Income",            # alt form
}

def _normalise(text: str) -> str:
    """Lower-case, strip and collapse white-space for matching."""
    return " ".join(str(text).lower().strip().split())


def extract_financial_items(file_path: str, required_items: list) -> Dict[str, Any]:
    """
    Extracts required financial items from a KAP-format Excel file.

    The file is expected to have:
      - A single sheet named 'Sheet1'
      - Column 0: item description (Turkish)
      - Column 1: Yıllık (annual) value

    Args:
        file_path: Absolute path to the Excel file.
        required_items: English keys to extract (e.g. ["Current Assets", ...]).

    Returns:
        Dict mapping each required English key → float value (np.nan if missing).
    """
    result: Dict[str, Any] = {item: np.nan for item in required_items}

    try:
        df = pd.read_excel(file_path, sheet_name=0, header=None)

        # Column 0 = labels, Column 1 = Yıllık (annual)
        if df.shape[1] < 2:
            print(f"Warning: unexpected column count in {file_path}")
            return result

        # Build a lookup: normalised_turkish_label → annual_value
        lookup: Dict[str, float] = {}

        for _, row in df.iterrows():
            raw_label = str(row.iloc[0])
            if raw_label in ("nan", ""):
                continue
            norm = _normalise(raw_label)
            try:
                val = float(str(row.iloc[1]).replace(",", "."))
            except (ValueError, TypeError):
                continue
            lookup[norm] = val

        # Map Turkish keys → English required keys
        for turkish_norm, english_key in TURKISH_TO_ENGLISH.items():
            if english_key in required_items and english_key not in result:
                # first match wins (so preferred forms listed first in the dict)
                pass  # already seeded with nan; we'll overwrite below

        found: Dict[str, float] = {}
        for turkish_norm, english_key in TURKISH_TO_ENGLISH.items():
            if english_key not in required_items:
                continue
            if english_key in found:
                continue  # first match wins
            if turkish_norm in lookup:
                found[english_key] = lookup[turkish_norm]

        result.update(found)

    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")

    return result
