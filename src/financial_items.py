"""
Module to extract financial items from Turkish KAP/BIST formatted Excel files.

Each Excel file has a single 'Sheet1' with:
  - Column 0: Turkish financial statement item name
  - Column 1: Yıllık (Annual) value  ← we extract this

All Turkish names are mapped to standardised English keys.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

# ── Turkish → English mapping ─────────────────────────────────────────────
# First match wins when multiple Turkish labels map to the same English key.
TURKISH_TO_ENGLISH: Dict[str, str] = {
    # ── BALANCE SHEET ─────────────────────────────────────────────────────
    "dönen varlıklar":                                  "Current Assets",
    "nakit ve nakit benzerleri":                         "Cash",
    "ticari alacaklar":                                  "Accounts Receivable",  # dönen section (row 4)
    "stoklar":                                           "Inventory",            # dönen section (row 8)
    "duran varlıklar":                                   "Non-Current Assets",
    "toplam varlıklar":                                  "Total Assets",
    "toplam kaynaklar":                                  "Total Assets",

    # Current Liabilities
    "kısa vadeli yükümlülükler":                         "Current Liabilities",

    # Long-Term Liabilities
    "uzun vadeli yükümlülükler":                         "Long-Term Liabilities",

    # Equity
    "özkaynaklar":                                       "Equity",

    # Paid-in Capital (TTK 376 hesabı için)
    "ödenmiş sermaye":                                   "Paid Capital",
    "çıkarılmış sermaye":                                "Paid Capital",

    # Legal Reserves (TTK 376 hesabı için)
    "kardan ayrılan kısıtlanmış yedekler":               "Legal Reserves",
    "yasal yedekler":                                    "Legal Reserves",
    "kanuni yedek akçeler":                              "Legal Reserves",

    # Retained Earnings
    "geçmiş yıllar kar/zararları":                       "Retained Earnings",

    # ── INCOME STATEMENT ──────────────────────────────────────────────────
    "satış gelirleri":                                   "Revenue",
    "satışların maliyeti (-)":                           "COGS",
    "brüt kar (zarar)":                                  "Gross Profit",
    "ticari faaliyetlerden brüt kar (zarar)":            "Gross Profit",
    "faaliyet karı (zararı)":                            "Operating Income",
    "finansman gideri öncesi faaliyet karı/zararı":      "EBIT",
    "sürdürülen faaliyetler vergi öncesi karı (zararı)": "Pretax Income",
    "sürdürülen faaliyetler vergi geliri (gideri)":      "Tax Expense",          # negative = expense
    "dönem karı (zararı)":                               "Net Income",
    "sürdürülen faaliyetler dönem karı/zararı":          "Net Income",

    # Interest / Finance
    "finansman giderleri":                               "Interest Expense",
    "(esas faaliyet dışı) finansal giderler (-)":        "Interest Expense",

    # Operating expense details (for Total Expense calculation)
    "pazarlama, satış ve dağıtım giderleri (-)":        "SGA Expense",
    "genel yönetim giderleri (-)":                       "Admin Expense",
    "araştırma ve geliştirme giderleri (-)":             "RD Expense",

    # ── CASH FLOW ─────────────────────────────────────────────────────────
    "işletme faaliyetlerinden kaynaklanan net nakit":    "CFO",
}


def _normalise(text: str) -> str:
    """
    Normalise a Turkish financial-statement label for matching.

    Steps:
      1. lower-case
      2. strip U+0307 combining dot above (artifact of İ.lower())
      3. NFC normalise
      4. fold Turkish ı → i  (so that uppercase I → i and ı → i both
         produce the same output)
      5. collapse whitespace
    """
    import unicodedata
    s = str(text).lower().strip()
    s = s.replace("\u0307", "")           # remove combining dot above
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ı", "i")              # fold Turkish ı → i
    return " ".join(s.split())

# Pre-normalise the dictionary keys so that comparisons are consistent
_NORMALISED_MAP: Dict[str, str] = {_normalise(k): v for k, v in TURKISH_TO_ENGLISH.items()}


def extract_financial_items(file_path: str, required_items: list) -> Dict[str, Any]:
    """
    Extracts required financial items from a KAP-format Excel file.

    Args:
        file_path: Path to the Excel file.
        required_items: English keys to extract.

    Returns:
        Dict mapping each required English key → float value (np.nan if missing).
    """
    result: Dict[str, Any] = {item: np.nan for item in required_items}

    try:
        df = pd.read_excel(file_path, sheet_name=0, header=None)

        if df.shape[1] < 2:
            print(f"Warning: unexpected column count in {file_path}")
            return result

        # Build normalised look-up from Excel
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
            # Only keep the first occurrence of each normalised label
            # (handles duplicate "Ticari Alacaklar" rows – dönen vs duran)
            if norm not in lookup:
                lookup[norm] = val

        # Map Turkish → English using the pre-normalised dictionary
        found: Dict[str, float] = {}
        for turkish_norm, english_key in _NORMALISED_MAP.items():
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

