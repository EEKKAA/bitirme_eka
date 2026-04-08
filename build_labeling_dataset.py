"""
build_labeling_dataset.py
=========================
data/raw/companies/ altındaki düz dosya yapısından (TICKER_YEAR.xlsx)
panel dataset oluşturur. Etiketler data/raw/dataset_distress_v2.xlsx'ten gelir.

Strateji:
  - Distressed şirketler : TTK 376 v2'ye göre distress_376=1 olan TÜM şirketler
                            (her şirketin 2018-2024 arası tüm yılları dahil edilir)
  - Sağlıklı şirketler   : 2018-2024 arası HİÇ TTK 376 tetiklememiş şirketler
                            içinden 7 tam yılı olan 100 şirket rastgele seçilir
  - Toplam                : 331 şirket × ~7 yıl = 2229 firma-yıl gözlemi

Çıktı: outputs/dataset_final.csv
"""

import sys
import random
import re
from pathlib import Path

import pandas as pd
import numpy as np

# ── Proje kökünü Python path'ine ekle ─────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
COMPANIES_DIR = RAW_DIR / "companies"
LABELS_V2     = RAW_DIR / "dataset_distress_v2.xlsx"
OUTPUT_PATH   = PROJECT_ROOT / "outputs" / "dataset_final.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from src.financial_items import extract_financial_items
from src.ratio_calculator import calculate_ratios
from src.interaction_features import create_interaction_features
from src.preprocessing import preprocess_data
from config import (
    REQUIRED_FINANCIAL_ITEMS,
    MACRO_DATA_FILENAME,
    MACRO_FEATURES,
    CANDIDATE_FEATURES,
    TARGET,
    OUTPUTS_DIR,
)

YEAR_MIN   = 2018
YEAR_MAX   = 2024
N_HEALTHY  = 100      # Rastgele seçilecek sağlıklı şirket sayısı
RANDOM_SEED = 42


# ── Şirket seçimi ─────────────────────────────────────────────────────────────

def select_companies(labels_path: Path):
    """
    TTK 376 v2 sonuçlarından şirket listelerini çıkar.

    Returns:
        distressed_companies : list[str]  — en az 1 yıl distress=1 olan firmalar
        healthy_companies    : list[str]  — hiç distress tetiklemeyen firmalar
                                           (7 tam yılı olanlar arasından rastgele N_HEALTHY)
    """
    df = pd.read_excel(labels_path)
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()

    dist_set   = set(df[df["distress_376"] == 1]["firm_id"].unique())
    all_set    = set(df["firm_id"].unique())
    healthy_set = all_set - dist_set

    # 7 tam yılı olan sağlıklı şirketler
    hdf        = df[df["firm_id"].isin(healthy_set)]
    full_7yr   = hdf.groupby("firm_id")["year"].count()
    full_7yr   = full_7yr[full_7yr == 7].index.tolist()

    random.seed(RANDOM_SEED)
    selected_healthy = random.sample(full_7yr, min(N_HEALTHY, len(full_7yr)))

    print(f"  Distressed sirket           : {len(dist_set)}")
    print(f"  Saglikli aday (7 yil)       : {len(full_7yr)}")
    print(f"  Secilen saglikli            : {len(selected_healthy)}")

    return sorted(dist_set), sorted(selected_healthy)


# ── TTK 376 label lookup ──────────────────────────────────────────────────────

def build_label_lookup(labels_path: Path) -> dict:
    """
    (firm_id, year) -> distress_376 eşleme sözlüğü döndürür.
    """
    df = pd.read_excel(labels_path)
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row["firm_id"]).upper(), int(row["year"]))] = int(row["distress_376"])
    return lookup


# ── Düz klasörden veri yükleme ────────────────────────────────────────────────

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def load_flat_companies(folder: Path, company_list: list) -> pd.DataFrame:
    """
    TICKER_YEAR.xlsx formatındaki düz dosya yapısından veri yükler.

    Args:
        folder       : data/raw/companies klasörü
        company_list : yüklenecek ticker listesi

    Returns:
        Birleşik DataFrame — her satır bir firma-yıl
    """
    ticker_set = {t.upper() for t in company_list}
    records    = []

    # Dosyaları bir kez tarayıp ticker'a göre grupla
    for fpath in sorted(folder.glob("*.xlsx")):
        fname = fpath.stem  # TICKER_YEAR
        m = re.match(r"^([A-Z0-9]+)_(\d{4})$", fname, re.IGNORECASE)
        if not m:
            continue

        ticker = m.group(1).upper()
        year   = int(m.group(2))

        if ticker not in ticker_set:
            continue
        if not (YEAR_MIN <= year <= YEAR_MAX):
            continue

        items = extract_financial_items(str(fpath), REQUIRED_FINANCIAL_ITEMS)
        items["company"] = ticker
        items["year"]    = year
        records.append(items)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["company", "year"]).reset_index(drop=True)
    print(f"  Yuklendi: {len(df):,} firma-yil ({df['company'].nunique()} sirket)")
    return df


# ── Makro veri ────────────────────────────────────────────────────────────────

def get_macro_data() -> pd.DataFrame:
    if not MACRO_DATA_FILENAME.exists():
        print("  Uyari: makro veri bulunamadi.")
        return pd.DataFrame(columns=["year"])
    macro_df = pd.read_csv(MACRO_DATA_FILENAME).sort_values("year").reset_index(drop=True)
    return macro_df


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 65

    print(SEP)
    print("  LABELING DATASET BUILDER")
    print(f"  Kaynak: {COMPANIES_DIR}")
    print(SEP)

    # ── 1) Şirket seçimi ─────────────────────────────────────────────────────
    print("\n[1] Sirket secimi...")
    dist_companies, healthy_companies = select_companies(LABELS_V2)
    all_companies = dist_companies + healthy_companies
    print(f"  Toplam sirket             : {len(all_companies)}")

    # ── 2) Finansal veriyi yükle ──────────────────────────────────────────────
    print("\n[2] Finansal veri yukleniyor...")
    df = load_flat_companies(COMPANIES_DIR, all_companies)

    if df.empty:
        print("Hata: Veri yüklenemedi.")
        return

    # ── 3) Finansal oran hesapla ─────────────────────────────────────────────
    print("\n[3] Finansal oranlar hesaplaniyor...")
    df = calculate_ratios(df)

    # ── 4) Total Assets = 0 olan satırları çıkar ─────────────────────────────
    if "Total Assets" in df.columns:
        bad = (df["Total Assets"] == 0) | df["Total Assets"].isna()
        n_bad = bad.sum()
        if n_bad > 0:
            print(f"  Kaldirilan: {n_bad} satir (Total Assets=0 veya NaN)")
            df = df[~bad].reset_index(drop=True)

    # ── 5) TTK 376 etiketlerini ata ──────────────────────────────────────────
    print("\n[4] TTK 376 etiketleri ataniyor...")
    label_lookup = build_label_lookup(LABELS_V2)

    df[TARGET] = df.apply(
        lambda r: label_lookup.get((str(r["company"]).upper(), int(r["year"])), 0),
        axis=1,
    )

    n1 = int(df[TARGET].sum())
    n0 = len(df) - n1
    print(f"  Label=1 (distress) : {n1}")
    print(f"  Label=0 (saglikli) : {n0}")

    # ── 5b) Label smoothing — izole distress yıllarını kaldır ────────────
    print("\n[4b] Label smoothing (izole yil filtresi)...")
    before = int(df[TARGET].sum())
    for company, grp in df.groupby("company"):
        idx = grp.sort_values("year").index
        labels = df.loc[idx, TARGET].values
        for i in range(1, len(labels) - 1):
            if labels[i] == 1 and labels[i - 1] == 0 and labels[i + 1] == 0:
                df.loc[idx[i], TARGET] = 0
    after = int(df[TARGET].sum())
    print(f"  Smoothing: {before} -> {after} distress ({before - after} izole yil kaldirildi)")

    # ── 6) Makro veri birleştir ───────────────────────────────────────────────
    print("\n[5] Makro veriler birlestiriliyor...")
    macro_df = get_macro_data()
    if not macro_df.empty and "year" in macro_df.columns:
        macro_lag1 = macro_df.copy()
        macro_lag1["year"] = macro_lag1["year"] + 1
        lag_rename = {c: f"{c}_lag1" for c in macro_lag1.columns if c != "year"}
        macro_lag1 = macro_lag1.rename(columns=lag_rename)
        df = df.merge(macro_df,   on="year", how="left")
        df = df.merge(macro_lag1, on="year", how="left")
        print(f"  Makro ozellikler eklendi.")
    else:
        print("  Makro veri yok, atlaniyor.")

    # ── 7) Interaction features ──────────────────────────────────────────────
    print("\n[6] Interaction features olusturuluyor...")
    df = create_interaction_features(df)

    # ── 8) Önişleme ──────────────────────────────────────────────────────────
    print("\n[7] Onisleme (imputation + winsor + scale)...")
    df = preprocess_data(df)

    # ── 9) Özet ve kaydet ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SONUC")
    print(SEP)
    print(f"  Toplam gozlem    : {len(df):,}")
    print(f"  Distress (1)     : {int(df[TARGET].sum()):,}  "
          f"({df[TARGET].mean()*100:.2f}%)")
    print(f"  Saglikli (0)     : {len(df) - int(df[TARGET].sum()):,}")
    print(f"  Sirket sayisi    : {df['company'].nunique()}")

    feats_present = [f for f in CANDIDATE_FEATURES if f in df.columns]
    print(f"  Ozellik sayisi   : {len(feats_present)} / {len(CANDIDATE_FEATURES)}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Dataset kaydedildi: {OUTPUT_PATH}")
    print(SEP)


if __name__ == "__main__":
    main()
