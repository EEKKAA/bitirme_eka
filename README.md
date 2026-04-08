# BIST Finansal Distress Tahmini (TTK 376)

BIST'te işlem gören **331 Türk şirketi** için 2018-2024 yıllarını kapsayan, **TTK 376 v2** metodolojisine dayalı finansal distress (mali darboğaz) tahmini yapan uçtan uca bir makine öğrenmesi pipeline'ı. Bitirme tezi projesidir.

Metodoloji referansı: Büyükarıkan & Büyükarıkan (2025).

---

## Dataset Özeti

| Metrik | Değer |
|---|---|
| **Şirket sayısı** | **331** |
| Panel gözlemi | 2229 (firma-yıl) |
| Dönem | 2018 – 2024 |
| Distress (label=1) | 649 (≈ %29.1) |
| Sağlıklı (label=0) | 1580 (≈ %70.9) |
| Feature sayısı | 88 sütun |

**Etiketleme stratejisi (TTK 376 v2):**
- Distressed şirketler: 2018-2024 aralığında en az bir yıl `distress_376 = 1` olan tüm şirketler (tüm yılları dahil edilir).
- Sağlıklı şirketler: Aynı dönemde hiç distress tetiklememiş ve 7 tam yılı bulunan şirketler arasından sabit seed (42) ile rastgele 100 tane seçilir.
- Label smoothing: İzole distress yılları (örn. `0-1-0`) gürültü sayılıp 0'a çekilir.

**Feature kategorileri:**
- 23 ham bilanço/gelir tablosu kalemi
- 18 finansal oran (Büyükarıkan & Büyükarıkan 2025 + literatür)
- 4 trend (1 yıllık momentum)
- 11 mikro × makro etkileşim terimi
- 27 makroekonomik özellik (cari + lag-1 + lag-2 + YoY Δ)
- 5 meta alan (company, year, is_first_year, vb.)

---

## Proje Yapısı

```
bitirme_claude/
│
├── build_labeling_dataset.py    # [1] Ham xlsx → outputs/dataset_final.csv
├── train_models.py              # [2] 5-fold temporal CV + hiperparametre + SHAP
├── compare_kmv_ml.py            # [3] KMV (Merton) vs ML karşılaştırması
├── run_shap_only.py             # [4] Standalone SHAP analizi
├── generate_presentation_charts.py  # Sunum grafikleri üretir
│
├── config.py                    # Merkezi ayarlar (feature listesi, paths, vb.)
├── requirements.txt             # Python bağımlılıkları
│
├── data/
│   └── raw/
│       ├── companies/           # 331 şirketin xlsx'leri (TICKER_YEAR.xlsx)
│       ├── dataset_distress_v2.xlsx   # TTK 376 v2 etiket kaynağı
│       └── macro_data.csv       # Makroekonomik göstergeler (yıllık)
│
├── src/                         # Modüller
│   ├── financial_items.py       # Excel'den kalem çıkarımı
│   ├── ratio_calculator.py      # 18 finansal oran
│   ├── interaction_features.py  # Mikro × makro etkileşimleri
│   ├── preprocessing.py         # Imputation, winsorization, scaling
│   ├── feature_selection.py     # F-classif + mutual info
│   ├── ml_models.py             # Model tanımları
│   ├── evaluation.py            # Temporal CV + threshold optimization
│   ├── stacking.py              # Stacking ensemble
│   ├── kmv_model.py             # Merton distance-to-default (benchmark)
│   ├── shap_analysis.py         # SHAP explainability
│   └── visualization.py         # Tez grafikleri
│
├── outputs/                     # Dataset, metrikler, model, grafikler
│   ├── dataset_final.csv        # [1]'in çıktısı — 2229 × 88
│   ├── cv_results.csv           # Model × fold metrik tablosu
│   ├── selected_ratios.json     # SHAP feature ranking
│   ├── kmv_vs_ml_comparison.json    # KMV vs ML benchmark
│   ├── threshold_config.json    # Optimal F1 eşiği
│   ├── train_log.txt            # Eğitim çıktı log'u
│   ├── financial_distress_model.pkl   # En iyi model (gitignored)
│   └── plots/                   # ROC, CM, SHAP, feature importance (gitignored)
│
└── presentation.html            # Tez sunumu
```

---

## Kurulum

```bash
# Python 3.10+ önerilir
pip install -r requirements.txt
```

---

## Pipeline'ı Çalıştırma

Dört adımlı uçtan uca çalıştırma (her biri önceki çıktıyı kullanır):

```bash
# 1) Ham xlsx'lerden nihai dataset'i oluştur  →  outputs/dataset_final.csv
python build_labeling_dataset.py

# 2) Modelleri eğit (5-fold temporal CV, GridSearchCV, SMOTE, SHAP)
python train_models.py

# 3) KMV Merton benchmark'ı ile karşılaştır
python compare_kmv_ml.py

# 4) Standalone SHAP analizi (opsiyonel, train_models.py'den sonra)
python run_shap_only.py
```

`build_labeling_dataset.py` çıktısında şunu görmelisiniz:

```
Toplam gozlem    : 2,229
Distress (1)     : 649  (29.12%)
Saglikli (0)     : 1,580
Sirket sayisi    : 331
```

---

## Metodoloji

- **Cross-validation:** 5-fold temporal walk-forward (look-ahead bias yok).
- **Feature selection:** Her fold içinde F-classif ile top-20 seçim (nested).
- **Imbalance:** SMOTE (minority < %40 olduğunda uygulanır).
- **Modeller:** Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM + Stacking ensemble.
- **Hyperparameter tuning:** GridSearchCV (iç CV stratified).
- **Primary metric:** ROC-AUC
- **Secondary metric:** F1 (eşik optimizasyonu için)
- **Benchmark:** KMV Merton structural model (distance-to-default).

### Sonuçlar (5-fold temporal CV)

| Model | Mode | AUC | F1@opt | MCC |
|---|---|---|---|---|
| **Random Forest** | No SMOTE | **0.7403** | 0.5599 | 0.3545 |
| CatBoost | No SMOTE | 0.7389 | 0.5615 | 0.3473 |
| Stacking Ensemble | – | 0.7228 | 0.5555 | 0.3454 |
| XGBoost | No SMOTE | 0.7145 | 0.5601 | 0.2804 |
| LightGBM | No SMOTE | 0.6963 | 0.5138 | 0.2439 |
| Logistic Regression | No SMOTE | 0.6759 | 0.5027 | 0.1514 |
| KMV (Merton) — benchmark | – | 0.6137 | 0.3774 | 0.2730 |

En iyi ML model (Random Forest) KMV Merton benchmark'ına göre **%20.6 AUC iyileştirmesi** sağlıyor.

---

## Özelleştirme

Tüm ana ayarlar `config.py`'de:
- `REQUIRED_FINANCIAL_ITEMS`: Excel'den çıkarılacak kalemler
- `CANDIDATE_FEATURES`: Model girdi feature havuzu
- `TEST_SIZE`, `RANDOM_STATE`: Bölme ayarları
- `CV_FOLDS`: Temporal fold sayısı
- Hyperparameter grid'leri

---

## Atıf

Bu proje Büyükarıkan & Büyükarıkan (2025) metodolojisini Türk BIST verileri üzerinde TTK 376 v2 etiketleme stratejisi ile uygular. Akademik kullanım için bitirme tezi referansı belirtilmelidir.
