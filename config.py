"""
Configuration settings for the Financial Distress Prediction System.
V3 – Büyükarıkan & Büyükarıkan (2025) methodology.
"""
import os
from pathlib import Path

# ── Base directories ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── File paths ────────────────────────────────────────────────────────────
DATASET_FILENAME = OUTPUTS_DIR / "dataset_final.csv"
CV_RESULTS_FILENAME = OUTPUTS_DIR / "cv_results.csv"
BEST_MODEL_FILENAME = OUTPUTS_DIR / "financial_distress_model.pkl"
SELECTED_RATIOS_FILENAME = OUTPUTS_DIR / "selected_ratios.json"
LABELS_FILENAME = RAW_DATA_DIR / "bankruptcy_labels.csv"
MACRO_DATA_FILENAME = RAW_DATA_DIR / "macro_data.csv"

# ── Financial items to extract from Excel ─────────────────────────────────
REQUIRED_FINANCIAL_ITEMS = [
    "Current Assets",
    "Cash",
    "Accounts Receivable",
    "Inventory",
    "Non-Current Assets",
    "Total Assets",
    "Current Liabilities",
    "Long-Term Liabilities",
    "Equity",
    "Retained Earnings",
    "Revenue",
    "COGS",
    "Gross Profit",
    "Operating Income",
    "EBIT",
    "Pretax Income",
    "Tax Expense",
    "Net Income",
    "Interest Expense",
    "SGA Expense",
    "Admin Expense",
    "RD Expense",
    "CFO",
    "Paid Capital",       # TTK 376 hesabı: Ödenmiş Sermaye
    "Legal Reserves",     # TTK 376 hesabı: Kanuni Yedek Akçe
]

# ── Selected financial ratios (Buyukarikan & Buyukarikan 2025 — 22 ratios) ──
# 17 from paper (redundant pairs removed) + 5 solvency/cash flow ratios
# Redundant pairs removed after correlation analysis:
#   debt_ratio (r=1.0 with equity_to_assets)
#   working_capital_to_net_sales (outlier-prone, WC/TA already present)
#   inventories_to_current_assets (r=0.816 with inventories_to_total_assets)
CANDIDATE_RATIOS = [
    # Capital structure (6)
    "short_term_liabilities_to_assets", # F7:  Current Liabilities / Total Assets
    "equity_to_assets",                 # F4:  Equity / Total Assets
    "equity_to_short_term_liabilities", # F18: Equity / Current Liabilities
    "equity_to_long_term_liabilities",  # F19: Equity / Long-Term Liabilities
    "fixed_assets_to_total_liabilities",# F16: Non-Current Assets / Total Liabilities
    "gross_profit_to_long_term_liabilities", # F15: Gross Profit / Long-Term Liabilities

    # Liquidity (3)
    "current_assets_to_total_liabilities",  # F17: Current Assets / Total Liabilities
    "quick_ratio",                          # F0:  (CA - Inventory) / CL
    "working_capital_to_total_assets",      # F2:  (CA - CL) / Total Assets

    # Profitability (5)
    "return_on_assets",                # F8:  Net Income / Total Assets
    "gross_profitability_ratio",       # F9:  Gross Profit / Revenue
    "operating_income_to_assets",      # F14: Operating Income / Total Assets
    "ebit_to_current_liabilities",     # F10: EBIT / Current Liabilities
    "net_operating_profit_margin",     # F13: Operating Income / Revenue

    # Activity (3)
    "sales_to_current_assets",         # F6:  Revenue / Current Assets
    "asset_turnover",                  # F5:  Revenue / Total Assets
    "inventories_to_total_assets",     # F12: Inventory / Total Assets

    # Solvency & Cash Flow (5) — Resilience + Altman Z-Score variables
    "cfo_to_assets",                   # CFO / Total Assets — operational cash generation
    "retained_earnings_to_assets",     # RE / Total Assets — Altman X2: cumulative profitability
    "interest_coverage",               # EBIT / Interest Expense — debt service capacity
    "ebit_to_total_assets",            # EBIT / Total Assets — Altman X3: earning power
    "equity_to_total_liabilities",     # Equity / Total Liabilities — Altman X4: solvency
]

# ── Companies to exclude (no usable financial data) ──────────────────────
EXCLUDED_COMPANIES = ["BIMEKS", "EGELYH", "MENSA"]

# ── Macroeconomic indicators ──────────────────────────────────────────────
MACRO_FEATURES = [
    "gdp_growth",                # Overall economic expansion/contraction
    "inflation_rate",            # Price instability and cost pressure
    "interest_rate",             # Cost of borrowing
    "usdtry_avg",                # Average exchange rate level
    "usdtry_change",             # Annual currency shock magnitude
    "unemployment_rate",         # Domestic demand weakness
    "industrial_prod_growth",    # Real sector production dynamics
    "credit_growth",             # Availability of bank financing
    "m2_growth",                 # Monetary expansion and liquidity
]

# ── Macroeconomic indicators (Lag-1) ──────────────────────────────────────
MACRO_LAG_FEATURES = [f"{col}_lag1" for col in MACRO_FEATURES]

# ── Trend (Momentum) Features ─────────────────────────────────────────────
TREND_FEATURES = [
    "operating_income_to_assets_trend_1yr",
    "short_term_liabilities_to_assets_trend_1yr",
    "asset_turnover_trend_1yr",
    "gross_profit_to_long_term_liabilities_trend_1yr"
]

# ── Interaction features v5: 3 micro × macro terms ─────────────────────────
# Reduced from 11 to 3 after correlation analysis:
#   8 interactions removed (r>0.85 with base ratio due to macro having only
#   7 unique values per year, making interaction ≈ scaled copy of base ratio)
# Kept only interactions with lower base-ratio correlation and higher SHAP:
INTERACTION_FEATURES = [
    "stl_ta_x_interest",         # short_term_liabilities_to_assets × interest_rate — liquidity/leverage cost
    "wc_ta_x_credit",            # working_capital_to_total_assets × credit_growth  — liquidity crunch
    "cfo_x_interest",            # (CFO/TA) × interest_rate                         — cash flow vs. int burden
]

# ── Direct macro main effects (literature-supported subset) ──────────
# Reduced from 6 to 4 after correlation analysis:
#   usdtry_change removed (r=0.959 with gdp_growth_lag1)
#   industrial_prod_growth_lag1 removed (r=0.954 with gdp_growth_lag1)
SELECTED_MACRO_DIRECT = [
    "inflation_rate",              # Price instability (Tinoco & Wilson 2013)
    "credit_growth",               # Bank financing availability
    "unemployment_rate",           # Demand-side weakness
    "gdp_growth_lag1",             # Lagged economic cycle
]

# ── Auxiliary features ───────────────────────────────────────────────
AUXILIARY_FEATURES = [
    "is_first_year",       # Binary: 1 if company's first year in panel (trend imputation flag)
    "log_total_assets",    # Firm size control (log scale)
]

# Combined feature set: ratios + trends + interactions + macro + auxiliary
CANDIDATE_FEATURES = (CANDIDATE_RATIOS + TREND_FEATURES + INTERACTION_FEATURES
                      + SELECTED_MACRO_DIRECT + AUXILIARY_FEATURES)

TARGET = "bankruptcy_label"

# ── Feature selection ─────────────────────────────────────────────────────
FEATURE_SELECTION_METHOD = "f_classif"
FEATURE_SELECTION_K = 18   # Select top 18 features from 35-feature pool via f_classif

# ── Class imbalance ───────────────────────────────────────────────────────
SMOTE_THRESHOLD = 0.60     # Apply SMOTE if distress ratio > 60%

# ── Cross-validation ──────────────────────────────────────────────────────
CV_FOLDS = 10
RANDOM_STATE = 42

# ── Hyperparameter grids ──────────────────────────────────────────────────
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
    },
    "Random Forest": {
        "n_estimators": [200, 500],
        "max_depth": [5, 10],
        "min_samples_leaf": [1, 3],
    },
    "XGBoost": {
        "n_estimators": [200, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "scale_pos_weight": [1, 1.7],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 3],
    },
    "CatBoost": {
        "iterations": [300, 500, 800],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.1],
        "l2_leaf_reg": [1, 3, 5],
    },
    "LightGBM": {
        "n_estimators": [200, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [31, 63],
        "min_child_samples": [10, 20],
        "subsample": [0.8],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 3],
    },
}

# ── RandomizedSearchCV settings ──────────────────────────────────────
RANDOMIZED_N_ITER = 20  # Number of random combinations to try per model

# ── Evaluation metrics ────────────────────────────────────────────────────
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
PRIMARY_METRIC = "auc"
SECONDARY_METRIC = "f1"
