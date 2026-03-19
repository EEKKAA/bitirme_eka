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
DATASET_FILENAME = OUTPUTS_DIR / "dataset.csv"
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
]

# ── Selected financial ratios (Büyükarıkan & Büyükarıkan 2025 + Literature Substs) ─
# 14 ratios in 4 categories
CANDIDATE_RATIOS = [
    # Capital structure (6)
    "short_term_liabilities_to_assets",# Current liabilities / Total assets (replaced debt_ratio)
    "equity_to_assets",                # Equity / Total assets
    "equity_to_short_term_liabilities",# Equity / Current liabilities
    "equity_to_long_term_liabilities", # Equity / Long-term liabilities
    "fixed_assets_to_total_liabilities",# Non-current assets / Total liabilities
    "gross_profit_to_long_term_liabilities", # Gross profit / Long-term liabilities

    # Liquidity (3)
    "current_assets_to_total_liabilities",  # Current assets / Total liabilities
    "quick_ratio",                          # (Current assets - Inventory) / Current liabilities
    "working_capital_to_total_assets",      # (CA - CL) / Total assets

    # Profitability (3)
    "operating_income_to_assets",      # Operating income / Total assets (replaced ROA)
    "ebit_to_current_liabilities",     # EBIT / Current liabilities
    "net_operating_profit_margin",     # Operating income / Revenue

    # Activity (2)
    "sales_to_current_assets",         # Revenue / Current assets
    "asset_turnover",                  # Revenue / Total assets
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

# ── Interaction features v4.1: 11 micro × macro terms ──────────────────────
# SHAP-validated + literature-supported + updated base ratios
INTERACTION_FEATURES = [
    "stl_ta_x_interest",         # short_term_liabilities_to_assets × int  — liquidity/leverage cost
    "oi_ta_x_unemployment",      # operating_income_to_assets × unemp      — demand collapse risk
    "gp_ltl_x_inflation",        # gross_profit_to_long_term_liab × inf    — real debt service capacity
    "margin_x_usdtry",           # net_op_profit_margin × usdtry_change    — FX margin impact
    "fixed_assets_x_interest",   # fixed_assets_to_total_liabilities × int — LT debt burden
    "quick_x_unemployment",      # quick_ratio × unemployment_rate         — liquidity-demand squeeze
    "turnover_x_gdp",            # asset_turnover × gdp_growth             — cyclical efficiency
    "equity_stl_x_interest",     # equity_to_short_term_liabilities × int  — equity buffer
    "wc_ta_x_credit",            # working_capital_to_total_assets × cred  — liquidity crunch
    "cfo_x_interest",            # (CFO/TA) × interest_rate                — cash flow vs. int burden
    "log_assets_x_gdp",          # log(Total Assets) × gdp_growth          — firm size cyclicality
]

# Combined feature set for v5.1: Strictly the Top 10 SHAP features to prevent overfitting
TOP_SHAP_FEATURES = [
    "cfo_x_interest",
    "equity_to_long_term_liabilities",
    "gp_ltl_x_inflation",
    "asset_turnover",
    "equity_to_short_term_liabilities",
    "short_term_liabilities_to_assets_trend_1yr",
    "fixed_assets_to_total_liabilities",
    "gross_profit_to_long_term_liabilities_trend_1yr",
    "quick_ratio",
    "quick_x_unemployment"
]

CANDIDATE_FEATURES = TOP_SHAP_FEATURES

TARGET = "bankruptcy_label"

# ── Feature selection ─────────────────────────────────────────────────────
FEATURE_SELECTION_METHOD = "f_regression"
FEATURE_SELECTION_K = len(TOP_SHAP_FEATURES)   # Do not drop any of these 10 curated features

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
        "n_estimators": [100, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
    },
    "CatBoost": {
        "iterations": [200, 500],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
    },
    "LightGBM": {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "min_child_samples": [10, 20],
        "subsample": [0.8, 1.0],
    },
}

# ── Evaluation metrics ────────────────────────────────────────────────────
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
PRIMARY_METRIC = "auc"
SECONDARY_METRIC = "f1"
