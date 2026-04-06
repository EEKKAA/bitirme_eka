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
THRESHOLD_FILENAME = OUTPUTS_DIR / "threshold_config.json"
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
    # Capital structure (7)
    "short_term_liabilities_to_assets",      # Current liabilities / Total assets
    "debt_ratio",                            # Total liabilities / Total assets  [NEW]
    "equity_to_assets",                      # Equity / Total assets
    "equity_to_short_term_liabilities",      # Equity / Current liabilities
    "equity_to_long_term_liabilities",       # Equity / Long-term liabilities
    "fixed_assets_to_total_liabilities",     # Non-current assets / Total liabilities
    "gross_profit_to_long_term_liabilities", # Gross profit / Long-term liabilities

    # Liquidity (5)
    "current_assets_to_total_liabilities",   # Current assets / Total liabilities
    "quick_ratio",                           # (CA - Inventory) / Current liabilities
    "working_capital_to_total_assets",       # (CA - CL) / Total assets
    "working_capital_to_net_sales",          # (CA - CL) / Revenue  [NEW]
    "inventories_to_current_assets",         # Inventory / Current assets  [NEW]
    "inventories_to_total_assets",           # Inventory / Total assets  [NEW]

    # Profitability (4)
    "operating_income_to_assets",            # Operating income / Total assets
    "return_on_assets",                      # Net income / Total assets  [NEW]
    "ebit_to_current_liabilities",           # EBIT / Current liabilities
    "net_operating_profit_margin",           # Operating income / Revenue

    # Activity (2)
    "sales_to_current_assets",               # Revenue / Current assets
    "asset_turnover",                        # Revenue / Total assets
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

# ── Macroeconomic trend features (year-over-year Δ) ───────────────────────
# Campbell, Hilscher & Szilagyi (2008): level + direction of change together
# provide stronger signal than level alone. usdtry_change already exists.
MACRO_TREND_FEATURES = [
    "gdp_growth_change",       # GDP momentum (acceleration/deceleration)
    "inflation_rate_change",   # Inflation shock direction
    "interest_rate_change",    # Rate hike/cut signal
    "credit_growth_change",    # Credit supply shift
    "unemployment_rate_change", # Labour market direction
]

# ── Macroeconomic lag-2 features (key variables only) ─────────────────────
# Duffie, Saita & Wang (2007): GDP/interest/inflation effects on firm
# balance sheets can take up to 2 years to fully materialise.
MACRO_LAG2_VARS = ["gdp_growth", "interest_rate", "inflation_rate"]
MACRO_LAG2_FEATURES = [f"{col}_lag2" for col in MACRO_LAG2_VARS]

# ── Trend (Momentum) Features ─────────────────────────────────────────────
TREND_FEATURES = [
    "operating_income_to_assets_trend_1yr",
    "short_term_liabilities_to_assets_trend_1yr",
    "asset_turnover_trend_1yr",
    "gross_profit_to_long_term_liabilities_trend_1yr"
]

# ── Interaction features v5: 11 micro × macro terms ────────────────────────
# SHAP-validated + literature-supported + academically strengthened (v5 revision)
# REMOVED (v5): fixed_assets_x_interest (corr +0.044), quick_x_unemployment
#               (corr +0.006), equity_stl_x_interest (corr -0.030)
# ADDED   (v5): roa_x_gdp (Beaver 1966, Altman 1968), debt_ratio_x_interest
#               (Shumway 2001, Campbell 2008), roa_x_inflation (TR macro context)
INTERACTION_FEATURES = [
    "stl_ta_x_interest",         # short_term_liabilities_to_assets × int  — liquidity/leverage cost
    "oi_ta_x_unemployment",      # operating_income_to_assets × unemp      — demand collapse risk
    "gp_ltl_x_inflation",        # gross_profit_to_long_term_liab × inf    — real debt service capacity
    "margin_x_usdtry",           # net_op_profit_margin × usdtry_change    — FX margin impact (TR)
    "turnover_x_gdp",            # asset_turnover × gdp_growth             — cyclical efficiency
    "roa_x_gdp",                 # return_on_assets × gdp_growth           — profitability cycle risk
    "debt_ratio_x_interest",     # debt_ratio × interest_rate              — total leverage cost
    "roa_x_inflation",           # return_on_assets × inflation_rate       — real profitability erosion
    "wc_ta_x_credit",            # working_capital_to_total_assets × cred  — liquidity crunch
    "cfo_x_interest",            # (CFO/TA) × interest_rate                — cash flow vs. int burden
    "log_assets_x_gdp",          # log(Total Assets) × gdp_growth          — firm size cyclicality
]

# Full feature pool — combined F-classif + mutual-info selects top-20 inside each fold.
# Pool = financial ratios + micro trends + interactions + macro (current + lag-1 + lag-2 + Δ trends)
# Literature basis: Shumway (2001), Duffie et al. (2007), Campbell et al. (2008)
CANDIDATE_FEATURES = (
    CANDIDATE_RATIOS          # 18 financial ratios
    + TREND_FEATURES          # 4 micro momentum features
    + INTERACTION_FEATURES    # 11 micro × macro interaction terms
    + MACRO_FEATURES          # 9 current-year macro indicators
    + MACRO_LAG_FEATURES      # 9 lag-1 macro indicators
    + MACRO_TREND_FEATURES    # 5 macro year-over-year change (Δt) — Campbell 2008
    + MACRO_LAG2_FEATURES     # 3 lag-2 macro (GDP, interest, inflation) — Duffie 2007
                              # Total: 59 candidates → top-20 selected per fold
)

TARGET = "bankruptcy_label"

# ── Feature selection ─────────────────────────────────────────────────────
FEATURE_SELECTION_METHOD = "f_classif"
FEATURE_SELECTION_K = 20   # Top-20 per Büyükarıkan & Büyükarıkan (2025) Scenario 2

# ── Class imbalance ───────────────────────────────────────────────────────
SMOTE_THRESHOLD = 0.40     # Apply SMOTE if distress ratio < 40% (minority class)

# ── Cross-validation ──────────────────────────────────────────────────────
# Outer CV: temporal walk-forward (expanding window, 5 folds: test years 2020-2024)
# Inner CV: stratified k-fold for hyperparameter tuning (on training data only)
CV_FOLDS = 5          # inner-CV folds for hyperparameter search
INNER_CV_FOLDS = 5    # alias used by stacking inner loop
TEMPORAL_MIN_TRAIN_YEARS = 2   # minimum years in training before first test fold
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

# ── Optuna hyperparameter search spaces ───────────────────────────────────
# Used instead of PARAM_GRIDS when Optuna is available.
# OptunaSearchCV accepts optuna.distributions objects.
# N_OPTUNA_TRIALS: number of TPE trials per outer fold per model.
# Wider continuous ranges replace the coarse discrete grids.
N_OPTUNA_TRIALS = 50

try:
    from optuna.distributions import (
        FloatDistribution, IntDistribution, CategoricalDistribution,
    )

    OPTUNA_PARAM_SPACES = {
        "Logistic Regression": {
            "C": FloatDistribution(1e-3, 1e2, log=True),
        },
        "Random Forest": {
            "n_estimators":      IntDistribution(100, 500),
            "max_depth":         CategoricalDistribution([5, 8, 10, None]),
            "min_samples_split": IntDistribution(2, 10),
            "min_samples_leaf":  IntDistribution(1, 5),
        },
        "XGBoost": {
            "n_estimators":     IntDistribution(100, 500),
            "max_depth":        IntDistribution(3, 8),
            "learning_rate":    FloatDistribution(0.01, 0.3, log=True),
            "subsample":        FloatDistribution(0.6, 1.0),
            "colsample_bytree": FloatDistribution(0.6, 1.0),
            "min_child_weight": IntDistribution(1, 10),
            "gamma":            FloatDistribution(0.0, 1.0),
        },
        "CatBoost": {
            "iterations":    IntDistribution(200, 800),
            "depth":         IntDistribution(4, 8),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "l2_leaf_reg":   FloatDistribution(1.0, 10.0),
        },
        "LightGBM": {
            "n_estimators":      IntDistribution(100, 500),
            "max_depth":         IntDistribution(3, 8),
            "learning_rate":     FloatDistribution(0.01, 0.3, log=True),
            "num_leaves":        IntDistribution(20, 100),
            "min_child_samples": IntDistribution(5, 30),
            "subsample":         FloatDistribution(0.6, 1.0),
            "colsample_bytree":  FloatDistribution(0.6, 1.0),
        },
    }
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_PARAM_SPACES = {}
    OPTUNA_AVAILABLE = False

# ── Evaluation metrics ────────────────────────────────────────────────────
METRICS = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
PRIMARY_METRIC = "auc"
SECONDARY_METRIC = "f1"
