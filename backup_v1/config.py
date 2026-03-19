"""
Configuration settings for the Financial Bankruptcy Prediction System.
This module stores paths, constant parameters, and model configurations
to keep the main code free of hard-coded values.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Ensure output directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# File names
DATASET_FILENAME = OUTPUTS_DIR / "dataset.csv"
METRICS_FILENAME = OUTPUTS_DIR / "metrics.csv"
MODEL_COMP_FILENAME = OUTPUTS_DIR / "model_comparison.csv"
LABELS_FILENAME = RAW_DATA_DIR / "bankruptcy_labels.csv" # External labels file if needed
MACRO_DATA_FILENAME = RAW_DATA_DIR / "macro_data.csv" # External macroeconomic data file

# Expected columns in raw excel data
REQUIRED_FINANCIAL_ITEMS = [
    "Current Assets",
    "Current Liabilities",
    "Total Assets",
    "Total Debt",
    "Equity",
    "Net Income",
    "Revenue",
    "Inventory"
]

# Ratios to compute
TARGET_RATIOS = [
    "current_ratio",
    "quick_ratio",
    "debt_to_assets",
    "debt_to_equity",
    "roa",
    "roe",
    "net_profit_margin",
    "asset_turnover"
]

# Macroeconomic features
MACRO_FEATURES = [
    "gdp_growth",
    "inflation_rate",
    "interest_rate"
]

# ML configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features for modeling
FEATURES = TARGET_RATIOS + MACRO_FEATURES
TARGET = "bankruptcy_label"
