# Financial Bankruptcy Prediction System

This is a complete, production-quality Python system for analyzing companies' financial statements, computing financial ratios, building a structured dataset, and predicting bankruptcy risk using machine learning. It is designed to be modular, scalable, academically rigorous, and reproducible.

## 1. Project Structure

```
financial_project/
│
├── data/
│   ├── raw/                 # Put your Excel files here
│   │   ├── Company_A/       # One folder per company
│   │   │   ├── 2018.xlsx    # Yearly Excel files containing Balance Sheet & Income Statement
│   │   │   └── 2019.xlsx
│   │   ├── Company_B/
│   │   ├── bankruptcy_labels.csv # (Optional) External file with bankruptcy years
│   │   └── macro_data.csv        # (Optional) External file with macroeconomic indicators by year
│   └── processed/           # Temporary processed files
│
├── outputs/                 # Output artifacts are saved here
│   ├── plots/               # ROC curve, confusion matrices, etc.
│   ├── dataset.csv          # The final consolidated dataset
│   ├── metrics.csv          # Detailed ML evaluation metrics in JSON
│   └── model_comparison.csv # Summary of model performances
│
├── src/                     # Source code modules
│   ├── data_loader.py       # Reads Excel files and folders
│   ├── dataset_builder.py   # Joins data and assigns labels
│   ├── evaluation.py        # Computes ML metrics (Accuracy, F1, AUC, etc.)
│   ├── financial_items.py   # Extracts specific line items from raw statements
│   ├── ml_models.py         # Defines Logistic Regression, Random Forest, XGBoost
│   ├── preprocessing.py     # Handles missing values, outliers, scaling
│   ├── ratio_calculator.py  # Computes financial ratios safely
│   └── visualization.py     # Generates thesis-ready plots
│
├── config.py                # Configuration file (constants, file paths, feature lists)
├── generate_dummy_data.py   # (Utility) Generates dummy data to test the pipeline
├── build_dataset.py         # Script 1: Builds the dataset from raw Excel files
└── train_models.py          # Script 2: Trains and evaluates ML models
```

## 2. Setup and Installation

1. **Install Prerequisites**: Ensure you have Python 3.8+ installed.
2. **Install Required Libraries**:
   Open a terminal and run the following command to install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost openpyxl
   ```

## 3. How to Use the System with Your Data

### Step 3.1: Prepare Your Data
1. Navigate to the `data/raw/` directory.
2. Create one folder for each company (e.g., `Apple`, `Tesla`, `LocalCompany_1`).
3. Inside each company folder, place their yearly financial statements as Excel files (`.xlsx`). Name the files using the year (e.g., `2018.xlsx`, `2019.xlsx`, `2020.xlsx`).
   - *Note*: Each Excel file must contain columns where one column has the financial item name and the adjacent column has the numerical value.

By default, the system looks for the following exact item names in your Excel files (you can change these in `config.py`):
- `Current Assets`
- `Current Liabilities`
- `Total Assets`
- `Total Debt`
- `Equity`
- `Net Income`
- `Revenue`
- `Inventory`

### Step 3.2: Prepare Bankruptcy Labels
1. In the `data/raw/` directory, create a CSV file named `bankruptcy_labels.csv`.
2. This file should have two columns: `company` and `bankruptcy_year`.
   - `company`: Must exactly match the name of the company's folder.
   - `bankruptcy_year`: The year the company went bankrupt. Leave blank or empty for healthy companies.
   
Example `bankruptcy_labels.csv`:
```csv
company,bankruptcy_year
Company_A,
Company_B,
Company_C,2023
Company_D,
Company_E,2022
```

### Step 3.3: Prepare Macroeconomic Data (Optional but Recommended)
1. In the `data/raw/` directory, create a CSV file named `macro_data.csv`.
2. This file should contain macroeconomic indicators by year to merge into the dataset.
   - Required column: `year`
   - Additional feature columns (as configured in `config.py`, by default): `gdp_growth`, `inflation_rate`, `interest_rate`.

Example `macro_data.csv`:
```csv
year,gdp_growth,inflation_rate,interest_rate
2018,3.0,2.4,2.0
2019,2.8,1.8,2.2
2020,-3.5,1.2,0.5
```

### Step 3.4: Build the Dataset
Once your raw data and labels are in place, run the first script to compile everything into a single dataset:
```bash
python build_dataset.py
```
This will:
- Load all Excel files.
- Extract the required line items.
- Calculate all financial ratios safely.
- Assign the `bankruptcy_label` (1 or 0) based on your labels file.
- Merge the macroeconomic indicators by matching the respective year.
- Save the result to `outputs/dataset.csv`.

### Step 3.5: Train Models and Generate Results
After building the dataset, run the machine learning pipeline:
```bash
python train_models.py
```
This will:
- Load the compiled dataset.
- Preprocess the data (impute missing values with median, clip extreme outliers at the 1st and 99th percentiles, and standardize scale).
- Split the data into Training (80%) and Testing (20%) sets.
- Train Logistic Regression, Random Forest, and XGBoost classifiers (handling class imbalance automatically).
- Evaluate models.
- Save metrics to `outputs/metrics.csv` and `outputs/model_comparison.csv`.
- Generate thesis-ready plots in `outputs/plots/` (ROC curves, confusion matrices, feature importance, ratio distributions, and correlation heatmaps).

## 4. Customizing the System

All primary configuration is intentionally isolated in the `config.py` file. If you need to change how the system behaves, look there first:
- **Change extracted items**: Edit `REQUIRED_FINANCIAL_ITEMS`.
- **Change ratios**: If you add a ratio in `src/ratio_calculator.py`, make sure to add its column name to `TARGET_RATIOS` and `FEATURES` in `config.py` so the ML models use it.
- **Change Test Size**: Modify `TEST_SIZE` (default is 0.2).
- **Change Random Seed**: Modify `RANDOM_STATE` (default is 42) to ensure reproducibility.

## 5. Testing with Dummy Data
If you want to test the entire pipeline without using real data yet, you can run the dummy data generator:
```bash
python generate_dummy_data.py
python build_dataset.py
python train_models.py
```
This simulates 5 companies and executes the full end-to-end pipeline.
