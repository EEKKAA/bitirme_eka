"""
Script to generate dummy data for testing the financial analysis pipeline.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_DATA_DIR, REQUIRED_FINANCIAL_ITEMS, LABELS_FILENAME, MACRO_DATA_FILENAME

def generate_dummy_data():
    companies = ['Company_A', 'Company_B', 'Company_C', 'Company_D', 'Company_E']
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    np.random.seed(42)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    for company in companies:
        company_dir = RAW_DATA_DIR / company
        os.makedirs(company_dir, exist_ok=True)
        
        is_bankrupt_future = company in ['Company_C', 'Company_E']
        
        for year in years:
            data = []
            assets = np.random.uniform(1000, 5000)
            
            # Simulate deteriorating conditions for bankrupt ones
            if is_bankrupt_future and year >= 2022:
                assets *= 0.5
                liabilities = assets * np.random.uniform(1.2, 2.0)
                net_income = -np.random.uniform(100, 500)
            else:
                liabilities = assets * np.random.uniform(0.3, 0.8)
                net_income = np.random.uniform(100, 500)
                
            equity = assets - liabilities
            revenue = assets * np.random.uniform(0.8, 1.5)
            
            items_dict = {
                "Current Assets": assets * 0.4,
                "Current Liabilities": liabilities * 0.5,
                "Total Assets": assets,
                "Total Debt": liabilities * 0.8,
                "Equity": equity,
                "Net Income": net_income,
                "Revenue": revenue,
                "Inventory": assets * 0.1
            }
            
            for item in REQUIRED_FINANCIAL_ITEMS:
                val = items_dict.get(item, np.random.uniform(10, 100))
                data.append([item, val])
                
            df = pd.DataFrame(data, columns=['Item', 'Value'])
            df.to_excel(company_dir / f"{year}.xlsx", index=False)
            
    # Labels
    labels_data = [
        {'company': 'Company_A', 'bankruptcy_year': np.nan},
        {'company': 'Company_B', 'bankruptcy_year': np.nan},
        {'company': 'Company_C', 'bankruptcy_year': 2023},
        {'company': 'Company_D', 'bankruptcy_year': np.nan},
        {'company': 'Company_E', 'bankruptcy_year': 2022},
    ]
    pd.DataFrame(labels_data).to_csv(LABELS_FILENAME, index=False)
    
    # Macro data
    macro_data = []
    # Base macroeconomic trends
    gdp_trends = {2018: 3.0, 2019: 2.8, 2020: -3.5, 2021: 5.5, 2022: 2.1, 2023: 1.8, 2024: 2.0}
    inflation_trends = {2018: 2.4, 2019: 1.8, 2020: 1.2, 2021: 4.7, 2022: 8.0, 2023: 4.1, 2024: 2.8}
    interest_trends = {2018: 2.0, 2019: 2.2, 2020: 0.5, 2021: 0.1, 2022: 1.5, 2023: 5.0, 2024: 4.5}
    
    for year in years:
        macro_data.append({
            'year': year,
            'gdp_growth': gdp_trends[year] + np.random.uniform(-0.5, 0.5), # Add slight noise
            'inflation_rate': inflation_trends[year] + np.random.uniform(-0.2, 0.2),
            'interest_rate': interest_trends[year] + np.random.uniform(-0.1, 0.1)
        })
    pd.DataFrame(macro_data).to_csv(MACRO_DATA_FILENAME, index=False)
    
    print("Dummy data successfully generated at", RAW_DATA_DIR)

if __name__ == "__main__":
    generate_dummy_data()
