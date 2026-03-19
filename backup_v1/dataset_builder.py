"""
Module for building the final dataset from raw company data and labels.
"""
import pandas as pd
from pathlib import Path
from src.data_loader import load_all_companies
from src.ratio_calculator import calculate_ratios
from config import LABELS_FILENAME, MACRO_DATA_FILENAME

def get_bankruptcy_labels() -> pd.DataFrame:
    """
    Loads bankruptcy labels from the external file if it exists.
    
    Returns:
        pd.DataFrame: DataFrame containing bankruptcy labels.
    """
    if LABELS_FILENAME.exists():
        try:
            return pd.read_csv(LABELS_FILENAME)
        except Exception as e:
            print(f"Error reading labels file: {e}")
            
    return pd.DataFrame(columns=['company', 'bankruptcy_year'])

def get_macro_data() -> pd.DataFrame:
    """
    Loads macroeconomic data from the external file if it exists.
    
    Returns:
        pd.DataFrame: DataFrame containing macroeconomic indicators by year.
    """
    if MACRO_DATA_FILENAME.exists():
        try:
            return pd.read_csv(MACRO_DATA_FILENAME)
        except Exception as e:
            print(f"Error reading macro data file: {e}")
            
    return pd.DataFrame(columns=['year'])

def build_dataset(root_folder: Path) -> pd.DataFrame:
    """
    Orchestrates data loading, ratio calculation, and label assignment.
    
    Args:
        root_folder (Path): Directory containing raw company data folders.
        
    Returns:
        pd.DataFrame: The complete panel dataset ready for preprocessing.
    """
    print(f"Loading companies from {root_folder}...")
    df = load_all_companies(root_folder)
    
    if df.empty:
        print("No data found!")
        return df
        
    print("Calculating financial ratios...")
    df = calculate_ratios(df)
    
    # Assign labels
    labels_df = get_bankruptcy_labels()
    
    if not labels_df.empty and 'company' in labels_df.columns:
        print("Merging bankruptcy labels...")
        df = df.merge(labels_df, on='company', how='left')
        
        if 'bankruptcy_year' in df.columns:
            # Went bankrupt later if current year <= bankruptcy_year
            is_bankrupt = df['bankruptcy_year'].notna()
            is_before_bankruptcy = df['year'] <= df['bankruptcy_year']
            df['bankruptcy_label'] = (is_bankrupt & is_before_bankruptcy).astype(int)
            df = df.drop(columns=['bankruptcy_year'])
        else:
            # If no year is provided, assume presence in file means bankrupt
            df['bankruptcy_label'] = df['company'].isin(labels_df['company']).astype(int)
    else:
        print("No external labels found or labels empty. Assigning default 0 to bankruptcy_label.")
        df['bankruptcy_label'] = 0
        
    # Merge macro data
    macro_df = get_macro_data()
    if not macro_df.empty and 'year' in macro_df.columns:
        print("Merging macroeconomic indicators...")
        df = df.merge(macro_df, on='year', how='left')
    else:
        print("No macro data found. Proceeding without macroeconomic features.")
        
    return df
