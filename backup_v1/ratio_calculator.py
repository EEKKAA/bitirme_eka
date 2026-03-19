"""
Module to calculate financial ratios from extracted financial items.
"""
import pandas as pd
import numpy as np

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two pandas Series, avoiding division by zero.
    
    Args:
        numerator (pd.Series): The numerator series.
        denominator (pd.Series): The denominator series.
        
    Returns:
        pd.Series: Result of the division, with np.nan where denominator is 0.
    """
    return pd.Series(
        np.where(denominator == 0, np.nan, numerator / denominator),
        index=numerator.index
    )

def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard financial ratios for the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing financial items.
        
    Returns:
        pd.DataFrame: DataFrame with the calculated ratios added.
    """
    df = df.copy()
    
    # 1. Liquidity
    # Current Ratio = Current Assets / Current Liabilities
    df['current_ratio'] = safe_divide(df.get('Current Assets', pd.Series(dtype=float)), 
                                      df.get('Current Liabilities', pd.Series(dtype=float)))
    
    # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    # If Inventory is missing, we approximate by skipping it or yielding NaN.
    # Here, if Inventory is not present, we assume it's 0 (meaning Quick Ratio = Current Ratio).
    inventory = df.get('Inventory', pd.Series(0, index=df.index))
    df['quick_ratio'] = safe_divide(df.get('Current Assets', pd.Series(dtype=float)) - inventory, 
                                    df.get('Current Liabilities', pd.Series(dtype=float)))
    
    # 2. Leverage
    # Debt / Assets = Total Debt / Total Assets
    df['debt_to_assets'] = safe_divide(df.get('Total Debt', pd.Series(dtype=float)), 
                                       df.get('Total Assets', pd.Series(dtype=float)))
    
    # Debt / Equity = Total Debt / Equity
    df['debt_to_equity'] = safe_divide(df.get('Total Debt', pd.Series(dtype=float)), 
                                       df.get('Equity', pd.Series(dtype=float)))
    
    # 3. Profitability
    # ROA = Net Income / Total Assets
    df['roa'] = safe_divide(df.get('Net Income', pd.Series(dtype=float)), 
                            df.get('Total Assets', pd.Series(dtype=float)))
    
    # ROE = Net Income / Equity
    df['roe'] = safe_divide(df.get('Net Income', pd.Series(dtype=float)), 
                            df.get('Equity', pd.Series(dtype=float)))
    
    # Net Profit Margin = Net Income / Revenue
    df['net_profit_margin'] = safe_divide(df.get('Net Income', pd.Series(dtype=float)), 
                                          df.get('Revenue', pd.Series(dtype=float)))
    
    # 4. Efficiency
    # Asset Turnover = Revenue / Total Assets
    df['asset_turnover'] = safe_divide(df.get('Revenue', pd.Series(dtype=float)), 
                                       df.get('Total Assets', pd.Series(dtype=float)))
    
    return df
