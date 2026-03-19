"""
Preprocessing module for handling missing values, outliers, and scaling.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config import FEATURES

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Imputes missing values in the feature columns.
    
    Args:
        df (pd.DataFrame): Dataset.
        strategy (str): Imputation strategy ('median', 'mean', etc.)
        
    Returns:
        pd.DataFrame: Dataset with missing values imputed.
    """
    imputer = SimpleImputer(strategy=strategy)
    df_out = df.copy()
    
    features_present = [f for f in FEATURES if f in df.columns]
    if features_present:
        df_out[features_present] = imputer.fit_transform(df[features_present])
        
    return df_out

def clip_outliers(df: pd.DataFrame, lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Winsorizes (clips) outliers in the feature columns to limit extreme values.
    
    Args:
        df (pd.DataFrame): Dataset.
        lower_percentile (float): Lower percentile for clipping.
        upper_percentile (float): Upper percentile for clipping.
        
    Returns:
        pd.DataFrame: Dataset with outliers clipped.
    """
    df_out = df.copy()
    features_present = [f for f in FEATURES if f in df.columns]
    
    for col in features_present:
        lower = df_out[col].quantile(lower_percentile)
        upper = df_out[col].quantile(upper_percentile)
        df_out[col] = df_out[col].clip(lower, upper)
        
    return df_out

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes feature columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Dataset.
        
    Returns:
        pd.DataFrame: Dataset with scaled features.
    """
    scaler = StandardScaler()
    df_out = df.copy()
    
    features_present = [f for f in FEATURES if f in df.columns]
    if features_present:
        df_out[features_present] = scaler.fit_transform(df_out[features_present])
        
    return df_out

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline: imputation, outlier clipping, and scaling.
    
    Args:
        df (pd.DataFrame): Raw dataset containing features.
        
    Returns:
        pd.DataFrame: Fully preprocessed dataset ready for ML modeling.
    """
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Clipping outliers...")
    df = clip_outliers(df)
    
    print("Scaling features...")
    df = scale_features(df)
    
    return df
