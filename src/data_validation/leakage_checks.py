"""
Module for checking data leakage in datasets.

This module provides functionality to:
1. Check for target leakage (high correlation with target).
2. Check for time leakage (future data in past records).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def check_target_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.95) -> None:
    """
    Checks for features that are highly correlated with the target variable,
    which might indicate target leakage.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        threshold: Correlation threshold for flagging potential leakage.
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Skipping target leakage check.")
        return

    # Select only numeric columns for correlation check
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        print("No numeric columns found to check correlation.")
        return

    if target_col not in numeric_df.columns:
        print(f"Target column '{target_col}' is not numeric. Skipping correlation check.")
        return

    correlations = numeric_df.corr()[target_col].drop(target_col).abs()
    leaky_features = correlations[correlations > threshold].index.tolist()
    
    if leaky_features:
        print(f"\n[WARNING] Potential Target Leakage Detected!")
        print(f"The following features have >{threshold} correlation with '{target_col}':")
        for feature in leaky_features:
            print(f"- {feature} (Correlation: {correlations[feature]:.4f})")
        print("Please investigate these features manually.")
    else:
        print(f"\nNo target leakage detected (Threshold: {threshold}).")

def check_time_leakage(df: pd.DataFrame, time_col: str) -> None:
    """
    Checks for time-based anomalies, such as future dates.

    Args:
        df: Input DataFrame.
        time_col: Name of the time column.
    """
    if time_col not in df.columns:
        print(f"Time column '{time_col}' not found. Skipping time leakage check.")
        return

    try:
        # Convert to datetime if not already
        dates = pd.to_datetime(df[time_col], errors='coerce')
        
        current_date = datetime.now()
        future_dates = dates[dates > current_date]
        
        if not future_dates.empty:
            print(f"\n[WARNING] Future Dates Detected!")
            print(f"Found {len(future_dates)} records with dates in the future (relative to system time).")
            print(f"First 5 future dates:\n{future_dates.head()}")
        else:
            print(f"\nNo future dates detected in '{time_col}'.")
            
    except Exception as e:
        print(f"Error checking time leakage: {e}")

def main():
    """
    Main function to execute the leakage check pipeline.
    """
    data_path = 'data/processed/cleaned_dataset.csv'
    target_column = 'salary'  # Assumed target for demo
    time_column = 'join_date' # Assumed time column for demo
    
    print("-" * 30)
    print("Starting Leakage Check Pipeline")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_processed_data(data_path)
        
        # 2. Check Target Leakage
        check_target_leakage(df, target_column)
        
        # 3. Check Time Leakage
        check_time_leakage(df, time_column)
        
        print("\nLeakage checks completed.")
        
    except FileNotFoundError as fnf:
        print(f"\nFile Error: {fnf}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
