"""
Module for performing variance checks on datasets.

This module provides functionality to:
1. Load a processed dataset.
2. Identify numeric features with near-zero variance.
3. Remove those features from the dataset.
4. Save the updated dataset.
"""

import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

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

def remove_low_variance(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Identifies and removes numeric features with variance below the threshold.
    Default threshold is 0.0 (removes constant features).

    Args:
        df: Input DataFrame.
        threshold: Variance threshold. Features with variance <= threshold are removed.

    Returns:
        pd.DataFrame: DataFrame with low variance features removed.
    """
    df_clean = df.copy()
    
    # Select only numeric columns for variance check
    numeric_df = df_clean.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        print("No numeric columns found to check variance.")
        return df_clean

    selector = VarianceThreshold(threshold=threshold)
    
    try:
        selector.fit(numeric_df)
        
        # Get columns to keep
        cols_to_keep_numeric = numeric_df.columns[selector.get_support()]
        cols_to_drop = list(set(numeric_df.columns) - set(cols_to_keep_numeric))
        
        if cols_to_drop:
            print("\nDropping low variance features:")
            for col in cols_to_drop:
                print(f"- {col}")
            
            # Drop from original dataframe (keeping non-numeric columns intact)
            df_clean = df_clean.drop(columns=cols_to_drop)
        else:
            print("\nNo low variance features found.")
            
        return df_clean
        
    except ValueError as e:
        # VarianceThreshold might fail if no features meet the criteria or other issues
        print(f"Error during variance check: {e}")
        return df_clean

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        filepath: Destination path.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Successfully saved updated data to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main function to execute the variance check pipeline.
    """
    data_path = 'data/processed/cleaned_dataset.csv'
    
    print("-" * 30)
    print("Starting Variance Check Pipeline")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_processed_data(data_path)
        
        # 2. Remove Low Variance Features
        df_clean = remove_low_variance(df, threshold=0.0)
        
        # 3. Save Data (Overwrite)
        save_data(df_clean, data_path)
        
        print("\nPipeline completed successfully.")
        
    except FileNotFoundError as fnf:
        print(f"\nFile Error: {fnf}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
