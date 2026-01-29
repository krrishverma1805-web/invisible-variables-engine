"""
Module for handling missing values in datasets.

This module provides functionality to:
1. Load a dataset.
2. Analyze missing values.
3. Validate if missing values exceed a specific threshold.
4. Impute missing values (median for numeric, mode for categorical).
5. Save the cleaned dataset.
"""

import os
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
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

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes missing values in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table with count and percentage of missing values.
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    summary_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    })
    
    # Filter to show only columns with missing values or show all
    # showing all is better for a complete summary
    return summary_df.sort_values(by='Missing Percentage', ascending=False)

def validate_threshold(df: pd.DataFrame, threshold: float = 40.0) -> None:
    """
    Validates if any column has missing values exceeding the threshold.

    Args:
        df: Input DataFrame.
        threshold: percentage threshold (0-100).

    Raises:
        ValueError: If any column has missing values > threshold.
    """
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    exceeding_cols = missing_percentage[missing_percentage > threshold].index.tolist()
    
    if exceeding_cols:
        raise ValueError(
            f"Columns {exceeding_cols} have more than {threshold}% missing values. "
            "Processing stopped."
        )

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame.
    - Numeric columns: Median
    - Categorical columns: Mode

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"Imputed column '{col}' (numeric) with median: {median_val}")
            else:
                # mode() returns a Series, take the first element
                if not df_clean[col].mode().empty:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    print(f"Imputed column '{col}' (categorical) with mode: {mode_val}")
                else:
                    print(f"Warning: Could not compute mode for column '{col}'. Left as NaN.")
    
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
        print(f"Successfully saved cleaned data to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main function to execute the data validation pipeline.
    """
    input_path = 'data/raw/sample_dataset.csv'
    output_path = 'data/processed/cleaned_dataset.csv'
    
    print("-" * 30)
    print("Starting Data Validation Pipeline")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_data(input_path)
        
        # 2. Analyze Missing Values
        summary = analyze_missing(df)
        print("\nMissing Values Summary:")
        print(summary)
        print("-" * 30)
        
        # 3. Validate Threshold
        validate_threshold(df, threshold=40.0)
        print("Validation checks passed. Proceeding to imputation...")
        
        # 4. Impute Data
        df_clean = impute_data(df)
        
        # 5. Save Data
        save_data(df_clean, output_path)
        
        print("\nPipeline completed successfully.")
        
    except ValueError as ve:
        print(f"\nValidation Error: {ve}")
    except FileNotFoundError as fnf:
        print(f"\nFile Error: {fnf}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
