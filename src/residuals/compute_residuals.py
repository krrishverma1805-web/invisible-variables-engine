"""
Module for computing model residuals.

Residuals (Actual - Predicted) are crucial for understanding model performance.
They help identify:
1. Systematic bias (if residuals are not centered around 0).
2. Heteroscedasticity (if variance of residuals changes with feature values).
3. Outliers (large absolute residuals).
4. "Invisible Variables" (patterns in residuals explainable by uncaptured features).

This module combines original features, predictions, and confidence scores to compute
and save residuals for further analysis.
"""

import os
import pandas as pd
import numpy as np

def load_datasets():
    """
    Loads the necessary datasets: cleaned features, predictions, and confidence scores.
    """
    cleaned_path = 'data/processed/cleaned_dataset.csv'
    preds_path = 'data/outputs/predictions.csv'
    conf_path = 'data/outputs/confidence.csv'
    
    if not all(os.path.exists(p) for p in [cleaned_path, preds_path, conf_path]):
        raise FileNotFoundError("One or more required input files are missing.")
        
    try:
        df_clean = pd.read_csv(cleaned_path)
        df_preds = pd.read_csv(preds_path)
        df_conf = pd.read_csv(conf_path)
        
        print("Successfully loaded all datasets.")
        return df_clean, df_preds, df_conf
    except Exception as e:
        raise Exception(f"Error loading datasets: {e}")

def compute_residuals(df_features: pd.DataFrame, df_preds: pd.DataFrame, df_conf: pd.DataFrame) -> pd.DataFrame:
    """
    Merges datasets and computes residuals.
    
    Args:
        df_features: Original features (including target).
        df_preds: Model predictions (expected to have 'target_predicted', 'target_actual').
        df_conf: Confidence scores (expected to have 'uncertainty_std').
        
    Returns:
        pd.DataFrame: Merged DataFrame with 'residual' and 'abs_residual'.
    """
    # Verify alignment
    if not (len(df_features) == len(df_preds) == len(df_conf)):
        print(f"Warning: Dataset lengths differ. Features: {len(df_features)}, Preds: {len(df_preds)}, Conf: {len(df_conf)}")
        # In a strict pipeline, we might raise an error. Here we assume row-wise alignment (same order).
    
    # Start with features
    result = df_features.copy()
    
    # Add predictions
    if 'target_predicted' in df_preds.columns:
        result['target_predicted'] = df_preds['target_predicted']
    else:
        raise ValueError("Predictions file missing 'target_predicted' column.")
        
    # Add actuals (sanity check against features if target is present)
    target_col = 'salary' # Assuming salary is target
    if target_col in result.columns:
        result['target_actual'] = result[target_col]
    elif 'target_actual' in df_preds.columns:
        result['target_actual'] = df_preds['target_actual']
    else:
        raise ValueError("Target actual values not found in features or predictions.")
    
    # Add confidence
    if 'uncertainty_std' in df_conf.columns:
        result['uncertainty_std'] = df_conf['uncertainty_std']
        
    # Compute Residuals
    # Residual = Actual - Predicted
    result['residual'] = result['target_actual'] - result['target_predicted']
    
    # Absolute Residual (magnitude of error)
    result['abs_residual'] = result['residual'].abs()
    
    print("Computed residuals and merged data.")
    return result

def save_residuals(df: pd.DataFrame, filepath: str):
    """
    Saves the residuals dataframe.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Successfully saved residuals to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving residuals: {e}")

def main():
    """
    Main execution flow.
    """
    output_path = 'data/outputs/residuals/residuals.csv'
    
    print("-" * 30)
    print("Starting Residuals Computation")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df_clean, df_preds, df_conf = load_datasets()
        
        # 2. Compute Residuals
        df_residuals = compute_residuals(df_clean, df_preds, df_conf)
        
        # 3. Save Output
        save_residuals(df_residuals, output_path)
        
        print("\nPipeline completed.")
        print("\nSample Output:")
        print(df_residuals[['target_actual', 'target_predicted', 'residual', 'abs_residual', 'uncertainty_std']].head())
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
