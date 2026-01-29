"""
Module for filtering residuals to identify high-signal errors.

High-signal residuals are those that:
1. Have large magnitude (top 20% of absolute errors).
2. Have high confidence (model is certain about its prediction, yet wrong).
3. Are not isolated incidents (part of a pattern).

These residuals are prime candidates for uncovering "Invisible Variables" or missing features.
"""

import os
import pandas as pd
import numpy as np

def load_residuals(filepath: str) -> pd.DataFrame:
    """
    Loads the residuals dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Residuals file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} residuals from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading residuals: {e}")

def calculate_confidence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts uncertainty (std dev) to a confidence score (0-1).
    Lower uncertainty = Higher confidence.
    
    Formula: 1 - MinMaxScaled(uncertainty)
    """
    if 'uncertainty_std' not in df.columns:
        print("Warning: 'uncertainty_std' not found. Assuming confidence=1.0 for all.")
        df['confidence_score'] = 1.0
        return df
    
    std = df['uncertainty_std']
    
    # Avoid division by zero if all values are same
    if std.max() == std.min():
        df['confidence_score'] = 0.5 # Neutral confidence if no variance in uncertainty
    else:
        # MinMax Scale to 0-1
        scaled_uncertainty = (std - std.min()) / (std.max() - std.min())
        # Invert so low uncertainty -> high confidence
        df['confidence_score'] = 1.0 - scaled_uncertainty
        
    return df

def filter_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters residuals based on magnitude, confidence, and recurrence.
    """
    # 1. Magnitude Filter (Top 20% abs residual)
    threshold_mag = df['abs_residual'].quantile(0.8)
    high_magnitude = df[df['abs_residual'] >= threshold_mag]
    print(f"Magnitude Filter (>={threshold_mag:.2f}): Reduced from {len(df)} to {len(high_magnitude)}")
    
    # 2. Confidence Filter (Score >= 0.7)
    # We need to compute confidence on the whole DF first to have correct scaling
    high_magnitude_conf = high_magnitude[high_magnitude['confidence_score'] >= 0.7]
    print(f"Confidence Filter (>=0.7): Reduced from {len(high_magnitude)} to {len(high_magnitude_conf)}")
    
    # 3. Isolated Error Check
    # Remove errors that appear only once per categorical group?
    # Requirement: "Remove residuals that appear only once (isolated errors)"
    # This implies looking for repeated patterns. Let's group by categorical keys.
    # We'll use 'department' as the key for this demo.
    
    if high_magnitude_conf.empty:
        return high_magnitude_conf
        
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'department' in categorical_cols: # Heuristic for demo
        group_col = 'department'
    elif categorical_cols:
        group_col = categorical_cols[0]
    else:
        print("No categorical columns found for isolation check. Skipping.")
        return high_magnitude_conf

    # Count occurrences in the filtered set
    counts = high_magnitude_conf[group_col].value_counts()
    
    # Keep only those appearing > 1 time (not isolated to a single row in that category)
    # Note: If the dataset is small (like 10 rows), this might filter everything.
    # For robust demo, let's just warn if we filter too much.
    
    valid_groups = counts[counts > 1].index.tolist()
    final_df = high_magnitude_conf[high_magnitude_conf[group_col].isin(valid_groups)]
    
    print(f"Isolation Filter (Grouped by {group_col}): Reduced from {len(high_magnitude_conf)} to {len(final_df)}")
    
    return final_df

def save_filtered_data(df: pd.DataFrame, filepath: str):
    """
    Saves the filtered residuals.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Successfully saved {len(df)} high-signal residuals to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    input_path = 'data/outputs/residuals/residuals.csv'
    output_path = 'data/outputs/residuals/high_signal_residuals.csv'
    
    print("-" * 30)
    print("Starting Residuals Filtering")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_residuals(input_path)
        
        # 2. Compute Confidence Score
        df = calculate_confidence_score(df)
        
        # 3. Filter Residuals
        df_filtered = filter_residuals(df)
        
        # 4. Save Output
        if not df_filtered.empty:
            save_filtered_data(df_filtered, output_path)
        else:
            print("No high-signal residuals found after filtering.")
            # Create empty file to ensure workflow consistency
            save_filtered_data(df_filtered, output_path)
            
        print("\nSummary:")
        print(f"Total Residuals: {len(df)}")
        print(f"High-Signal Residuals: {len(df_filtered)}")
        
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
