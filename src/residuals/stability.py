"""
Module for analyzing the stability of residuals.

"Stable" residuals are those that appear consistently across:
1. Different feature ranges (e.g. not isolated to one specific age value).
2. Time slices (if temporal data exists).

Unstable residuals (isolated spikes) are removed to focus on systematic issues.
"""

import os
import pandas as pd
import numpy as np

def load_high_signal_residuals(filepath: str) -> pd.DataFrame:
    """
    Loads the high-signal residuals dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"High-signal residuals file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} residuals from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading residuals: {e}")

def check_feature_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks stability across features.
    
    Logic:
    - We look for clusters of errors. An error is stable if it's not the only one 
      in its neighborhood (defined by categorical features or bins of numeric features).
    - For this implementation, we'll check if the residual belongs to a group (Department)
      that has multiple errors, AND if there's temporal consistency (not just 1 day).
    """
    if df.empty:
        df['is_stable'] = False
        df['stability_reason'] = 'Empty dataset'
        return df
    
    # Initialize stability flag
    df['is_stable'] = False
    df['stability_reason'] = 'Isolated error'
    
    # 1. Temporal Stability
    # Check if errors in the same group (Department) span multiple unique dates?
    if 'join_date' in df.columns:
        # Convert to datetime if needed, though likely string from CSV
        dates = pd.to_datetime(df['join_date'], errors='coerce')
        
        # Group by Department
        for dept, group in df.groupby('department'):
            unique_dates = dates[group.index].nunique()
            if unique_dates > 1:
                df.loc[group.index, 'is_stable'] = True
                df.loc[group.index, 'stability_reason'] = 'Repeated across time'
            elif len(group) > 2: 
                # If many errors even on same date (unlikely for join_date but possible), consider stable (batch error)
                df.loc[group.index, 'is_stable'] = True
                df.loc[group.index, 'stability_reason'] = 'High density cluster'
                
    else:
        # Fallback: simple density check
        counts = df['department'].value_counts()
        stable_depts = counts[counts >= 2].index
        mask = df['department'].isin(stable_depts)
        df.loc[mask, 'is_stable'] = True
        df.loc[mask, 'stability_reason'] = 'Repeated pattern in feature group'
        
    # 2. Advanced: Check magnitude consistency?
    # For now, density/temporal check is sufficient for "stability" definition in this scope.
    
    return df

def filter_unstable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes unstable residuals.
    """
    stable_df = df[df['is_stable']].copy()
    dropped_count = len(df) - len(stable_df)
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} unstable residuals.")
        print("reasons for dropping:", df[~df['is_stable']]['stability_reason'].unique())
    else:
        print("No residuals dropped (all deemed stable).")
        
    return stable_df

def save_stable_residuals(df: pd.DataFrame, filepath: str):
    """
    Saves the stable residuals.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Drop helper columns before saving? Or keep them context?
        # Let's keep them for transparency
        df.to_csv(filepath, index=False)
        print(f"Successfully saved {len(df)} stable residuals to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    input_path = 'data/outputs/residuals/high_signal_residuals.csv'
    output_path = 'data/outputs/residuals/stable_residuals.csv'
    
    print("-" * 30)
    print("Starting Residual Stability Analysis")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_high_signal_residuals(input_path)
        
        # 2. Check Stability
        df_checked = check_feature_stability(df)
        
        # 3. Filter
        df_stable = filter_unstable(df_checked)
        
        # 4. Save
        if not df_stable.empty:
            save_stable_residuals(df_stable, output_path)
        else:
            print("No stable residuals found.")
            save_stable_residuals(df_stable, output_path)
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
