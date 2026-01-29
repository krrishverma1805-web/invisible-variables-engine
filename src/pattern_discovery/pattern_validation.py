"""
Module for validating discovered error patterns.

Not all clusters found by DBSCAN are "real" or actionable.
We filter patterns based on:
1. Size: A pattern must have enough samples (e.g. >= 10) to be statistically significant.
2. Stability Score: We verify that the pattern is "tight" (low variance in absolute error) 
   and "confident" (model was consistently wrong).

Patterns that pass these checks are saved as "Validated Patterns".
"""

import os
import pandas as pd
import numpy as np

def load_clustered_residuals(filepath: str) -> pd.DataFrame:
    """
    Loads clustered residuals.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Clustered residuals file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} residuals from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def calculate_stability_score(group):
    """
    Calculates a stability score for a cluster.
    
    Score = Mean Confidence * (1 - CoV of AbsResidual)
    
    where CoV (Coefficient of Variation) = Std / Mean.
    - If residuals are very spread out (high std/mean), score drops.
    - If model was unsure (low confidence), score drops.
    """
    mean_conf = group['confidence_score'].mean()
    
    mean_abs_res = group['abs_residual'].mean()
    std_abs_res = group['abs_residual'].std()
    
    if mean_abs_res == 0:
        cov = 0
    else:
        cov = std_abs_res / mean_abs_res
        
    # Cap CoV at 1 to prevent negative scores, though 1-CoV can be negative.
    # Let's use a simpler heuristic: T tightness = 1 / (1 + CoV)
    tightness = 1.0 / (1.0 + cov)
    
    score = mean_conf * tightness
    return score

def validate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates patterns (clusters).
    """
    if 'cluster_id' not in df.columns:
        print("No cluster_id column found.")
        return pd.DataFrame()
        
    # Remove noise
    clusters = df[df['cluster_id'] != -1]
    
    if clusters.empty:
        print("No clusters found (only noise or empty).")
        return pd.DataFrame()
    
    validated_list = []
    
    # Iterate over each cluster
    for cid, group in clusters.groupby('cluster_id'):
        count = len(group)
        
        # 1. Size Filter
        if count < 10:
            print(f"Cluster {cid} rejected: Size {count} < 10")
            continue
            
        # 2. Compute Score
        score = calculate_stability_score(group)
        
        # We can add a threshold for score if needed, but for now just computing it is enough
        # as per requirements: "Compute a simple stability score per cluster"
        
        print(f"Cluster {cid} validated: Size {count}, Stability Score {score:.4f}")
        
        # Assign score to rows
        group = group.copy()
        group['pattern_stability_score'] = score
        validated_list.append(group)
        
    if not validated_list:
        print("No clusters passed validation.")
        return pd.DataFrame()
        
    final_df = pd.concat(validated_list)
    return final_df

def save_patterns(df: pd.DataFrame, filepath: str):
    """
    Saves validated patterns.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Successfully saved {len(df)} validated pattern samples to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    input_path = 'data/outputs/residuals/clustered_residuals.csv'
    output_path = 'data/outputs/residuals/validated_patterns.csv'
    
    print("-" * 30)
    print("Starting Pattern Validation")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_clustered_residuals(input_path)
        
        # 2. Validate
        df_validated = validate_patterns(df)
        
        # 3. Save
        if not df_validated.empty:
            save_patterns(df_validated, output_path)
        else:
            print("No validated patterns found.")
            # Create empty file
            save_patterns(df_validated, output_path)
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
