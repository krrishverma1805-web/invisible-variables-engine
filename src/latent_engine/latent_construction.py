"""
Module for constructing latent variables from validated error patterns.

Each validated cluster represents a potential "Invisible Variable" (Latent Variable).
We construct a score for each latent variable to quantify its presence in each sample.

Method:
1. For each sample in a cluster, LatentScore = AbsResidual * Confidence.
   (High error + High confidence = Strong evidence of the latent variable).
2. For samples NOT in the cluster, LatentScore = 0.
3. Scores are normalized to [0, 1] for interpretability.

Interpretation Limits:
- These are *proxy* variables constructed from errors.
- They indicate *where* the model failed systematically.
- High correlation with an external unobserved feature suggests the latent variable *is* that feature.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_patterns(filepath: str) -> pd.DataFrame:
    """
    Loads validated patterns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Validated patterns file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} pattern samples from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def construct_latent_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs latent variable scores from pattern clusters.
    Returns a DataFrame with 'id' and 'latent_var_X' columns.
    """
    if df.empty:
        return pd.DataFrame(columns=['id'])
        
    if 'cluster_id' not in df.columns or 'id' not in df.columns:
        raise ValueError("Input dataframe must contain 'cluster_id' and 'id'.")
        
    # Calculate raw score
    # Score = Magnitude * Confidence
    # This represents the "strength" of the anomaly
    df = df.copy()
    df['raw_score'] = df['abs_residual'] * df['confidence_score']
    
    # Create pivot table
    # Index: id
    # Columns: cluster_id (prefixed with latent_var_)
    # Values: raw_score
    
    latent_df = df.pivot(index='id', columns='cluster_id', values='raw_score')
    
    # Fill missing with 0 (samples not in a specific cluster have 0 score for that latent variable)
    latent_df = latent_df.fillna(0.0)
    
    # Rename columns
    latent_df.columns = [f"latent_var_{c}" for c in latent_df.columns]
    
    # Reset index to make 'id' a column
    latent_df = latent_df.reset_index()
    
    print(f"Constructed scores for {len(latent_df.columns) - 1} latent variables.")
    return latent_df

def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes latent scores to [0, 1].
    """
    if df.empty or len(df.columns) <= 1: # Only 'id' or empty
        return df
        
    # Select feature columns (exclude 'id')
    feature_cols = [c for c in df.columns if c.startswith('latent_var_')]
    
    if not feature_cols:
        return df
        
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print("Normalized latent scores to [0, 1].")
    return df

def save_scores(df: pd.DataFrame, filepath: str):
    """
    Saves latent scores.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Successfully saved latent scores to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    input_path = 'data/outputs/residuals/validated_patterns.csv'
    output_path = 'data/outputs/latent_variables/latent_scores.csv'
    
    print("-" * 30)
    print("Starting Latent Variable Construction")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_patterns(input_path)
        
        # 2. Construct Scores
        latent_df = construct_latent_scores(df)
        
        # 3. Normalize
        latent_df = normalize_scores(latent_df)
        
        # 4. Save
        if not latent_df.empty:
            save_scores(latent_df, output_path)
        else:
            print("No latent variables constructed.")
            # Save empty file with id column
            pd.DataFrame(columns=['id']).to_csv(output_path, index=False)
            
        # Preview
        if not latent_df.empty:
            print("\nSample Latent Scores:")
            print(latent_df.head())
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
