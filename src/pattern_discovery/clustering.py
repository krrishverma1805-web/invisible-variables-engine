"""
Module for grouping residuals into clusters to discover error patterns.

We use DBSCAN (Density-Based Spatial Clustering of Applications with Noise) because:
1. It can find arbitrarily shaped clusters (linear, spherical, etc.).
2. It has a notion of "noise" (outliers that don't belong to any cluster), which is consistent with our logic of finding *patterns* (clusters) vs isolated errors (noise).
3. We don't need to specify the number of clusters (K) beforehand.

Clusters of high-signal residuals indicate shared characteristics that likely point to a specific "Invisible Variable".
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_stable_residuals(filepath: str) -> pd.DataFrame:
    """
    Loads stable residuals.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stable residuals file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} residuals from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def cluster_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clusters residuals using DBSCAN.
    """
    if df.empty:
        df['cluster_id'] = -1
        return df
        
    # Select features for clustering
    # We want to cluster based on WHERE errors happen (features) and HOW BIG they are (abs_residual)
    # Ideally, we should check which features contribute to the variance.
    # For this standard implementation, we use all available numeric features + residuals.
    
    # 1. Identify relevant numeric columns
    # Exclude IDs, target (unless we want to cluster by target range), and metadata columns
    exclude_cols = ['id', 'is_stable', 'stability_reason', 'residual', 'target_predicted', 'target_actual', 'uncertainty_std']
    
    numeric_df = df.select_dtypes(include=['number'])
    feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]
    
    # Ensure abs_residual is included as it's a key dimension of the error pattern
    if 'abs_residual' not in feature_cols and 'abs_residual' in df.columns:
        feature_cols.append('abs_residual')
        
    print(f"Clustering based on features: {feature_cols}")
    
    if not feature_cols:
        print("No features for clustering found.")
        df['cluster_id'] = -1
        return df
        
    X = df[feature_cols].copy()
    
    # Fill NaNs if any remaining (should be clean, but safety first)
    X = X.fillna(X.median())
    
    # 2. Scale Features
    # DBSCAN is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Apply DBSCAN
    # Parameters Choice:
    # eps=0.5: Standard default for standardized data. Points within 0.5 std devs are neighbors.
    # min_samples=2: We are looking for PATTERNS. Even 2 repeated errors can be a pattern in small data.
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(X_scaled)
    
    df['cluster_id'] = clusters
    
    # Summary
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    
    return df

def save_clustered_data(df: pd.DataFrame, filepath: str):
    """
    Saves clustered residuals.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Successfully saved clustered residuals to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    input_path = 'data/outputs/residuals/stable_residuals.csv'
    output_path = 'data/outputs/residuals/clustered_residuals.csv'
    
    print("-" * 30)
    print("Starting Pattern Discovery (Clustering)")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_stable_residuals(input_path)
        
        # 2. Cluster
        df_clustered = cluster_residuals(df)
        
        # 3. Save
        save_clustered_data(df_clustered, output_path)
        
        # Preview Clusters
        if 'cluster_id' in df_clustered.columns:
            print("\nCluster Distribution:")
            print(df_clustered['cluster_id'].value_counts())
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
