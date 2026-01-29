"""
Module for validating latent variables by finding proxy features.

"Invisible Variables" are often unobserved, but they might be partially observed
through other features (proxies).
- High correlation with an existing feature -> The latent variable might just be a non-linear transformation of that feature, or that feature is a strong proxy.
- Low correlation with ALL features -> True "Invisible Variable" (or noise).

We compute Pearson correlation to identify these relationships.
WARNING: Correlation != Causation. A proxy feature might just co-occur with the systematic error.
"""

import os
import pandas as pd
import numpy as np

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cleaned feature data and latent scores.
    """
    features_path = 'data/processed/cleaned_dataset.csv'
    latent_path = 'data/outputs/latent_variables/latent_scores.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent scores file not found: {latent_path}")
        
    try:
        df_features = pd.read_csv(features_path)
        df_latent = pd.read_csv(latent_path)
        print(f"Loaded features ({len(df_features)} rows) and latent scores ({len(df_latent)} rows).")
        return df_features, df_latent
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def validate_proxies(df_features: pd.DataFrame, df_latent: pd.DataFrame) -> pd.DataFrame:
    """
    Computes correlations between latent variables and original features.
    """
    # Merge on ID
    # Ensure ID columns are present
    if 'id' not in df_features.columns or 'id' not in df_latent.columns:
        raise ValueError("Both datasets must have an 'id' column.")
        
    merged = pd.merge(df_features, df_latent, on='id', how='inner')
    
    if merged.empty:
        print("Merged dataset is empty. Check IDs.")
        return pd.DataFrame()
    
    # Identify Latent Columns and Feature Columns
    latent_cols = [c for c in df_latent.columns if c.startswith('latent_var_')]
    feature_cols = [c for c in df_features.columns if c != 'id']
    
    # We only care about numeric features for Pearson correlation
    # For categorical features, we'd need ANOVA or similar, but for this scope we'll slice to numeric.
    numeric_features = merged[feature_cols].select_dtypes(include=['number']).columns.tolist()
    
    if not latent_cols:
        print("No latent variables found.")
        return pd.DataFrame()
        
    correlation_results = []
    
    print("\nCorrelation Analysis (Latent Variable vs Features):")
    print("-" * 50)
    
    for latent in latent_cols:
        print(f"\nAnalyzing {latent}:")
        max_corr = 0
        best_proxy = None
        
        for feature in numeric_features:
            # Skip target or the latent variable itself (if it leaked in?)
            if feature == latent: 
                continue
                
            corr = merged[latent].corr(merged[feature])
            
            # Store result
            correlation_results.append({
                'latent_variable': latent,
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
            
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                best_proxy = feature
                
        print(f"  Max Correlation: {max_corr:.4f} (with '{best_proxy}')")
        
        if max_corr < 0.2:
            print("  -> WEAK PROXY: This latent variable appears unique/orthogonal to existing numeric features.")
        elif max_corr > 0.8:
            print("  -> STRONG PROXY: Likely redundant or a direct transformation of an existing feature.")
            
    # Create DataFrame Summary
    results_df = pd.DataFrame(correlation_results)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by=['latent_variable', 'abs_correlation'], ascending=[True, False])
        
    return results_df

def save_validation_summary(df: pd.DataFrame, filepath: str):
    """
    Saves the correlation summary.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\nSuccessfully saved proxy validation summary to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    output_path = 'data/outputs/latent_variables/proxy_validation_summary.csv'
    
    print("-" * 30)
    print("Starting Proxy Validation")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df_features, df_latent = load_data()
        
        # 2. Compute Correlations
        results_df = validate_proxies(df_features, df_latent)
        
        # 3. Save
        if not results_df.empty:
            save_validation_summary(results_df, output_path)
            
            # Print top 5 rows
            print("\nTop Correlations:")
            print(results_df.head())
        else:
            print("No correlation results generated.")
            # Save empty file
            pd.DataFrame(columns=['latent_variable','feature','correlation']).to_csv(output_path, index=False)
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
