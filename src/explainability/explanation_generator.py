"""
Module for generating human-readable explanations for discovered invisible variables.

This module aggregates findings from:
1. Pattern Discovery (What groups are affected?)
2. Proxy Validation (Is it a proxy for an existing feature?)
3. Retraining Test (Does it actually improve the model?)

It outputs a plain English narrative for each discovered latent variable.
"""

import os
import pandas as pd
import json

def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads necessary artifacts for explanation generation.
    Returns: patterns_df, proxies_df, retraining_df
    """
    patterns_path = 'data/outputs/residuals/validated_patterns.csv'
    proxies_path = 'data/outputs/latent_variables/proxy_validation_summary.csv'
    retraining_path = 'data/outputs/latent_variables/retraining_comparison.csv'
    
    if not os.path.exists(patterns_path):
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")
    
    # Optional artifacts (might not exist if prev steps failed/skipped, but we assume success here)
    if os.path.exists(proxies_path):
        proxies_df = pd.read_csv(proxies_path)
    else:
        proxies_df = pd.DataFrame()
        
    if os.path.exists(retraining_path):
        retraining_df = pd.read_csv(retraining_path)
    else:
        retraining_df = pd.DataFrame()
        
    patterns_df = pd.read_csv(patterns_path)
    return patterns_df, proxies_df, retraining_df

def identify_context(cluster_df: pd.DataFrame) -> dict:
    """
    Identifies common characteristics (low variance features) in the cluster.
    """
    context = {}
    
    # Check categorical columns for single values
    cat_cols = cluster_df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_vals = cluster_df[col].dropna().unique()
        if len(unique_vals) == 1:
            context[col] = str(unique_vals[0])
            
    # Check numerical columns for tight ranges (low std dev)
    # This is a bit heuristic; let's stick to categorical for simple "Plain English" context first.
    
    return context

def generate_explanations(patterns_df: pd.DataFrame, proxies_df: pd.DataFrame, retraining_df: pd.DataFrame) -> list:
    """
    Generates a list of explanation dictionaries.
    """
    explanations = []
    
    if 'cluster_id' not in patterns_df.columns:
        return []
        
    # Group by cluster to analyze each pattern
    unique_clusters = patterns_df['cluster_id'].unique()
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
            
        cluster_df = patterns_df[patterns_df['cluster_id'] == cluster_id]
        latent_name = f"latent_var_{cluster_id}"
        
        # 1. Context (Where does it happen?)
        context = identify_context(cluster_df)
        context_desc = ", ".join([f"{k} is {v}" for k, v in context.items()])
        if not context_desc:
            context_desc = "various conditions (no single common categorical feature)"
            
        # 2. Impact (Overprediction vs Underprediction)
        # Residual = Actual - Predicted
        # Mean Res > 0 => Actual > Predicted => Model Underpredicted
        # Mean Res < 0 => Actual < Predicted => Model Overpredicted
        mean_residual = cluster_df['residual'].mean()
        if mean_residual > 0:
            impact = "Systematic Underprediction (Actual values are higher than predicted)"
        else:
            impact = "Systematic Overprediction (Actual values are lower than predicted)"
            
        # 3. Proxy Status
        proxy_info = "Unique Signal"
        if not proxies_df.empty:
            # Check for this latent variable
            relevant_proxy = proxies_df[proxies_df['latent_variable'] == latent_name]
            if not relevant_proxy.empty:
                # Sort by abs_correlation
                best_match = relevant_proxy.iloc[0] # Assumes already sorted in prev step
                corr = best_match['abs_correlation']
                feat = best_match['feature']
                
                if corr > 0.8:
                    proxy_info = f"Redundant (Strongly correlated with '{feat}')"
                elif corr > 0.5:
                    proxy_info = f"Related to '{feat}'"
                else:
                    proxy_info = "Likely a New Unobserved Factor (Weak correlation with existing features)"
                    
        # 4. Retraining Validation
        validation_status = "Not Verified"
        if not retraining_df.empty:
            improvement = retraining_df.iloc[0]['pct_improvement']
            if improvement > 1.0:
                 validation_status = f"Validated (Improves model accuracy by {improvement:.1f}%)"
            else:
                 validation_status = f"Weak Validation (Minimal improvement of {improvement:.1f}%)"

        # Construct Narrative
        narrative = (
            f"Invisible Variable #{cluster_id} affects samples where {context_desc}. "
            f"It causes {impact}. "
            f"Analysis suggests this is {proxy_info.lower()}. "
            f"Validation status: {validation_status}."
        )
        
        explanations.append({
            "id": int(cluster_id),
            "name": latent_name,
            "context": context,
            "impact": impact,
            "proxy_analysis": proxy_info,
            "validation_note": validation_status,
            "narrative": narrative
        })
        
    return explanations

def save_json(data: list, filepath: str):
    """
    Saves explanations to JSON.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved {len(data)} explanations to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    output_path = 'data/outputs/latent_variables/explanations.json'
    
    print("-" * 30)
    print("Starting Explanation Generation")
    print("-" * 30)
    
    try:
        # 1. Load Artifacts
        patterns_df, proxies_df, retraining_df = load_artifacts()
        print(f"Loaded {len(patterns_df)} pattern samples.")
        
        # 2. Generate
        explanations = generate_explanations(patterns_df, proxies_df, retraining_df)
        
        # 3. Save
        save_json(explanations, output_path)
        
        # Print first narrative
        if explanations:
            print("\nSample Narrative:")
            print(explanations[0]['narrative'])
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
