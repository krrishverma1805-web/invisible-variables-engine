"""
Module for generating visualizations of Invisible Variables.

Generates:
1. Residual Heatmap: Where are the errors concentrated?
2. Latent Distributions: How common is the invisible variable?
3. Impact Curves: Relationship between latent variable score and prediction error.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional plots
sns.set_theme(style="whitegrid")

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads clustered residuals and latent scores.
    """
    residuals_path = 'data/outputs/residuals/clustered_residuals.csv'
    latent_path = 'data/outputs/latent_variables/latent_scores.csv'
    
    if not os.path.exists(residuals_path):
        raise FileNotFoundError(f"Residuals file not found: {residuals_path}")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent scores file not found: {latent_path}")
        
    try:
        df_res = pd.read_csv(residuals_path)
        df_latent = pd.read_csv(latent_path)
        
        # Merge them for plotting
        merged = pd.merge(df_res, df_latent, on='id', how='inner')
        return merged
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def plot_residual_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Plots residuals projected onto top 2 numeric features.
    """
    # Identify top numeric features (variance based or just pick first 2)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Exclude metadata metrics
    exclude = ['id', 'target_predicted', 'target_actual', 'uncertainty_std', 'residual', 'abs_residual', 'confidence_score', 'cluster_id']
    latent_cols = [c for c in df.columns if c.startswith('latent_var_')]
    exclude += latent_cols
    
    features = [c for c in numeric_cols if c not in exclude]
    
    if len(features) < 2:
        print("Not enough numeric features for heatmap.")
        return
        
    x_col, y_col = features[0], features[1]
    
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue='abs_residual', 
        palette='rocket_r',
        size='confidence_score',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title(f'Residual Heatmap: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    filepath = os.path.join(output_dir, 'residual_heatmap.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved residual heatmap to {filepath}")

def plot_latent_distributions(df: pd.DataFrame, output_dir: str):
    """
    Plots distribution of latent variable scores.
    """
    latent_cols = [c for c in df.columns if c.startswith('latent_var_')]
    
    if not latent_cols:
        return
        
    for latent in latent_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[latent], kde=True, bins=20, color='purple')
        plt.title(f'Distribution of {latent}')
        plt.xlabel('Latent Score (0=Absent, 1=Strong)')
        plt.ylabel('Count')
        
        filepath = os.path.join(output_dir, f'{latent}_distribution.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution plot to {filepath}")

def plot_outcome_impact(df: pd.DataFrame, output_dir: str):
    """
    Plots Latent Score vs Residual to show impact.
    """
    latent_cols = [c for c in df.columns if c.startswith('latent_var_')]
    
    if not latent_cols:
        return
        
    for latent in latent_cols:
        plt.figure(figsize=(8, 5))
        sns.regplot(data=df, x=latent, y='residual', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'Impact of {latent} on Prediction Error')
        plt.xlabel(f'{latent} Score')
        plt.ylabel('Residual (Actual - Predicted)')
        
        filepath = os.path.join(output_dir, f'{latent}_impact.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved impact plot to {filepath}")

def main():
    """
    Main execution flow.
    """
    output_dir = 'data/outputs/visualizations'
    
    print("-" * 30)
    print("Generating Visualizations")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_data()
        print(f"Loaded {len(df)} samples.")
        
        # 2. Plots
        plot_residual_heatmap(df, output_dir)
        plot_latent_distributions(df, output_dir)
        plot_outcome_impact(df, output_dir)
        
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
