"""
Module for validating the utility of discovered latent variables via model retraining.

We compare two models:
1. Baseline: Trained on original features.
2. Augmented: Trained on Original Features + Latent Variables.

If the Latent Variables capture real systematic signal, the Augmented model should significantly outperform the Baseline.
NOTE: Since latent variables are derived from the *training* errors of the baseline, some improvement is expected (leakage/overfitting risk).
However, large improvements indicate that the "Invisible Variable" is a strong predictor that was previously missing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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

def get_pipeline(categorical_cols):
    """
    Returns a standard preprocessing + model pipeline.
    """
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, lambda x: x.select_dtypes(include=['int64', 'float64']).columns),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline

def train_evaluate(X, y, name="Model"):
    """
    Evaluates a model using Cross-Validation and returns mean RMSE.
    """
    print(f"\nEvaluating {name}...")
    
    # Identify categorical columns for pipeline
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    pipeline = get_pipeline(categorical_cols)
    
    # Neg MSE because cross_val_score maximize score
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    print(f"  RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    return mean_rmse

def compare_models(df_features: pd.DataFrame, df_latent: pd.DataFrame):
    """
    Compares Baseline vs Augmented models.
    """
    # Merge Data
    if 'id' not in df_features.columns or 'id' not in df_latent.columns:
        raise ValueError("Both datasets must have an 'id' column.")
        
    # Full merge
    merged = pd.merge(df_features, df_latent, on='id', how='inner')
    
    if merged.empty:
        raise ValueError("Merged dataset is empty.")
    
    target_col = 'salary' # Hardcoded for this pipeline
    if target_col not in merged.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    y = merged[target_col]
    
    # --- 1. Baseline Model ---
    # Exclude latent variables and ID
    latent_cols = [c for c in df_latent.columns if c.startswith('latent_var_')]
    exclude_baseline = latent_cols + ['id', target_col]
    X_baseline = merged.drop(columns=exclude_baseline, errors='ignore')
    
    print(f"Baseline Features: {list(X_baseline.columns)}")
    rmse_baseline = train_evaluate(X_baseline, y, name="Baseline Model")
    
    # --- 2. Augmented Model ---
    # Include latent variables
    exclude_augmented = ['id', target_col]
    X_augmented = merged.drop(columns=exclude_augmented, errors='ignore')
    
    print(f"Augmented Features: {list(X_augmented.columns)}")
    rmse_augmented = train_evaluate(X_augmented, y, name="Augmented Model (with Latent Vars)")
    
    # --- Comparison ---
    improvement = rmse_baseline - rmse_augmented
    pct_improvement = (improvement / rmse_baseline) * 100
    
    results = {
        'baseline_rmse': rmse_baseline,
        'augmented_rmse': rmse_augmented,
        'rmse_improvement': improvement,
        'pct_improvement': pct_improvement
    }
    
    print("\nComparison Results:")
    print(f"  Baseline RMSE: {rmse_baseline:.2f}")
    print(f"  Augmented RMSE: {rmse_augmented:.2f}")
    print(f"  Improvement: {improvement:.2f} ({pct_improvement:.2f}%)")
    
    if pct_improvement > 5.0:
        print("  -> SIGNIFICANT IMPROVEMENT: Latent variables are adding strong predictive value.")
    elif pct_improvement > 0:
        print("  -> MARGINAL IMPROVEMENT: Latent variables help slightly.")
    else:
        print("  -> NO IMPROVEMENT: Latent variables might be noise or fully redundant.")
        
    return results

def save_comparison(results: dict, filepath: str):
    """
    Saves comparison results.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame([results])
        df.to_csv(filepath, index=False)
        print(f"\nSuccessfully saved comparison results to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving data: {e}")

def main():
    """
    Main execution flow.
    """
    output_path = 'data/outputs/latent_variables/retraining_comparison.csv'
    
    print("-" * 30)
    print("Starting Retraining Validation Test")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df_features, df_latent = load_data()
        
        # 2. Compare Models
        results = compare_models(df_features, df_latent)
        
        # 3. Save Output
        save_comparison(results, output_path)
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
