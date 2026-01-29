"""
Module for estimating prediction confidence.

This module provides functionality to:
1. Load a trained model and processed dataset.
2. Estimate uncertainty using the standard deviation of predictions across the ensemble trees (for RandomForest).
3. Save confidence scores.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def load_artifacts(model_path: str, data_path: str):
    """
    Loads the trained model and dataset.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
        
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        print(f"Successfully loaded model from {model_path} and data from {data_path}")
        return model, df
    except Exception as e:
        raise Exception(f"Error loading artifacts: {e}")

def preprocess_features(df: pd.DataFrame, target_col: str = 'target'):
    """
    Prepares features for the model using similar logic to training.
    """
    data = df.copy()
    
    # Rename target column if it exists and is different
    original_target = 'salary'
    if original_target in data.columns and target_col != original_target:
        data = data.rename(columns={original_target: target_col})
        
    # We might not need y, but we definitely need X
    if target_col in data.columns:
        X = data.drop(columns=[target_col])
    else:
        X = data.copy()
    
    # Handle Date Columns: Same logic as train_model.py
    if 'join_date' in X.columns:
        X['join_date'] = pd.to_datetime(X['join_date'], errors='coerce')
        X['join_year'] = X['join_date'].dt.year
        X['join_month'] = X['join_date'].dt.month
        
        # Imputation (using simple median as in training)
        # In a real pipeline, we should load the imputer fitted during training.
        # For this exercise, we re-impute or assume data is clean enough.
        # Since cleaned_dataset.csv is already imputed for missing values, we just need to handle the new date features.
        X['join_year'] = X['join_year'].fillna(X['join_year'].median())
        X['join_month'] = X['join_month'].fillna(X['join_month'].median())
        X = X.drop(columns=['join_date'])
        
    return X

def estimate_confidence(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates confidence scores.
    
    Logic:
    - For RandomForestRegressor, we can access individual trees via `estimators_`.
    - We calculate the standard deviation of predictions across all trees.
    - Low StdDev = High Confidence (trees agree).
    - High StdDev = Low Confidence (trees disagree).
    """
    confidence_df = pd.DataFrame()
    
    try:
        # Access the regressor step
        # Assuming the pipeline structure: 'preprocessor', 'model'
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']
        
        # Transform data
        X_transformed = preprocessor.transform(X)
        
        if isinstance(model, RandomForestRegressor):
            print("Model is RandomForestRegressor. Computing ensemble variance...")
            
            # Get predictions from all trees
            # estimators_ is a list of DecisionTreeRegressors
            predictions = []
            for estimator in model.estimators_:
                predictions.append(estimator.predict(X_transformed))
            
            predictions = np.array(predictions)
            
            # Calculate standard deviation across trees (axis 0)
            prediction_std = np.std(predictions, axis=0)
            prediction_mean = np.mean(predictions, axis=0)
            
            confidence_df['prediction_mean'] = prediction_mean
            confidence_df['uncertainty_std'] = prediction_std
            # A simple confidence score could be inverse of std, but std itself is the uncertainty measure.
            
        else:
            print(f"Model type {type(model).__name__} does not support tree variance estimation in this script.")
            # Fallback for models without direct sub-estimators access like basic XGB/LGB interface in pipeline
            prediction_mean = pipeline.predict(X)
            confidence_df['prediction_mean'] = prediction_mean
            confidence_df['uncertainty_std'] = 0.0 # Placeholder
            
        return confidence_df
        
    except Exception as e:
        print(f"Error computing confidence: {e}")
        return pd.DataFrame()

def main():
    """
    Main execution flow.
    """
    model_path = 'models/best_model.pkl'
    data_path = 'data/processed/cleaned_dataset.csv'
    output_path = 'data/outputs/confidence.csv'
    
    print("-" * 30)
    print("Starting Confidence Estimation")
    print("-" * 30)
    
    try:
        # 1. Load Artifacts
        model_pipeline, df = load_artifacts(model_path, data_path)
        
        # 2. Preprocess Data
        # Ensure we send the same structure as training
        X = preprocess_features(df)
        
        # 3. Estimate Confidence
        confidence_scores = estimate_confidence(model_pipeline, X)
        
        if not confidence_scores.empty:
            # Add IDs if available for reference
            if 'id' in df.columns:
                confidence_scores.insert(0, 'id', df['id'])
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            confidence_scores.to_csv(output_path, index=False)
            print(f"Successfully saved confidence scores to {output_path}")
            print("\nFirst 5 rows:")
            print(confidence_scores.head())
        else:
            print("Failed to generate confidence scores.")
            
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
