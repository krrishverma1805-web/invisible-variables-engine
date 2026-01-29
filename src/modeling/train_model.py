"""
Module for training machine learning models.

This module provides functionality to:
1. Load the cleaned dataset.
2. Preprocess features (encoding, handling dates).
3. Train RandomForest and XGBoost models.
4. Evaluate models using 5-fold Cross-Validation.
5. Save the best model and predictions.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def preprocess_features(df: pd.DataFrame, target_col: str = 'target'):
    """
    Separates target and features, handles categorical encoding and dates.
    """
    # Create a copy to avoid setting copy warning
    data = df.copy()
    
    # Rename target column if it exists and is different
    # For this demo, assuming 'salary' is the target based on previous steps
    original_target = 'salary'
    if original_target in data.columns and target_col != original_target:
        data = data.rename(columns={original_target: target_col})
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    y = data[target_col]
    X = data.drop(columns=[target_col])
    
    # Handle Date Columns: 'join_date' -> 'join_year', 'join_month'
    if 'join_date' in X.columns:
        X['join_date'] = pd.to_datetime(X['join_date'], errors='coerce')
        X['join_year'] = X['join_date'].dt.year
        X['join_month'] = X['join_date'].dt.month
        # Fill NaNs for dates if any (though imputation handled earlier)
        X['join_year'] = X['join_year'].fillna(X['join_year'].median())
        X['join_month'] = X['join_month'].fillna(X['join_month'].median())
        X = X.drop(columns=['join_date'])
        
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    return X, y, categorical_cols, numeric_cols

def train_evaluate(X, y, categorical_cols):
    """
    Trains models and evaluates them.
    """
    # Preprocessing for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' # Keep numeric columns as is
    )
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_model_score = float('inf')
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Negative MSE is the default scoring for regressors usually, but we can compute manual RMSE
        # cross_val_score returns negative values for errors in sklearn
        neg_mse_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        neg_mae_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
        mae_scores = -neg_mae_scores
        
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)
        
        results[name] = {'RMSE': avg_rmse, 'MAE': avg_mae}
        print(f"{name} Results - RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}")
        
        # Fit on full data for saving
        pipeline.fit(X, y)
        
        if avg_rmse < best_model_score:
            best_model_score = avg_rmse
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with RMSE: {best_model_score:.4f}")
    return best_pipeline, best_model_name

def save_model(model, filepath):
    """
    Saves the trained model using joblib.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Successfully saved model to {filepath}")
    except Exception as e:
        raise Exception(f"Error saving model: {e}")

def main():
    """
    Main execution flow.
    """
    input_path = 'data/processed/cleaned_dataset.csv'
    predictions_path = 'data/outputs/predictions.csv'
    model_path = 'models/best_model.pkl'
    
    print("-" * 30)
    print("Starting Model Training Pipeline")
    print("-" * 30)
    
    try:
        # 1. Load Data
        df = load_data(input_path)
        
        # 2. Preprocess
        X, y, cat_cols, num_cols = preprocess_features(df, target_col='target')
        
        # 3. Train & Evaluate
        best_pipeline, best_model_name = train_evaluate(X, y, cat_cols)
        
        # 4. Save Model
        save_model(best_pipeline, model_path)
        
        # 5. Generate Predictions (on training data for demo purposes/output requirement)
        predictions = best_pipeline.predict(X)
        output_df = X.copy()
        output_df['target_actual'] = y
        output_df['target_predicted'] = predictions
        
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        output_df.to_csv(predictions_path, index=False)
        print(f"Successfully saved predictions to {predictions_path}")
        
        print("\nPipeline completed successfully.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
