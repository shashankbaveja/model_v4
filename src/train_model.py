import pandas as pd
import yaml
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.utils import class_weight
import numpy as np

# Add the root directory for local library import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import load_config


def load_data(strategy, config):
    """Loads feature and target data for a given strategy."""
    print(f"Loading data for strategy: {strategy}")
    
    processed_dir = 'data/processed'
    
    base_train_filename = f'train_{strategy}_with_patterns_features.parquet'
    train_filename = os.path.join(processed_dir, base_train_filename)
    print(f"Train filename: {train_filename}")
    
    try:
        train_df = pd.read_parquet(train_filename)
        y_train = train_df.pop('target')
        X_train = train_df.drop(columns=['instrument_token', 'timestamp'], errors='ignore')

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Have you run the feature generation? Details: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Target column 'target' not found. Details: {e}")
        sys.exit(1)

    return X_train, y_train

def get_model(model_type, random_state, class_weights=None):
    """Returns a model instance based on the type, with optional class weights."""
    if model_type == "lightgbm":
        # Use 'multiclass' objective for multi-class classification
        return lgb.LGBMClassifier(random_state=random_state, objective='multiclass', class_weight=class_weights)
    
    elif model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state, class_weight=class_weights, n_jobs=-1)
        
    elif model_type == "logistic_regression":
        # Logistic Regression benefits from feature scaling and supports multi-class out of the box
        return Pipeline([
            ('scaler', StandardScaler()),
            ('logit', LogisticRegression(random_state=random_state, class_weight=class_weights, solver='lbfgs', multi_class='auto'))
        ])
    elif model_type == "catboost":
        # Use 'MultiClass' loss function and pass the calculated class weights
        return CatBoostClassifier(random_state=random_state, loss_function='MultiClass', class_weights=class_weights, verbose=0)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(strategy, model_type, config):
    """Trains a single model for a given strategy and model type."""
    model_config = config['modeling']
    random_state = model_config['random_state']
    
    X_train, y_train = load_data(strategy, config)
    
    print(f"Training a {model_type} model for multi-class classification...")
    print(f"  Training data loaded. Number of features: {len(X_train.columns)}")

    # --- Calculate Class Weights ---
    # Compute weights to handle class imbalance
    unique_classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = dict(zip(unique_classes, weights))
    print(f"  Calculated class weights: {class_weights_dict}")
    # --- End of Class Weight Calculation ---

    model = get_model(model_type, random_state, class_weights=class_weights_dict)
    
    # Fit the model without a validation set or early stopping
    model.fit(X_train, y_train, verbose=False)
        
    print("Model training complete.")

    # Save the model
    model_filename = f'{strategy}_{model_type}_model.joblib'
    
    model_path = os.path.join('models', model_filename)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

def main():
    """Main function to run the model training pipeline."""
    print("--- Starting Model Training Pipeline ---")
    config = load_config('config/parameters.yml')
    
    # Get parameters from the unified config
    model_config = config.get('modeling', {})
    model_types = model_config.get('model_types', ['catboost'])
    strategies = model_config.get('strategies_to_train', ['combined'])
    
    for strategy in strategies:
        for model_type in model_types:
            print(f"\n--- Training {model_type.upper()} Model for {strategy.upper()} Strategy ---")
            train_model(strategy, model_type, config)

    print("\n--- Model Training Pipeline Finished ---")

if __name__ == "__main__":
    main()
