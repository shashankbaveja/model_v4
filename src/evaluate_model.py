import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import glob
from sklearn.pipeline import Pipeline
# Add the root directory for local library import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_pipeline import load_config
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, X_test, y_test):
    """
    Evaluates a single trained multi-class model and returns the classification
    report string and the confusion matrix.
    """
    model = joblib.load(model_path)
    
    # Predict the class directly
    y_pred = model.predict(X_test)
    
    # Generate a full classification report as a string
    report_str = classification_report(
        y_test, y_pred, 
        target_names=['SELL (-1)', 'HOLD (0)', 'BUY (1)'], 
        zero_division=0
    )
    
    # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    
    return report_str, cm

def main():
    """
    Main function to evaluate all trained models based on the current config
    and generate a consolidated report.
    """
    print("--- Starting Model Evaluation ---")
    config = load_config('config/parameters.yml')
    model_config = config.get('modeling', {})
    strategies = model_config.get('strategies_to_train', ['combined'])
    model_types = model_config.get('model_types', ['catboost'])
    processed_dir = 'data/processed'

    # --- Load All Test Data ---
    print(f"Loading test datasets from '{processed_dir}'")
    test_data = {}
    for s in strategies:
        try:
            path = os.path.join(processed_dir, f'test_{s}_with_patterns_features.parquet')
            test_data[s] = pd.read_parquet(path)
            
        except FileNotFoundError:
            print(f"Warning: Test data for strategy '{s}' not found at {path}. Skipping.")

    if not test_data:
        print("Error: No test data could be loaded. Exiting.")
        sys.exit(1)

    # --- Evaluate All Models based on config ---
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear previous reports
    if os.path.exists(os.path.join(output_dir, 'classification_report.txt')):
        os.remove(os.path.join(output_dir, 'classification_report.txt'))

    for strategy_name in strategies:
        for model_type in model_types:
            model_filename = f'{strategy_name}_{model_type}_model.joblib'
            model_path = os.path.join('models', model_filename)

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}. Skipping.")
                continue
                
            print(f"\n--- Evaluating model: {model_filename} ---")

            if strategy_name not in test_data:
                print(f"Warning: No test data found for strategy '{strategy_name}'. Skipping.")
                continue
                
            test_df = test_data[strategy_name].copy()
            
            if 'target' not in test_df.columns:
                print(f"Warning: Target column 'target' not found for {model_filename}. Skipping.")
                continue
                    
            y_test = test_df.pop('target')
            X_test = test_df.drop(columns=['instrument_token', 'timestamp'], errors='ignore')

            report_str, cm = evaluate_model(model_path, X_test, y_test)
            
            # --- Display and Save Report ---
            print("Classification Report:")
            print(report_str)
            
            # Save the text report
            report_path = os.path.join(output_dir, 'classification_report.txt')
            with open(report_path, 'a') as f:
                f.write(f"--- Report for {model_filename} ---\n")
                f.write(report_str)
                f.write("\n\n")
            print(f"Full report appended to {report_path}")

            # --- Display and Save Confusion Matrix ---
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['SELL (-1)', 'HOLD (0)', 'BUY (1)'],
                        yticklabels=['SELL (-1)', 'HOLD (0)', 'BUY (1)'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for {model_filename}')
            
            cm_path = os.path.join(output_dir, f'confusion_matrix_{strategy_name}_{model_type}.png')
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            plt.close() # Close the plot to avoid displaying it in some environments

    print("\n--- Model Evaluation Finished ---")

if __name__ == "__main__":
    main()
