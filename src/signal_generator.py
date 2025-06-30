import pandas as pd
import joblib
import os
import sys
from datetime import datetime

# Add the parent directory to the Python path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config

log_file_path = "reports/trades/daily_trades.csv"

def generate_all_signals():
    """
    Loads all trained multi-class models, generates trade signals based on model predictions,
    and saves them as new datasets containing OHLCV and signal data.
    """
    print("--- Starting Signal Generation Pipeline ---")
    config = load_config()

    # Get parameters from config
    model_config = config.get('modeling', {})
    data_config = config.get('data', {})
    strategies = model_config.get('strategies_to_train', ['combined'])
    model_types = model_config.get('model_types', ['catboost'])

    # --- Setup Directories ---
    processed_dir = 'data/processed'
    signals_dir = 'data/signals'
    models_dir = 'models'
    os.makedirs(signals_dir, exist_ok=True)

    # --- Load Raw Price Data ---
    # This contains the OHLCV data we need to join with the signals.
    try:
        # Assuming test_raw is still the source of truth for prices
        raw_test_data = pd.read_parquet(os.path.join(processed_dir, 'test_raw.parquet'))
        print("Loaded raw test data for OHLCV information.")
    except FileNotFoundError as e:
        print(f"Error: Raw test data not found. It should be in 'data/processed/'. Details: {e}")
        sys.exit(1)

    # --- Main Loop for Signal Generation ---
    for strategy in strategies:
        print(f"\nProcessing strategy: {strategy.upper()}")

        # Load the corresponding feature data for the strategy
        feature_data_path = os.path.join(processed_dir, f'test_{strategy}_with_patterns_features.parquet')
        try:
            feature_df = pd.read_parquet(feature_data_path)
            print(f"  Loaded feature data from {feature_data_path}")
        except FileNotFoundError:
            print(f"Warning: Feature data not found for strategy '{strategy}'. Skipping.")
            continue
        
        # Columns to keep for the final output, but not for prediction
        cols_to_keep_for_output = ['instrument_token', 'timestamp', 'target']

        # Prepare feature matrix X_test by dropping non-feature columns
        X_test = feature_df.drop(columns=cols_to_keep_for_output, errors='ignore')

        for model_type in model_types:
            model_name = f'{strategy}_{model_type}_model.joblib'
            model_path = os.path.join(models_dir, model_name)

            if not os.path.exists(model_path):
                print(f"  - Model not found: {model_path}. Skipping.")
                continue

            # Load the model and predict classes and probabilities
            print(f"  - Generating signals for: {model_name}")
            model = joblib.load(model_path)
            
            # Predict the class directly (-1, 0, 1)
            signals = model.predict(X_test)
            
            # Get the probability for ALL classes
            y_pred_proba = model.predict_proba(X_test)
            
            # Create a temporary DataFrame with identifiers and the signal
            temp_df = feature_df[cols_to_keep_for_output].copy()
            temp_df['signal'] = signals
            
            # --- Assign probabilities for each class to new columns ---
            # Assuming the model's classes_ are ordered [-1, 0, 1]
            # It's safer to get the order directly from the trained model
            class_order = model.classes_
            prob_map = {
                -1: 'prob_sell',
                0: 'prob_hold',
                1: 'prob_buy'
            }
            
            for i, class_label in enumerate(class_order):
                col_name = prob_map.get(class_label, f'prob_class_{class_label}')
                temp_df[col_name] = y_pred_proba[:, i]
            # --- End of probability assignment ---

            # Merge with raw data to get OHLCV columns
            final_signal_df = pd.merge(raw_test_data, temp_df, on=['instrument_token', 'timestamp'])

            # --- Print signals generated for today ---
            todays_signals = final_signal_df[
                (pd.to_datetime(final_signal_df['timestamp']).dt.date == pd.to_datetime('today').date()) &
                (final_signal_df['signal'] != 0) # Show both BUY (1) and SELL (-1) signals
            ]
            if not todays_signals.empty:
                print(f"    >>> Today's Signals ({model_name}):")
                print(todays_signals[['instrument_token', 'close', 'signal', 'target', 'prob_buy', 'prob_sell', 'prob_hold']])
                print("-" * 40)
            else:
                print(f"    >>> No BUY/SELL signals generated for today ({model_name})")
            # --- End of section ---

            # Save the final signal file
            output_filename = f'{strategy}_{model_type}_signals.csv'
            output_path = os.path.join(signals_dir, output_filename)
            final_signal_df.to_csv(output_path, index=False)

            data_end_date = data_config.get('test_end_date', datetime.now().strftime('%Y-%m-%d'))
            signal_df_for_gemini = final_signal_df[final_signal_df['timestamp'] == data_end_date]
            
            # We need to keep the new probability columns
            prob_cols_to_keep = [col for col in final_signal_df.columns if 'prob_' in col]
            cols_to_keep = ['instrument_token', 'signal', 'target'] + prob_cols_to_keep
            
            signal_df_for_gemini = signal_df_for_gemini[signal_df_for_gemini['signal'] != 0][cols_to_keep]
            
            signal_df_for_gemini.to_csv(log_file_path, index=False)
        
            print(f"    - Saved signals to {output_path}")

    print("\n--- Signal Generation Pipeline Finished ---")

if __name__ == "__main__":
    generate_all_signals() 