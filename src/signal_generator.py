import pandas as pd
import joblib
import os
import sys
from datetime import datetime

# Add the parent directory to the Python path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import load_config

log_file_path = "reports/trades/daily_trades.csv"

def convert_multiclass_signals_to_binary(source_signal_file: str, output_filename: str, thresholds: list):
    """
    Loads a multi-class signal file and converts it into one or more
    binary-compatible signal files based on probability thresholds.
    """
    print(f"  > Converting '{os.path.basename(source_signal_file)}' for backtest compatibility...")
    df = pd.read_csv(source_signal_file)

    # Extract base name for new files, e.g., 'combined_catboost'
    
    for threshold in thresholds:
        print(f"    - Applying threshold: {threshold}")
        compatible_df = df.copy()

        # 1. "prob_buy" column is the new "signal_prob" column
        if 'prob_buy' in compatible_df.columns:
            compatible_df.rename(columns={'prob_buy': 'signal_prob'}, inplace=True)
        else:
            print(f"    - WARNING: 'prob_buy' column not found in {source_signal_file}. Skipping.")
            continue
            
        # 2. Modify "signal" column based on threshold
        compatible_df['signal'] = (compatible_df['signal_prob'] >= threshold).astype(int)
        
        # 3. Drop remaining unused columns
        cols_to_drop = ['prob_sell', 'prob_hold']
        compatible_df.drop(columns=[col for col in cols_to_drop if col in compatible_df.columns], inplace=True, errors='ignore')

        # Generate a compatible filename, e.g., combined_up_catboost_thresh_0.70_signals.csv
        
        
        compatible_df.to_csv(output_filename, index=False)
        print(f"    - Saved compatible file to: {output_filename}")
        return compatible_df


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
    backtest_thresholds = config.get('trading', {}).get('backtest_thresholds', [0.70])


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

            compatible_filename = f'{strategy}_{model_type}_compatible.csv'
            compatible_filepath = os.path.join(signals_dir, compatible_filename)
            compatible_df = convert_multiclass_signals_to_binary(output_path, compatible_filepath, backtest_thresholds)

            data_end_date = data_config.get('test_end_date', datetime.now().strftime('%Y-%m-%d'))
            signal_df_for_gemini = compatible_df[compatible_df['timestamp'] == data_end_date]
            
            # We need to keep the new probability columns
            prob_cols_to_keep = [col for col in compatible_df.columns if 'signal_prob' in col]
            cols_to_keep = ['instrument_token', 'signal', 'target'] + prob_cols_to_keep
            
            signal_df_for_gemini = signal_df_for_gemini[signal_df_for_gemini['signal'] != 0][cols_to_keep]
            
            signal_df_for_gemini.to_csv(log_file_path, index=False)
        
            print(f"    - Saved signals to {output_path}")

    print("\n--- Signal Generation Pipeline Finished ---")

if __name__ == "__main__":
    generate_all_signals() 