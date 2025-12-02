import pandas as pd
import numpy as np
import glob
import pickle
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Model Imports ---
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

print("Script started. Defining models and configurations...")

# --- Configuration ---

# 1. File identification pattern
files_to_process = glob.glob('spectra_with_target_T*.xls') # Still use .xls pattern for glob

# 2. The target column name identified from the inspection
TARGET_COLUMN = 'target'

# 3. Define the models (RandomForest replaces Cubist)
MODELS = {
    'PLSR': PLSRegression(n_components=10),
    'GBRT': GradientBoostingRegressor(random_state=42),
    'KRR': KernelRidge(kernel='rbf'),
    'SVR': SVR(kernel='rbf'),
    'RandomForest': RandomForestRegressor(random_state=42)
}

# 4. Define the preprocessing functions
def preprocess_reflectance(X):
    """Uses raw reflectance data (no change)."""
    return X.copy()

def preprocess_absorbance(X):
    """Calculates Absorbance = log(1/R)."""
    # Ensure all values are slightly positive before log transformation
    X_safe = X.clip(lower=1e-9) 
    return np.log(1 / X_safe)

# Continuum Removal is omitted as it requires specialized libraries outside of core stack
PREPROCESSING = {
    'Reflectance': preprocess_reflectance,
    'Absorbance': preprocess_absorbance
}

# --- Main Processing Loop ---

results_log = []

for file_path in files_to_process:
    print(f"\n--- Processing File: {file_path} ---")
    try:
        # --- FIX: Read as CSV with comma delimiter ---
        # We read the raw file content and then use StringIO to force CSV reading 
        # as the filename extension is misleading.
        with open(file_path, 'rb') as f:
            content = f.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=',')
        
    except Exception as e:
        print(f"  Error reading {file_path} as CSV: {e}. Skipping.")
        continue
    
    # Check if the target column exists
    if TARGET_COLUMN not in df.columns:
        print(f"  Required target column '{TARGET_COLUMN}' not found in {file_path}. Skipping.")
        continue

    # Identify features (all columns EXCEPT the target)
    y = df[TARGET_COLUMN]
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    X_raw = df[feature_cols]

    # Handle NaNs and non-numeric data
    valid_indices = y.notna() & X_raw.apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
    
    X_clean = X_raw.loc[valid_indices].apply(pd.to_numeric, errors='coerce')
    y_clean = y.loc[valid_indices]

    if X_clean.empty or y_clean.empty:
        print(f"  No valid data points after cleaning for {file_path}. Skipping.")
        continue

    # --- Loop 1: Preprocessing ---
    for preproc_name, preproc_func in PREPROCESSING.items():
        print(f"  Preprocessing: {preproc_name}")
        
        # Apply preprocessing
        try:
            X_processed = preproc_func(X_clean)
            
            # 1. Scale data *after* preprocessing
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            
            # 2. Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clean, test_size=0.2, random_state=42
            )
            
        except Exception as e:
            print(f"    Error during preprocessing or splitting for {preproc_name}: {e}. Skipping pipeline.")
            continue

        # --- Loop 2: Models ---
        for model_name, model_class in MODELS.items():
            
            try:
                # Initialize a fresh model instance
                model = model_class
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict and Validate
                y_pred = model.predict(X_test)
                
                # Calculate Metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # RPD (Ratio of Performance to Deviation) = SD of actuals / RMSE
                std_y_test = np.std(y_test)
                rpd = std_y_test / rmse if rmse != 0 else np.nan 

                # Prepare results dictionary
                results = {
                    'model': model,
                    'scaler': scaler, 
                    'metrics': {
                        'r2': r2,
                        'rmse': rmse,
                        'rpd': rpd
                    },
                    'validation_data': {
                        'y_true': y_test.tolist(),
                        'y_pred': y_pred.tolist(),
                        'feature_names': X_clean.columns.tolist()
                    },
                    'info': {
                        'file': file_path,
                        'target_column': TARGET_COLUMN,
                        'preprocessing': preproc_name,
                        'model': model_name
                    }
                }
                
                # Generate filename and save pickle
                file_basename = os.path.basename(file_path).replace('.xls', '')
                pickle_filename = f"{file_basename}_target_{preproc_name}_{model_name}.pkl"
                
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(results, f)
                    
                log_entry = f"Saved: {pickle_filename} (R2: {r2:.3f}, RMSE: {rmse:.3f}, RPD: {rpd:.3f})"
                print(f"    -> {log_entry}")
                results_log.append(log_entry)

            except Exception as e:
                print(f"    -> FAILED for {model_name}: {e}")

print("\n--- Script Finished ---")
print(f"Successfully generated {len(results_log)} pickle files.")