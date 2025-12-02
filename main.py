import pandas as pd
import numpy as np
import glob
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Model Imports ---
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

print("Script started. Finding files and defining models...")

# --- Configuration ---

# 1. Find all files to process
files_to_process = glob.glob('spectra_with_target_T*.xls')
if not files_to_process:
    print("Warning: No 'spectra_with_target_T*.xls' files found.")
    print("Make sure this script is in the same folder as your files.")

# 2. Define the target variables to look for in the files
POTENTIAL_TARGETS = ['Clay', 'Sand', 'TOC']

# 3. Define the models
# NOTE: Cubist is not in scikit-learn. Replaced with RandomForestRegressor.
MODELS = {
    'PLSR': PLSRegression(n_components=10), # PLSR needs n_components
    'GBRT': GradientBoostingRegressor(random_state=42),
    'KRR': KernelRidge(kernel='rbf'),
    'SVR': SVR(kernel='rbf'),
    'RandomForest': RandomForestRegressor(random_state=42) # Replaced Cubist
}

# 4. Define the preprocessing functions
def preprocess_reflectance(X):
    """Uses raw reflectance data."""
    print("      Preprocessing: Reflectance (Raw Data)")
    return X

def preprocess_absorbance(X):
    """Calculates Absorbance = log(1/R)."""
    print("      Preprocessing: Absorbance (log(1/R))")
    # Add a small epsilon to avoid log(0)
    X_safe = X + 1e-9 
    # Clip values at 0, as reflectance can't be negative
    X_safe[X_safe <= 0] = 1e-9
    return np.log(1 / X_safe)

PREPROCESSING = {
    'Reflectance': preprocess_reflectance,
    'Absorbance': preprocess_absorbance
}
# NOTE: Continuum Removal is not included as it requires
# specialized non-sklearn libraries (e.g., pysptools).

# --- Main Processing Loop ---

results_log = []

for file_path in files_to_process:
    print(f"\n--- Processing File: {file_path} ---")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}. Skipping.")
        continue
    
    # Identify which targets are in this file
    found_targets = [col for col in POTENTIAL_TARGETS if col in df.columns]
    if not found_targets:
        print(f"  No potential targets {POTENTIAL_TARGETS} found in {file_path}. Skipping.")
        continue

    # Identify features (all numeric columns MINUS the targets)
    all_numeric_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in all_numeric_cols if col not in found_targets]
    
    if not feature_cols:
        print(f"  No feature columns found in {file_path}. Skipping.")
        continue

    X_raw = df[feature_cols]

    # --- Loop 1: Targets ---
    for target_name in found_targets:
        print(f"  Target: {target_name}")
        y = df[target_name]
        
        # Remove rows with NaN in this target or in features
        valid_indices = y.notna() & X_raw.notna().all(axis=1)
        if not valid_indices.any():
            print(f"    No valid data for target {target_name}. Skipping.")
            continue
            
        X_clean = X_raw[valid_indices]
        y_clean = y[valid_indices]
        
        # --- Loop 2: Preprocessing ---
        for preproc_name, preproc_func in PREPROCESSING.items():
            
            # Apply preprocessing
            try:
                X_processed = preproc_func(X_clean)
                
                # Scale data *after* preprocessing
                # This is crucial for models like SVR and KRR
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_processed)
                
            except Exception as e:
                print(f"    Error during preprocessing {preproc_name}: {e}. Skipping.")
                continue

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clean, test_size=0.2, random_state=42
            )

            # --- Loop 3: Models ---
            for model_name, model_class in MODELS.items():
                print(f"        Model: {model_name}")
                
                try:
                    # Initialize a fresh model instance
                    model = model_class
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Predict and Validate
                    y_pred = model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    rpd = np.std(y_test) / rmse
                    
                    # Prepare results dictionary
                    results = {
                        'model': model,
                        'scaler': scaler, # Save the scaler! Need it for new data.
                        'metrics': {
                            'r2': r2,
                            'rmse': rmse,
                            'rpd': rpd
                        },
                        'validation_data': {
                            'y_true': y_test,
                            'y_pred': y_pred
                        },
                        'info': {
                            'file': file_path,
                            'target': target_name,
                            'preprocessing': preproc_name,
                            'model': model_name
                        }
                    }
                    
                    # Generate filename and save pickle
                    file_basename = os.path.basename(file_path).replace('.xls', '')
                    pickle_filename = f"{file_basename}_{target_name}_{preproc_name}_{model_name}.pkl"
                    
                    with open(pickle_filename, 'wb') as f:
                        pickle.dump(results, f)
                        
                    log_entry = f"Saved: {pickle_filename} (R2: {r2:.3f}, RMSE: {rmse:.3f})"
                    print(f"          {log_entry}")
                    results_log.append(log_entry)

                except Exception as e:
                    print(f"        FAILED for {model_name}: {e}")

print("\n--- Script Finished ---")
print(f"Successfully generated {len(results_log)} pickle files.")