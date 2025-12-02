import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration for User Selection ---
T_FILES = ['T1', 'T2', 'T3', 'T4', 'T5']
PREPROCESSORS = ['Reflectance', 'Absorbance']
MODELS = ['PLSR', 'GBRT', 'KRR', 'SVR', 'RandomForest']

def get_user_choice(prompt, options):
    """Prompts the user to select from a list of options."""
    print(f"\nSelect the {prompt}:")
    # Handle both list of strings and dictionary keys
    option_names = list(options) if isinstance(options, dict) else options
    for i, option in enumerate(option_names):
        print(f"  [{i+1}] {option}")
    
    while True:
        try:
            choice = input(f"Enter number (1-{len(option_names)}): ")
            index = int(choice) - 1
            if 0 <= index < len(option_names):
                return option_names[index]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def plot_residuals(y_true, y_pred, info):
    """Plots the residuals (Prediction Error) vs. the Actual Values."""
    
    residuals = y_pred - y_true
    
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, residuals, alpha=0.7, edgecolors='k', color='#ff7f0e')
    
    # Horizontal line at 0 to indicate perfect prediction
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5) 

    plt.title(f"Residual Plot: {info['file'].replace('.xls', '')} ({info['preprocessing']} + {info['model']})", fontsize=14)
    plt.xlabel(f"Actual {info['target_column']} Value", fontsize=12)
    plt.ylabel("Residuals (Predicted - Actual)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    print("\nDisplaying residual plot (Error vs. Actual)...")
    plt.show()

def plot_feature_importance(model, feature_names, info):
    """Plots feature importance for tree-based models (GBRT, RandomForest) and lists the top 5 features."""
    
    if info['model'] not in ['RandomForest', 'GBRT']:
        return

    try:
        importance = model.feature_importances_
        wavelengths = np.array([float(f) for f in feature_names])
        
        # --- Print Top 5 Features ---
        top_5_indices = np.argsort(importance)[::-1][:5]
        
        print(f"\n--- Top 5 Most Important Wavelengths ({info['model']}) ---")
        for i in top_5_indices:
            print(f"  {feature_names[i]} nm: {importance[i]:.4f}")

        # --- Plot Feature Importance ---
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, importance, color='darkorange', linewidth=2)
        plt.title(f"Feature Importance Plot ({info['model']})", fontsize=14)
        plt.xlabel("Wavelength (nm)", fontsize=12)
        plt.ylabel("Feature Importance Score", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        print(f"\nDisplaying feature importance plot for {info['model']}...")
        plt.show()

    except Exception as e:
        print(f"\nWarning: Could not plot feature importance for {info['model']}: {e}")
        pass

def plot_plsr_coefficients(model, feature_names, info):
    """Plots and lists the regression coefficients for PLSR models."""
    
    if info['model'] != 'PLSR':
        return

    try:
        coefficients = model.coef_.flatten()
        abs_coefficients = np.abs(coefficients)
        wavelengths = np.array([float(f) for f in feature_names])

        # --- Print Top 5 Features ---
        top_5_indices = np.argsort(abs_coefficients)[::-1][:5]
        
        print(f"\n--- Top 5 Most Influential Wavelengths ({info['model']} Coefficients) ---")
        for i in top_5_indices:
            print(f"  {feature_names[i]} nm: {coefficients[i]:.4f}")

        # --- Plot Coefficients ---
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, coefficients, color='darkred', linewidth=2)
        plt.axhline(0, color='gray', linestyle='--') # Add a zero line
        plt.title(f"PLSR Regression Coefficient Plot", fontsize=14)
        plt.xlabel("Wavelength (nm)", fontsize=12)
        plt.ylabel("Regression Coefficient Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        print(f"\nDisplaying PLSR coefficient plot...")
        plt.show()

    except Exception as e:
        print(f"\nWarning: Could not plot PLSR coefficients: {e}")
        pass


def summarize_all_results():
    """Loads all pickle files and prints a summary table."""
    all_results = []
    
    print("\nLoading metrics for all 50 possible runs...")
    
    # Iterate through all combinations
    for t_file in T_FILES:
        for preproc_name in PREPROCESSORS:
            for model_name in MODELS:
                
                file_basename = f"spectra_with_target_{t_file}"
                pickle_filename = f"{file_basename}_target_{preproc_name}_{model_name}.pkl"
                
                try:
                    with open(pickle_filename, 'rb') as f:
                        results = pickle.load(f)
                        
                        # Store essential metadata and metrics
                        all_results.append({
                            'File': t_file,
                            'Preproc': preproc_name,
                            'Model': model_name,
                            'R2': results['metrics']['r2'],
                            'RMSE': results['metrics']['rmse'],
                            'RPD': results['metrics']['rpd'],
                        })
                except FileNotFoundError:
                    # Silently skip missing files (if run_pipeline failed for some)
                    pass
                except Exception as e:
                    # Log other errors
                    print(f"Warning: Failed to load {pickle_filename}: {e}")

    if not all_results:
        print("\n!!! No result files found. Please run 'run_pipeline.py' first. !!!")
        return

    # --- NEW: Get User Sorting Preference ---
    sort_options = ['R2', 'RMSE', 'RPD']
    selected_sort_metric = get_user_choice("Sort the summary table by which metric", sort_options)

    # Determine sorting direction: Higher is better for R2 and RPD, lower is better for RMSE.
    reverse_sort = selected_sort_metric in ['R2', 'RPD']

    # Sort results based on the chosen metric
    all_results.sort(key=lambda x: x[selected_sort_metric], reverse=reverse_sort)

    # --- Display Summary Table ---
    
    sort_direction = "Highest is Best" if reverse_sort else "Lowest is Best"
    
    print("\n==========================================================================")
    print(f"                 Overall Model Performance Summary (Sorted by {selected_sort_metric})")
    print(f"                 Sorting Direction: ({sort_direction})")
    print("==========================================================================")
    
    # Print Header
    print(f"{'File':<5} | {'Preproc':<12} | {'Model':<12} | {'R2':>8} | {'RMSE':>8} | {'RPD':>8}")
    print("-" * 65)

    # Print Data
    for r in all_results:
        print(f"{r['File']:<5} | {r['Preproc']:<12} | {r['Model']:<12} | {r['R2']:>8.4f} | {r['RMSE']:>8.4f} | {r['RPD']:>8.4f}")
    
    # Highlight Best
    best = all_results[0]
    print("\n--------------------------------------------------------------------------")
    print(f"BEST PERFORMER: {best['File']} / {best['Preproc']} / {best['Model']} ({selected_sort_metric}: {best[selected_sort_metric]:.4f})")
    print("--------------------------------------------------------------------------")


def load_and_view_results():
    """Guides the user to select a single pickle file and plots all details."""
    
    # 1. Get User Selections
    selected_t = get_user_choice("T-File", T_FILES)
    selected_preproc = get_user_choice("Preprocessing Method", PREPROCESSORS)
    selected_model = get_user_choice("Model", MODELS)
    
    # 2. Construct the filename
    file_basename = f"spectra_with_target_{selected_t}"
    base_filename = f"{file_basename}_target_{selected_preproc}_{selected_model}.pkl"
    
    print(f"\nAttempting to load: {base_filename}")
    
    # 3. Load the pickle file
    try:
        with open(base_filename, 'rb') as f:
            results = pickle.load(f)
            print("Successfully loaded model results.")
    except FileNotFoundError:
        print(f"\n!!! ERROR: File not found !!!")
        print(f"The file '{base_filename}' does not exist.")
        print("Please ensure you have run 'run_pipeline.py' successfully.")
        return
    except Exception as e:
        print(f"\n!!! ERROR: Failed to load file !!!")
        print(f"Details: {e}")
        return

    # 4. Display Metrics
    metrics = results['metrics']
    info = results['info']
    
    print("\n--- Summary of Results ---")
    print(f"Data Source: {info['file'].replace('.xls', '')}")
    print(f"Target Property: {info['target_column']}")
    print(f"Preprocessing: {info['preprocessing']}")
    print(f"Model Algorithm: {info['model']}")
    print("-" * 30)
    print(f"R-squared ($R^2$): {metrics['r2']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Ratio of Performance to Deviation (RPD): {metrics['rpd']:.4f}")
    print("-" * 30)
    
    # 5. Prepare Data for Plotting and Feature Analysis
    y_true = np.array(results['validation_data']['y_true'])
    y_pred = np.array(results['validation_data']['y_pred'])
    feature_names = results['validation_data']['feature_names']
    
    # 6. Create the Scatter Plot (Actual vs. Predicted)
    
    # Find global min/max for the 1:1 line
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot of predicted vs actual values
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='w', linewidths=0.5, color='#4c72b0')
    
    # 1:1 line (perfect prediction line)
    plt.plot([min_val, max_val], [min_val, max_val], 
             '--', color='red', linewidth=2, label='1:1 Line')
             
    # Best Fit Line (optional, for comparison)
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + b, 
             '-', color='green', linewidth=1, label='Best Fit')

    plt.title(f"{file_basename} ({info['preprocessing']} + {info['model']})", fontsize=14)
    plt.xlabel(f"Actual {info['target_column']} Value", fontsize=12)
    plt.ylabel(f"Predicted {info['target_column']} Value", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # Ensures the 1:1 line looks correct
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add text box for key metrics
    textstr = '\n'.join((
        r'$R^2=%.4f$' % (metrics['r2'], ),
        r'$RMSE=%.4f$' % (metrics['rmse'], ),
        r'$RPD=%.4f$' % (metrics['rpd'], )))
    
    # Place the text box in the bottom right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    print("\nDisplaying scatter plot (Predicted vs. Actual)...")
    plt.show()
    
    # 7. Create Diagnostic Plots
    
    # Residual Plot (New)
    plot_residuals(y_true, y_pred, info)
    
    # Feature Analysis Plots
    plot_feature_importance(results['model'], feature_names, info)
    plot_plsr_coefficients(results['model'], feature_names, info)


def main():
    print("\n==========================================")
    print("       ML Model Performance Viewer        ")
    print("==========================================")
    
    choices = {
        'View detailed results for a single run': load_and_view_results,
        'Summarize and compare metrics for ALL runs': summarize_all_results
    }
    
    selected_mode = get_user_choice("Operation Mode", choices)
    
    print(f"\nMode selected: {selected_mode}")
    
    if selected_mode == 'View detailed results for a single run':
        load_and_view_results()
    elif selected_mode == 'Summarize and compare metrics for ALL runs':
        summarize_all_results()

if __name__ == '__main__':
    main()