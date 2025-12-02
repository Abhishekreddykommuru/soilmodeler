import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Configuration (Copied from view_results.py) ---
T_FILES = ['T1', 'T2', 'T3', 'T4', 'T5']
PREPROCESSORS = ['Reflectance', 'Absorbance']
MODELS = ['PLSR', 'GBRT', 'KRR', 'SVR', 'RandomForest']

# --- Helper Functions (Adapted from view_results.py) ---

def load_all_results():
    """Loads all pickle files and returns a list of dictionaries with metrics."""
    all_results = []
    
    # Use st.cache_data to speed up the app by caching this expensive operation
    # @st.cache_data 
    
    # Note: st.cache_data causes issues when files are externally modified, 
    # so we'll skip caching for now to ensure robustness during development.
    
    for t_file in T_FILES:
        for preproc_name in PREPROCESSORS:
            for model_name in MODELS:
                
                file_basename = f"spectra_with_target_{t_file}"
                pickle_filename = f"{file_basename}_target_{preproc_name}_{model_name}.pkl"
                
                try:
                    with open(pickle_filename, 'rb') as f:
                        results = pickle.load(f)
                        
                        all_results.append({
                            'File': t_file,
                            'Preproc': preproc_name,
                            'Model': model_name,
                            'R2': results['metrics']['r2'],
                            'RMSE': results['metrics']['rmse'],
                            'RPD': results['metrics']['rpd'],
                            'Filename': pickle_filename
                        })
                except FileNotFoundError:
                    # Skip missing files
                    pass
                except Exception:
                    # Skip files that failed to load
                    pass
                    
    return all_results

def load_single_result(filename):
    """Loads a single pickle file."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

def plot_scatter(y_true, y_pred, info):
    """Generates the Actual vs. Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 7))

    min_val = min(y_true.min(), y_pred.min()) * 0.95
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='w', linewidths=0.5, color='#4c72b0')
    
    # 1:1 line
    ax.plot([min_val, max_val], [min_val, max_val], 
             '--', color='red', linewidth=2, label='1:1 Line')
             
    # Best Fit Line
    m, b = np.polyfit(y_true, y_pred, 1)
    ax.plot(y_true, m*y_true + b, 
             '-', color='green', linewidth=1, label='Best Fit')

    ax.set_title(f"Predicted vs. Actual ({info['preprocessing']} + {info['model']})", fontsize=14)
    ax.set_xlabel(f"Actual {info['target_column']} Value", fontsize=12)
    ax.set_ylabel(f"Predicted {info['target_column']} Value", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    
    st.pyplot(fig)


def plot_residuals(y_true, y_pred, info):
    """Generates the Residual plot (Error vs. Actual)."""
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_true, residuals, alpha=0.7, edgecolors='k', color='#ff7f0e')
    
    # Horizontal line at 0
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5) 
    
    # Add a smoother to look for systematic bias (optional)
    # ax.plot(y_true, np.poly1d(np.polyfit(y_true, residuals, 1))(y_true), color='blue')

    ax.set_title("Residual Plot (Predicted Error)", fontsize=14)
    ax.set_xlabel(f"Actual {info['target_column']} Value", fontsize=12)
    ax.set_ylabel("Residuals (Predicted - Actual)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)


def plot_feature_influence(results):
    """Generates feature importance/coefficient plot and displays top 5."""
    model_name = results['info']['model']
    model = results['model']
    feature_names = np.array(results['validation_data']['feature_names'])
    wavelengths = np.array([float(f) for f in feature_names])

    st.subheader("Feature Influence Analysis")
    
    if model_name in ['RandomForest', 'GBRT']:
        # Feature Importance for Tree-based models
        try:
            importance = model.feature_importances_
            top_5_indices = np.argsort(importance)[::-1][:5]
            
            st.markdown(f"**Top 5 Most Important Wavelengths ({model_name}):**")
            top_features_list = [
                f"{feature_names[i]} nm (Score: {importance[i]:.4f})"
                for i in top_5_indices
            ]
            st.code('\n'.join(top_features_list))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(wavelengths, importance, color='darkorange', linewidth=2)
            ax.set_title(f"Feature Importance Plot ({model_name})")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Importance Score")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate feature importance plot for {model_name}.")

    elif model_name == 'PLSR':
        # Coefficients for PLSR
        try:
            coefficients = model.coef_.flatten()
            abs_coefficients = np.abs(coefficients)
            top_5_indices = np.argsort(abs_coefficients)[::-1][:5]

            st.markdown(f"**Top 5 Most Influential Wavelengths (PLSR Coefficients):**")
            top_features_list = [
                f"{feature_names[i]} nm (Coefficient: {coefficients[i]:.4f})"
                for i in top_5_indices
            ]
            st.code('\n'.join(top_features_list))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(wavelengths, coefficients, color='darkred', linewidth=2)
            ax.axhline(0, color='gray', linestyle='--')
            ax.set_title("PLSR Regression Coefficient Plot")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Regression Coefficient Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not generate PLSR coefficient plot.")
            
    else:
        st.info(f"Feature influence plotting is not directly supported for {model_name} (KRR/SVR).")


def display_summary_view():
    """Shows the table summarizing all model run metrics."""
    st.header("🔬 Comprehensive Model Performance Summary")
    
    all_results = load_all_results()

    if not all_results:
        st.warning("No pickle files found. Please run `python run_pipeline.py` first.")
        return
        
    df_results = pd.DataFrame(all_results)
    df_results = df_results.set_index('Filename')
    
    # User selection for sorting
    sort_options = ['RPD', 'R2', 'RMSE']
    selected_sort_metric = st.selectbox(
        "Sort results by:", 
        sort_options, 
        index=sort_options.index('RPD') # Default to RPD
    )
    
    reverse_sort = selected_sort_metric in ['R2', 'RPD']
    
    # Sort and reset index to display properly
    df_sorted = df_results.sort_values(
        by=selected_sort_metric, 
        ascending=not reverse_sort
    ).reset_index()

    # Format the metrics for display
    df_display = df_sorted.copy()
    for col in ['R2', 'RMSE', 'RPD']:
        df_display[col] = df_display[col].map(lambda x: f'{x:.4f}')

    st.dataframe(df_display, use_container_width=True)
    
    # Highlight Best Performer
    if not df_sorted.empty:
        best = df_sorted.iloc[0]
        st.success(f"🏆 **Best Performer:** {best['File']} / {best['Preproc']} / {best['Model']} ({selected_sort_metric}: {best[selected_sort_metric]:.4f})")


def display_detailed_view():
    """Shows the detailed dashboard for a single selected run."""
    st.header("📈 Detailed Model Diagnostic Dashboard")

    # --- Sidebar Selections ---
    st.sidebar.title("Select Model Run")
    
    selected_t = st.sidebar.selectbox("1. Data Source (T-File)", T_FILES)
    selected_preproc = st.sidebar.selectbox("2. Preprocessing Method", PREPROCESSORS)
    selected_model = st.sidebar.selectbox("3. Model Algorithm", MODELS)

    file_basename = f"spectra_with_target_{selected_t}"
    pickle_filename = f"{file_basename}_target_{selected_preproc}_{selected_model}.pkl"
    
    st.sidebar.markdown(f"**Loading:** `{pickle_filename}`")
    
    results = load_single_result(pickle_filename)

    if results is None:
        st.warning(f"Could not load the file: `{pickle_filename}`. Please ensure it exists.")
        return

    # Extract Data
    metrics = results['metrics']
    info = results['info']
    y_true = np.array(results['validation_data']['y_true'])
    y_pred = np.array(results['validation_data']['y_pred'])

    # --- Metrics Section ---
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared ($R^2$)", f"{metrics['r2']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.4f}", help="Root Mean Square Error (lower is better)")
    col3.metric("RPD", f"{metrics['rpd']:.4f}", help="Ratio of Performance to Deviation (higher is better)")

    # --- Plots Section ---
    
    st.markdown("---")
    
    st.subheader("Visual Diagnostics")
    
    # Scatter Plot and Residual Plot side-by-side
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        st.caption("Predicted vs. Actual Scatter Plot (Checks overall fit)")
        plot_scatter(y_true, y_pred, info)
        
    with plot_col2:
        st.caption("Residual Plot (Checks for systematic bias)")
        plot_residuals(y_true, y_pred, info)

    st.markdown("---")
    
    # Feature Influence Plot
    plot_feature_influence(results)


# --- Main Streamlit App Structure ---
def main():
    st.set_page_config(layout="wide", page_title="Spectral ML Results Viewer")
    st.title("Spectral Model Analysis Dashboard")
    st.markdown("A tool to compare and diagnose the results from the automated ML pipeline.")

    # Select Mode: Summary or Detailed
    mode = st.radio(
        "Choose Analysis Mode:",
        ('Summary Comparison', 'Detailed Run Analysis'),
        horizontal=True
    )
    
    st.markdown("---")

    if mode == 'Summary Comparison':
        display_summary_view()
    elif mode == 'Detailed Run Analysis':
        display_detailed_view()


if __name__ == '__main__':
    main()