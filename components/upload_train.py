"""
Upload & Train Component
========================
Page 2: Data upload and model training workflow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from modules.data_loader import DataLoader
from modules.preprocessing import SpectralPreprocessor
from modules.models import ModelFactory
from modules.training import TrainingOrchestrator
from modules.visualization import Visualizer
from modules.utils import (
    render_status_box, get_performance_color, format_time,
    save_results_to_session, get_medal
)


def render_upload_train():
    """Render the upload and train page."""
    
    st.markdown("""
    <h1 style="background: linear-gradient(90deg, #667eea, #764ba2); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📤 Upload & Train Models
    </h1>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs(["📂 Upload Data", "⚙️ Configuration", "🚀 Training", "🎮 Playground"])
    
    # Initialize objects
    data_loader = DataLoader()
    preprocessor = SpectralPreprocessor()
    model_factory = ModelFactory()
    visualizer = Visualizer()
    
    # Tab 1: Upload Data
    with tabs[0]:
        render_upload_tab(data_loader, visualizer)
    
    # Tab 2: Configuration
    with tabs[1]:
        render_config_tab(preprocessor, model_factory)
    
    # Tab 3: Training
    with tabs[2]:
        render_training_tab()
    
    # Tab 4: Playground (Expert Mode)
    with tabs[3]:
        render_playground_tab(model_factory)


def render_upload_tab(data_loader: DataLoader, visualizer: Visualizer):
    """Render the data upload tab."""
    
    st.markdown("### Step 1: Upload Your Spectral Dataset")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=['csv', 'xls', 'xlsx'],
        help="Supported formats: CSV, XLS, XLSX. Max size: 500 MB"
    )
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading file..."):
            df, message = data_loader.load_file(uploaded_file)
        
        if df is not None:
            st.session_state.data = df
            st.session_state.metadata = data_loader.metadata
            
            st.success(message)
            
            # Dataset Overview
            st.markdown("### Step 2: Dataset Overview")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            metadata = data_loader.metadata
            
            with col1:
                st.metric("Samples", metadata.get('n_samples', 0))
            with col2:
                st.metric("Columns", metadata.get('n_columns', 0))
            with col3:
                missing = metadata.get('missing_values', 0)
                st.metric("Missing Values", missing, delta="✅" if missing == 0 else "⚠️")
            with col4:
                quality = metadata.get('quality_label', 'Unknown')
                st.metric("Quality", quality)
            
            # Data quality indicators
            st.markdown("#### Data Quality Indicators")
            
            if metadata.get('missing_values', 0) == 0:
                render_status_box("No missing values detected", "success")
            else:
                render_status_box(f"Found {metadata.get('missing_values', 0)} missing values", "warning")
            
            if metadata.get('duplicates', 0) == 0:
                render_status_box("No duplicate rows detected", "success")
            else:
                render_status_box(f"Found {metadata.get('duplicates', 0)} duplicate rows", "warning")
            
            if metadata.get('spectral_range') != "Not detected":
                render_status_box(f"Spectral range: {metadata.get('spectral_range', 'N/A')}", "info")
            
            # Data preview
            with st.expander("📊 View First 10 Rows", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Target Selection
            st.markdown("### Step 3: Select Target Property")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Target column selection
                target_col = st.selectbox(
                    "Target Column",
                    options=df.columns.tolist(),
                    index=len(df.columns) - 1,  # Default to last column
                    help="Select the soil property to predict"
                )
                st.session_state.target_col = target_col
            
            with col2:
                # Category selection
                category = st.selectbox(
                    "Category",
                    options=["T1 (Nitrogen)", "T2 (Phosphorus)", "T3 (Potassium)", 
                             "T4 (Organic Carbon)", "T5 (pH)", "Custom"],
                    help="Select the soil property category"
                )
            
            # Target statistics
            if target_col:
                st.markdown("#### Target Statistics")
                target_stats = data_loader.get_target_statistics(df, target_col)
                
                if target_stats:
                    stat_cols = st.columns(6)
                    
                    with stat_cols[0]:
                        st.metric("Mean", f"{target_stats['mean']:.3f}")
                    with stat_cols[1]:
                        st.metric("Std", f"{target_stats['std']:.3f}")
                    with stat_cols[2]:
                        st.metric("Min", f"{target_stats['min']:.3f}")
                    with stat_cols[3]:
                        st.metric("Max", f"{target_stats['max']:.3f}")
                    with stat_cols[4]:
                        st.metric("Range", f"{target_stats['range']:.3f}")
                    with stat_cols[5]:
                        st.metric("CV (%)", f"{target_stats['cv']:.1f}")
                    
                    # Target distribution plot
                    with st.expander("📈 Target Distribution", expanded=True):
                        fig = visualizer.target_distribution(df[target_col].values, "Target Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Outlier info
                    if target_stats.get('n_outliers', 0) > 0:
                        st.warning(f"⚠️ Found {target_stats['n_outliers']} potential outliers ({target_stats['outlier_percentage']:.1f}%)")
                        
                        outlier_action = st.radio(
                            "Outlier handling:",
                            ["Keep all samples", "Remove outliers before training", "Winsorize outliers"],
                            horizontal=True
                        )
                        st.session_state.outlier_action = outlier_action
            
            # Validation
            if st.button("✅ Validate Data for Training", use_container_width=True):
                is_valid, validation_msg = data_loader.validate_for_training(df, target_col)
                if is_valid:
                    st.success(validation_msg)
                    st.info("✅ Data is ready for training. Proceed to the Configuration tab.")
                else:
                    st.error(validation_msg)
        
        else:
            st.error(message)
    
    else:
        if st.session_state.get('data') is not None:
            st.info(f"📊 Currently loaded: {st.session_state.metadata.get('filename', 'Unknown')}")


def render_config_tab(preprocessor: SpectralPreprocessor, model_factory: ModelFactory):
    """Render the configuration tab."""
    
    st.markdown("### Step 4: Training Configuration")
    
    if st.session_state.get('data') is None:
        st.warning("⚠️ Please upload data first in the Upload Data tab.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ Preprocessing Methods")
        
        # Preprocessing selection
        all_prep = list(SpectralPreprocessor.METHODS.keys())
        selected_prep = st.multiselect(
            "Select preprocessing transformations:",
            options=all_prep,
            default=all_prep,
            format_func=lambda x: SpectralPreprocessor.METHODS.get(x, x)
        )
        st.session_state.preprocessing_methods = selected_prep
        
        # Info about preprocessing
        for prep in selected_prep:
            with st.expander(f"ℹ️ {SpectralPreprocessor.METHODS.get(prep, prep)}", expanded=False):
                st.markdown(preprocessor.get_method_description(prep))
    
    with col2:
        st.markdown("#### 🤖 ML Algorithms")
        
        # Algorithm selection
        all_models = list(ModelFactory.MODELS.keys())
        selected_models = st.multiselect(
            "Select machine learning models:",
            options=all_models,
            default=all_models,
            format_func=lambda x: f"{ModelFactory.MODELS[x]['icon']} {ModelFactory.MODELS[x]['name']}"
        )
        st.session_state.model_types = selected_models
        
        # Info about models
        for model in selected_models:
            info = model_factory.get_model_info(model)
            with st.expander(f"{info['icon']} {info['name']}", expanded=False):
                st.write(info['description'])
    
    # Combination counter
    n_combinations = len(selected_prep) * len(selected_models)
    st.info(f"💡 **Total Combinations:** {n_combinations} models will be trained ({len(selected_prep)} preprocessing × {len(selected_models)} algorithms)")
    
    # Advanced settings (Expert mode)
    if st.session_state.get('user_mode') == 'Expert':
        with st.expander("🔧 Advanced Settings", expanded=True):
            st.markdown("##### 🎯 Validation Method")
            st.markdown("Choose **ONE** validation strategy:")
            
            validation_method = st.radio(
                "Select validation method:",
                options=['kfold', 'train_test', 'loo'],
                format_func=lambda x: {
                    'kfold': '🔄 K-Fold Cross-Validation (Recommended)',
                    'train_test': '📊 Train-Test Split',
                    'loo': '🎯 Leave-One-Out (LOO)'
                }[x],
                index=0,
                horizontal=False
            )
            st.session_state.validation_method = validation_method
            
            st.markdown("---")
            
            # Show options based on selected method
            if validation_method == 'kfold':
                cv_folds = st.slider(
                    "Number of folds",
                    min_value=2,
                    max_value=10,
                    value=st.session_state.get('cv_folds', 5)
                )
                st.session_state.cv_folds = cv_folds
                st.session_state.test_size = 0.2  # Default for final evaluation
                st.info(f"📊 Data will be split into {cv_folds} folds. Each fold uses {100//cv_folds}% as test set.")
                
            elif validation_method == 'train_test':
                test_size = st.slider(
                    "Test set size (%)",
                    min_value=10,
                    max_value=40,
                    value=int(st.session_state.get('test_size', 0.2) * 100),
                    step=5
                )
                st.session_state.test_size = test_size / 100
                st.session_state.cv_folds = 5  # Default but not used
                st.info(f"📊 Training: {100-test_size}% | Testing: {test_size}%")
                
            elif validation_method == 'loo':
                n_samples = len(st.session_state.data) if st.session_state.get('data') is not None else 0
                st.session_state.cv_folds = n_samples
                st.session_state.test_size = 0.2
                st.info(f"📊 Leave-One-Out: {n_samples} iterations (each sample tested once)")
                st.warning("⚠️ LOO can be slow for large datasets (>100 samples)")
            
            st.markdown("---")
            
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                st.markdown("##### 🎲 Reproducibility")
                random_state = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=9999,
                    value=st.session_state.get('random_state', 42)
                )
                st.session_state.random_state = random_state
            
            with adv_col2:
                st.markdown("##### ⚡ Performance")
                st.checkbox("Enable parallel processing", value=True, disabled=True)
                st.checkbox("Save intermediate results", value=True)
    
    # Ready to train indicator
    if selected_prep and selected_models:
        st.success(f"✅ Ready to train {n_combinations} model combinations. Go to the Training tab to start.")
    else:
        st.warning("⚠️ Please select at least one preprocessing method and one algorithm.")


def render_training_tab():
    """Render the training tab."""
    
    st.markdown("### Step 5: Training Execution")
    
    # Check prerequisites
    if st.session_state.get('data') is None:
        st.warning("⚠️ Please upload data first.")
        return
    
    if not st.session_state.get('preprocessing_methods') or not st.session_state.get('model_types'):
        st.warning("⚠️ Please select preprocessing methods and algorithms in the Configuration tab.")
        return
    
    df = st.session_state.data
    target_col = st.session_state.target_col
    prep_methods = st.session_state.preprocessing_methods
    model_types = st.session_state.model_types
    
    n_combinations = len(prep_methods) * len(model_types)
    
    # Training summary
    st.markdown(f"""
    <div style="background: #1E2329; border: 1px solid #3A3F4B; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h4 style="color: #4A90E2;">🚀 Ready to Train</h4>
        <p style="color: #FAFAFA;">
            <strong>{n_combinations}</strong> model combinations will be trained:
        </p>
        <ul style="color: #B0B0B0;">
            <li><strong>Preprocessing:</strong> {', '.join(prep_methods)}</li>
            <li><strong>Algorithms:</strong> {', '.join([m.upper() for m in model_types])}</li>
            <li><strong>Estimated Time:</strong> 2-5 minutes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Start training button
    if st.button("🚀 START AUTOMATED TRAINING", type="primary", use_container_width=True):
        run_training(df, target_col, prep_methods, model_types)
    
    # Show results if training is complete
    if st.session_state.get('training_complete'):
        st.markdown("---")
        render_training_results()


def run_training(df, target_col, prep_methods, model_types):
    """Run the automated training process."""
    
    orchestrator = TrainingOrchestrator()
    
    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_model_text = st.empty()
    results_container = st.container()
    
    completed_results = []
    
    def progress_callback(info):
        """Callback for training progress updates."""
        progress = info['current'] / info['total']
        progress_bar.progress(progress)
        status_text.markdown(f"**Progress:** {info['current']}/{info['total']} models ({progress*100:.0f}%)")
        current_model_text.markdown(f"**Current:** {info['preprocessing'].title()} + {info['model'].upper()} - {info['status'].capitalize()}")
        
        if info['status'] == 'completed' and 'result' in info:
            result = info['result']
            if result.get('status') == 'success':
                completed_results.append(result)
    
    # Run training
    with st.spinner("Training in progress..."):
        start_time = time.time()
        
        results = orchestrator.train_all_combinations(
            df=df,
            target_col=target_col,
            preprocessing_methods=prep_methods,
            model_types=model_types,
            test_size=st.session_state.get('test_size', 0.2),
            cv_folds=st.session_state.get('cv_folds', 5),
            random_state=st.session_state.get('random_state', 42),
            progress_callback=progress_callback
        )
        
        total_time = time.time() - start_time
    
    # Save results
    save_results_to_session(results)
    st.session_state.orchestrator = orchestrator
    
    # Clear progress indicators
    progress_bar.progress(1.0)
    status_text.empty()
    current_model_text.empty()
    
    # Success message
    st.balloons()
    st.success(f"🎉 Training complete! {results['successful_models']} models trained in {format_time(total_time)}")
    
    # Auto-navigate to Model Comparison
    st.session_state.current_page = "model_comparison"
    st.rerun()


def render_training_results():
    """Render training results summary."""
    
    results = st.session_state.get('training_results', {})
    
    if not results:
        return
    
    st.markdown("### 🎉 Training Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", results.get('total_models', 0))
    with col2:
        st.metric("Successful", results.get('successful_models', 0))
    with col3:
        st.metric("Total Time", format_time(results.get('total_time', 0)))
    with col4:
        best = results.get('best_model', {})
        st.metric("Best R²", f"{best.get('test_r2', 0):.4f}" if best else "N/A")
    
    # Best model highlight
    if results.get('best_model'):
        best = results['best_model']
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%); 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
            <h3 style="color: white;">🏆 Best Model</h3>
            <h2 style="color: white;">{best.get('model_name', 'N/A')}</h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem;">
                R² = {best.get('test_r2', 0):.4f} | RMSE = {best.get('test_rmse', 0):.4f} | RPD = {best.get('rpd', 0):.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Top 5 models
    st.markdown("#### 🏅 Top 5 Models")
    
    all_results = results.get('results', [])
    successful = [r for r in all_results if r.get('status') == 'success']
    sorted_results = sorted(successful, key=lambda x: x.get('test_r2', 0), reverse=True)[:5]
    
    for i, r in enumerate(sorted_results, 1):
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
        
        with col1:
            st.markdown(f"### {get_medal(i)}")
        with col2:
            st.write(f"**{r.get('model_name', 'N/A')}**")
        with col3:
            r2 = r.get('test_r2', 0)
            color = get_performance_color(r2)
            st.markdown(f"<span style='color: {color}'>R² = {r2:.4f}</span>", unsafe_allow_html=True)
        with col4:
            st.write(f"RMSE = {r.get('test_rmse', 0):.4f}")
        with col5:
            st.write(f"RPD = {r.get('rpd', 0):.2f}")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Full Comparison", use_container_width=True):
            st.session_state.current_page = "model_comparison"
            st.rerun()
    
    with col2:
        if st.button("🔬 Analyze Diagnostics", use_container_width=True):
            st.session_state.current_page = "diagnostics"
            st.rerun()
    
    with col3:
        if st.button("📈 Export Results", use_container_width=True):
            st.session_state.current_page = "analytics_export"
            st.rerun()


def render_playground_tab(model_factory: ModelFactory):
    """Render the model playground tab with Grid Search hyperparameter tuning."""
    
    st.markdown("### 🎮 Model Playground")
    
    if st.session_state.get('user_mode') != 'Expert':
        st.info("ℹ️ Switch to Expert mode in the sidebar to access the Model Playground.")
        return
    
    if not st.session_state.get('training_complete'):
        st.warning("⚠️ Complete a training run first to use the playground.")
        return
    
    st.markdown("""
    Use **Grid Search** to automatically find the best hyperparameters for your model.
    """)
    
    # Model selection
    results = st.session_state.get('training_results', {}).get('results', [])
    successful = [r for r in results if r.get('status') == 'success']
    model_names = [r.get('model_name', 'Unknown') for r in successful]
    
    selected_model = st.selectbox(
        "Select model to tune:",
        options=model_names
    )
    
    if selected_model:
        # Find the model result
        model_result = next((r for r in successful if r.get('model_name') == selected_model), None)
        
        if model_result:
            model_type = model_result.get('model_type', '')
            preprocessing = model_result.get('preprocessing', '')
            
            # Current performance display
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("#### 📈 Current Performance")
                st.metric("R²", f"{model_result.get('test_r2', 0):.4f}")
                st.metric("RMSE", f"{model_result.get('test_rmse', 0):.4f}")
                st.metric("RPD", f"{model_result.get('rpd', 0):.2f}")
            
            with col1:
                st.markdown("#### 🔍 Grid Search Parameters")
                
                # Define parameter grids for each model type
                param_grid = {}
                
                if model_type in ['gbrt', 'cubist']:
                    st.markdown("**Gradient Boosting Parameters:**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        n_est_options = st.multiselect(
                            "n_estimators (try multiple)",
                            options=[50, 100, 150, 200, 300],
                            default=[50, 100, 150],
                            key="gs_n_est"
                        )
                        param_grid['n_estimators'] = n_est_options if n_est_options else [100]
                    
                    with col_b:
                        max_depth_options = st.multiselect(
                            "max_depth (try multiple)",
                            options=[3, 5, 7, 10, 15],
                            default=[3, 5, 7],
                            key="gs_max_depth"
                        )
                        param_grid['max_depth'] = max_depth_options if max_depth_options else [5]
                    
                    lr_options = st.multiselect(
                        "learning_rate (try multiple)",
                        options=[0.01, 0.05, 0.1, 0.15, 0.2],
                        default=[0.05, 0.1],
                        key="gs_lr"
                    )
                    param_grid['learning_rate'] = lr_options if lr_options else [0.1]
                
                elif model_type == 'plsr':
                    st.markdown("**PLSR Parameters:**")
                    n_comp_options = st.multiselect(
                        "n_components (try multiple)",
                        options=[2, 5, 8, 10, 15, 20, 25, 30],
                        default=[5, 10, 15, 20],
                        key="gs_n_comp"
                    )
                    param_grid['n_components'] = n_comp_options if n_comp_options else [10]
                
                elif model_type == 'svr':
                    st.markdown("**SVR Parameters:**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        c_options = st.multiselect(
                            "C (try multiple)",
                            options=[0.1, 0.5, 1.0, 5.0, 10.0],
                            default=[0.1, 1.0, 10.0],
                            key="gs_c"
                        )
                        param_grid['C'] = c_options if c_options else [1.0]
                    
                    with col_b:
                        eps_options = st.multiselect(
                            "epsilon (try multiple)",
                            options=[0.01, 0.05, 0.1, 0.2],
                            default=[0.05, 0.1],
                            key="gs_eps"
                        )
                        param_grid['epsilon'] = eps_options if eps_options else [0.1]
                    
                    gamma_options = st.multiselect(
                        "gamma (try multiple)",
                        options=['scale', 'auto'],
                        default=['scale'],
                        key="gs_gamma"
                    )
                    param_grid['gamma'] = gamma_options if gamma_options else ['scale']
                
                elif model_type == 'krr':
                    st.markdown("**KRR Parameters:**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        alpha_options = st.multiselect(
                            "alpha (try multiple)",
                            options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                            default=[0.1, 1.0, 10.0],
                            key="gs_alpha"
                        )
                        param_grid['alpha'] = alpha_options if alpha_options else [1.0]
                    
                    with col_b:
                        kernel_options = st.multiselect(
                            "kernel (try multiple)",
                            options=['rbf', 'polynomial', 'linear'],
                            default=['rbf'],
                            key="gs_kernel"
                        )
                        param_grid['kernel'] = kernel_options if kernel_options else ['rbf']
                
                # Calculate total combinations
                total_combinations = 1
                for values in param_grid.values():
                    total_combinations *= len(values)
                
                st.info(f"📊 Total combinations to try: **{total_combinations}** (with 5-fold CV = {total_combinations * 5} fits)")
            
            st.markdown("---")
            
            # Grid Search settings
            col1, col2 = st.columns(2)
            with col1:
                cv_folds = st.selectbox("Cross-validation folds:", [3, 5, 10], index=1, key="gs_cv")
            with col2:
                scoring = st.selectbox("Scoring metric:", ['r2', 'neg_root_mean_squared_error'], index=0, key="gs_scoring")
            
            if st.button("🚀 Run Grid Search", use_container_width=True, type="primary"):
                with st.spinner(f"Running Grid Search ({total_combinations} combinations)..."):
                    try:
                        from sklearn.model_selection import GridSearchCV, train_test_split
                        from modules.preprocessing import SpectralPreprocessor
                        from modules.evaluation import ModelEvaluator
                        import time
                        
                        df = st.session_state.data
                        target_col = st.session_state.target_col
                        
                        # Prepare data
                        feature_cols = [c for c in df.columns if c != target_col]
                        X = df[feature_cols].select_dtypes(include=[np.number]).values
                        y = df[target_col].values
                        
                        # Apply preprocessing
                        preprocessor = SpectralPreprocessor()
                        X_processed = preprocessor.preprocess(X, preprocessing)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_processed, y, test_size=0.2, random_state=42
                        )
                        
                        # Create base model
                        base_model = model_factory.create_model(model_type)
                        
                        # Handle pipeline models (extract the actual model)
                        if hasattr(base_model, 'named_steps'):
                            actual_model = base_model.named_steps.get('model', base_model)
                            # Prefix param names for pipeline
                            param_grid_prefixed = {f'model__{k}': v for k, v in param_grid.items()}
                        else:
                            actual_model = base_model
                            param_grid_prefixed = param_grid
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run Grid Search
                        start_time = time.time()
                        
                        grid_search = GridSearchCV(
                            base_model,
                            param_grid_prefixed,
                            cv=cv_folds,
                            scoring=scoring,
                            n_jobs=-1,
                            verbose=0,
                            return_train_score=True
                        )
                        
                        status_text.text("Fitting models...")
                        grid_search.fit(X_train, y_train)
                        
                        progress_bar.progress(100)
                        search_time = time.time() - start_time
                        
                        # Get best model and evaluate
                        best_model = grid_search.best_estimator_
                        y_pred = best_model.predict(X_test)
                        
                        # Calculate metrics
                        evaluator = ModelEvaluator()
                        metrics = evaluator.calculate_metrics(y_test, y_pred)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display results
                        st.success(f"✅ Grid Search complete in {search_time:.1f}s!")
                        
                        st.markdown("#### 🏆 Best Parameters Found")
                        
                        # Clean up parameter names for display
                        best_params_display = {}
                        for k, v in grid_search.best_params_.items():
                            clean_key = k.replace('model__', '')
                            best_params_display[clean_key] = v
                        
                        # Display best params in nice format
                        params_cols = st.columns(len(best_params_display))
                        for i, (param, value) in enumerate(best_params_display.items()):
                            with params_cols[i % len(params_cols)]:
                                st.metric(param, value)
                        
                        st.markdown("#### 📊 Performance Comparison")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            old_r2 = model_result.get('test_r2', 0)
                            new_r2 = metrics.get('r2', 0)
                            delta = new_r2 - old_r2
                            st.metric("R²", f"{new_r2:.4f}", f"{delta:+.4f}")
                        
                        with col2:
                            old_rmse = model_result.get('test_rmse', 0)
                            new_rmse = metrics.get('rmse', 0)
                            delta = old_rmse - new_rmse  # Lower is better
                            st.metric("RMSE", f"{new_rmse:.4f}", f"{delta:+.4f}")
                        
                        with col3:
                            old_rpd = model_result.get('rpd', 0)
                            new_rpd = metrics.get('rpd', 0)
                            delta = new_rpd - old_rpd
                            st.metric("RPD", f"{new_rpd:.2f}", f"{delta:+.2f}")
                        
                        with col4:
                            st.metric("Best CV Score", f"{grid_search.best_score_:.4f}")
                        
                        # Improvement message
                        if new_r2 > old_r2:
                            improvement = ((new_r2 - old_r2) / old_r2) * 100 if old_r2 > 0 else 0
                            st.success(f"🎉 **Improved R² by {improvement:.1f}%!** Grid Search found better parameters.")
                        elif new_r2 < old_r2:
                            st.warning(f"⚠️ R² decreased. The default parameters may already be optimal for this dataset.")
                        else:
                            st.info("Performance unchanged. Current parameters are already optimal.")
                        
                        # Show all results in expander
                        with st.expander("📋 View All Grid Search Results"):
                            results_df = pd.DataFrame(grid_search.cv_results_)
                            
                            # Select relevant columns
                            display_cols = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
                            results_df = results_df[display_cols].sort_values('rank_test_score')
                            results_df.columns = ['Parameters', 'Mean Score', 'Std Score', 'Rank']
                            
                            st.dataframe(results_df.head(10), use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Error during Grid Search: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
