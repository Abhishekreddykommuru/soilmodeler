"""
Utility Functions Module
========================
Common utilities, session state management, and helper functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import io


def load_custom_css():
    """Load custom CSS styles for the application."""
    css = """
    <style>
    /* Main app styling */
    .main {
        padding: 1rem;
    }
    
    /* Card styling */
    .stCard {
        background-color: #262B33;
        border: 1px solid #3A3F4B;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #262B33 0%, #1E2329 100%);
        border: 1px solid #3A3F4B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #B0B0B0;
        text-transform: uppercase;
    }
    
    /* Navigation cards */
    .nav-card {
        background: #262B33;
        border: 1px solid #3A3F4B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-card:hover {
        background: #2E3440;
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .nav-card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .nav-card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }
    
    .nav-card-desc {
        font-size: 0.9rem;
        color: #B0B0B0;
    }
    
    /* Status boxes */
    .status-success {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
    }
    
    .status-warning {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #FFC107;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
    }
    
    .status-error {
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
    }
    
    .status-info {
        background: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
    }
    
    /* Progress bar custom styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Primary action button */
    .primary-button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%) !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    
    /* Upload area styling */
    .uploadedFile {
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.1);
        border: none;
        color: white;
        width: 100%;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* Leaderboard styling */
    .leaderboard-medal {
        font-size: 1.5rem;
    }
    
    /* Performance colors */
    .perf-excellent { color: #4CAF50; }
    .perf-good { color: #8BC34A; }
    .perf-fair { color: #FFC107; }
    .perf-poor { color: #FF6B6B; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'current_page': 'dashboard',
        'user_mode': 'Beginner',
        'data': None,
        'metadata': None,
        'target_col': None,
        'training_complete': False,
        'training_results': None,
        'trained_models': {},
        'best_model': None,
        'best_r2': None,
        'total_models': 0,
        'preprocessing_methods': ['reflectance', 'absorbance', 'continuum_removal'],
        'model_types': ['plsr', 'cubist', 'gbrt', 'krr', 'svr'],
        'test_size': 0.2,
        'cv_folds': 5,
        'random_state': 42,
        'training_history': [],
        'selected_model_for_analysis': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_metric(value: float, metric_type: str = 'r2') -> str:
    """Format a metric value for display."""
    if value is None or np.isnan(value) or np.isinf(value):
        return "N/A"
    
    if metric_type == 'r2':
        return f"{value:.4f}"
    elif metric_type == 'rmse':
        return f"{value:.4f}"
    elif metric_type == 'percentage':
        return f"{value:.1f}%"
    elif metric_type == 'time':
        return format_time(value)
    else:
        return f"{value:.3f}"


def get_performance_color(r2: float) -> str:
    """Get color class based on R² performance."""
    if r2 >= 0.85:
        return "#4CAF50"  # Excellent - Green
    elif r2 >= 0.75:
        return "#8BC34A"  # Good - Light green
    elif r2 >= 0.60:
        return "#FFC107"  # Fair - Yellow
    else:
        return "#FF6B6B"  # Poor - Red


def get_performance_label(r2: float) -> str:
    """Get label based on R² performance."""
    if r2 >= 0.85:
        return "Excellent"
    elif r2 >= 0.75:
        return "Good"
    elif r2 >= 0.60:
        return "Fair"
    else:
        return "Poor"


def get_medal(rank: int) -> str:
    """Get medal emoji for top ranks."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    return medals.get(rank, str(rank))


def create_download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    """Create a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    return f'<a href="data:text/csv;charset=utf-8,{csv}" download="{filename}">{label}</a>'


def render_metric_card(label: str, value: str, icon: str = "📊", delta: str = None):
    """Render a metric card."""
    delta_html = f'<div style="color: {"#4CAF50" if delta and "+" in str(delta) else "#FF6B6B"}; font-size: 0.8rem;">{delta}</div>' if delta else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_nav_card(title: str, description: str, icon: str, page_key: str):
    """Render a navigation card."""
    if st.button(f"{icon} {title}", key=f"nav_card_{page_key}", use_container_width=True):
        st.session_state.current_page = page_key
        st.rerun()


def render_status_box(message: str, status: str = "info"):
    """Render a status box."""
    class_name = f"status-{status}"
    icons = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }
    icon = icons.get(status, "ℹ️")
    
    st.markdown(f"""
    <div class="{class_name}">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)


def save_results_to_session(results: Dict[str, Any]):
    """Save training results to session state and disk."""
    st.session_state.training_complete = True
    st.session_state.training_results = results
    st.session_state.total_models = results.get('total_models', 0)
    
    if results.get('best_model'):
        st.session_state.best_model = results['best_model']
        st.session_state.best_r2 = results['best_model'].get('test_r2', 0)
    
    # Add to history
    run_id = results.get('run_id', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    history_entry = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'n_models': results.get('total_models', 0),
        'best_r2': st.session_state.best_r2,
        'best_model': results['best_model'].get('model_name', '') if results.get('best_model') else '',
        'results': results.get('results', [])  # Store full results in history for loading later
    }
    st.session_state.training_history.append(history_entry)
    
    # Save full results to disk for persistence across sessions
    try:
        models_dir = Path("models") / run_id
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON (remove non-serializable items like model objects)
        serializable_results = {
            'run_id': run_id,
            'timestamp': history_entry['timestamp'],
            'total_models': results.get('total_models', 0),
            'successful_models': results.get('successful_models', 0),
            'failed_models': results.get('failed_models', 0),
            'total_time': results.get('total_time', 0),
            'results': []
        }
        
        for r in results.get('results', []):
            # Create a serializable copy of each result
            result_copy = {
                'model_name': r.get('model_name', ''),
                'preprocessing': r.get('preprocessing', ''),
                'model_type': r.get('model_type', ''),
                'status': r.get('status', ''),
                'test_r2': r.get('test_r2', 0),
                'test_rmse': r.get('test_rmse', 0),
                'test_mae': r.get('test_mae', 0),
                'train_r2': r.get('train_r2', 0),
                'train_rmse': r.get('train_rmse', 0),
                'rpd': r.get('rpd', 0),
                'correlation': r.get('correlation', 0),
                'bias': r.get('bias', 0),
                'cv_mean': r.get('cv_mean', 0),
                'cv_std': r.get('cv_std', 0),
                'train_time': r.get('train_time', 0),
                'n_train': r.get('n_train', 0),
                'n_test': r.get('n_test', 0)
            }
            
            # Include predictions if available (for scatter plots and residuals)
            if 'y_test' in r and r['y_test'] is not None:
                result_copy['y_test'] = r['y_test'] if isinstance(r['y_test'], list) else list(r['y_test'])
            if 'y_pred_test' in r and r['y_pred_test'] is not None:
                result_copy['y_pred_test'] = r['y_pred_test'] if isinstance(r['y_pred_test'], list) else list(r['y_pred_test'])
            if 'y_pred' in r and r['y_pred'] is not None:
                result_copy['y_pred'] = r['y_pred'] if isinstance(r['y_pred'], list) else list(r['y_pred'])
            if 'cv_scores' in r and r['cv_scores'] is not None:
                result_copy['cv_scores'] = r['cv_scores'] if isinstance(r['cv_scores'], list) else list(r['cv_scores'])
            
            # Include feature importance if available
            if 'feature_importance' in r and r['feature_importance'] is not None:
                result_copy['feature_importance'] = r['feature_importance']
            
            serializable_results['results'].append(result_copy)
        
        # Find and store best model info
        successful = [r for r in serializable_results['results'] if r.get('status') == 'success']
        if successful:
            best = max(successful, key=lambda x: x.get('test_r2', 0))
            serializable_results['best_model'] = best
        
        # Write to disk
        with open(models_dir / "results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
    except Exception as e:
        st.warning(f"Could not save results to disk: {e}")


def clear_training_results():
    """Clear training results from session state."""
    st.session_state.training_complete = False
    st.session_state.training_results = None
    st.session_state.trained_models = {}
    st.session_state.best_model = None
    st.session_state.best_r2 = None
    st.session_state.total_models = 0


def export_results_to_excel(results: List[Dict]) -> bytes:
    """Export results to Excel file."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main results
        df = pd.DataFrame(results)
        cols_to_export = [
            'preprocessing', 'model_type', 'test_r2', 'test_rmse', 
            'rpd', 'test_mae', 'correlation', 'bias', 
            'cv_mean', 'cv_std', 'train_time'
        ]
        available_cols = [c for c in cols_to_export if c in df.columns]
        df[available_cols].to_excel(writer, sheet_name='Results', index=False)
        
        # Summary statistics
        if 'test_r2' in df.columns:
            summary = pd.DataFrame({
                'Metric': ['R²', 'RMSE', 'RPD'],
                'Mean': [df['test_r2'].mean(), df.get('test_rmse', pd.Series()).mean(), df.get('rpd', pd.Series()).mean()],
                'Std': [df['test_r2'].std(), df.get('test_rmse', pd.Series()).std(), df.get('rpd', pd.Series()).std()],
                'Min': [df['test_r2'].min(), df.get('test_rmse', pd.Series()).min(), df.get('rpd', pd.Series()).min()],
                'Max': [df['test_r2'].max(), df.get('test_rmse', pd.Series()).max(), df.get('rpd', pd.Series()).max()]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
    
    return output.getvalue()
