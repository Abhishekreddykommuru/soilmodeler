"""
Spectral Soil Modeler - Main Application Entry Point
=====================================================
Professional ML Platform for Soil Spectroscopy
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Spectral Soil Modeler",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components after page config
from components.dashboard import render_dashboard
from components.upload_train import render_upload_train
from components.model_comparison import render_model_comparison
from components.model_archive import render_model_archive
from components.diagnostics import render_diagnostics
from components.analytics_export import render_analytics_export
from modules.utils import load_custom_css, initialize_session_state

# Load custom CSS
load_custom_css()

# Initialize session state
initialize_session_state()

def render_sidebar():
    """Render the navigation sidebar"""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">🌱</span>
            <h2 style="color: white; margin: 0.5rem 0;">Spectral Soil Modeler</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        pages = {
            "🏠 Dashboard": "dashboard",
            "📤 Upload & Train": "upload_train",
            "📊 Model Comparison": "model_comparison",
            "📂 Model Archive": "model_archive",
            "🔬 Diagnostics": "diagnostics",
            "📈 Analytics & Export": "analytics_export"
        }
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
        
        st.markdown("---")
        
        # Settings section
        st.markdown("### ⚙️ Settings")
        
        # User mode toggle
        user_mode = st.radio(
            "Mode",
            ["Beginner", "Expert"],
            index=0 if st.session_state.get('user_mode', 'Beginner') == 'Beginner' else 1,
            horizontal=True,
            help="Expert mode shows advanced configuration options"
        )
        st.session_state.user_mode = user_mode
        
        st.markdown("---")
        
        # Quick stats if available
        if st.session_state.get('training_complete', False):
            st.markdown("### 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Models", st.session_state.get('total_models', 0))
            with col2:
                best_r2 = st.session_state.get('best_r2', 0)
                st.metric("Best R²", f"{best_r2:.3f}" if best_r2 else "N/A")

def main():
    """Main application function"""
    # Render sidebar
    render_sidebar()
    
    # Get current page
    current_page = st.session_state.get('current_page', 'dashboard')
    
    # Render the appropriate page
    if current_page == "dashboard":
        render_dashboard()
    elif current_page == "upload_train":
        render_upload_train()
    elif current_page == "model_comparison":
        render_model_comparison()
    elif current_page == "model_archive":
        render_model_archive()
    elif current_page == "diagnostics":
        render_diagnostics()
    elif current_page == "analytics_export":
        render_analytics_export()
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
