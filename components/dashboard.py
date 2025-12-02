"""
Dashboard Component
===================
Page 1: Home/Overview page with statistics and navigation.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.utils import render_metric_card, get_performance_color, get_medal


def render_dashboard():
    """Render the dashboard/home page."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; background: linear-gradient(90deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🌱 SPECTRAL SOIL MODELER
        </h1>
        <p style="color: #B0B0B0; font-size: 1.2rem;">
            Professional ML Platform for Soil Spectroscopy
        </p>
        <hr style="border: none; height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent);">
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Navigation Cards
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%); 
                    border-radius: 12px; padding: 1.5rem; text-align: center; 
                    transition: all 0.3s ease; cursor: pointer;">
            <div style="font-size: 2.5rem;">📤</div>
            <h3 style="color: white; margin: 0.5rem 0;">Upload Data</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Load your spectral dataset
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Upload", key="nav_upload", use_container_width=True):
            st.session_state.current_page = "upload_train"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #50C878 0%, #3CB371 100%); 
                    border-radius: 12px; padding: 1.5rem; text-align: center; 
                    transition: all 0.3s ease; cursor: pointer;">
            <div style="font-size: 2.5rem;">🎯</div>
            <h3 style="color: white; margin: 0.5rem 0;">Train Models</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Automated ML training
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Training", key="nav_train", use_container_width=True):
            st.session_state.current_page = "upload_train"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFB84D 0%, #FFA726 100%); 
                    border-radius: 12px; padding: 1.5rem; text-align: center; 
                    transition: all 0.3s ease; cursor: pointer;">
            <div style="font-size: 2.5rem;">📊</div>
            <h3 style="color: white; margin: 0.5rem 0;">Compare</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                View model leaderboard
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Results", key="nav_compare", use_container_width=True):
            st.session_state.current_page = "model_comparison"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%); 
                    border-radius: 12px; padding: 1.5rem; text-align: center; 
                    transition: all 0.3s ease; cursor: pointer;">
            <div style="font-size: 2.5rem;">📈</div>
            <h3 style="color: white; margin: 0.5rem 0;">Export</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                Generate reports
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Export Results", key="nav_export", use_container_width=True):
            st.session_state.current_page = "analytics_export"
            st.rerun()
    
    st.markdown("---")
    
    # Statistics Section
    st.markdown("### 📊 Quick Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    # Get statistics from session state
    training_runs = len(st.session_state.get('training_history', []))
    total_models = st.session_state.get('total_models', 0)
    best_r2 = st.session_state.get('best_r2', None)
    
    with stat_col1:
        st.metric(
            label="🔄 Training Runs",
            value=training_runs,
            delta="+1" if training_runs > 0 else None
        )
    
    with stat_col2:
        st.metric(
            label="🤖 Models Trained",
            value=total_models
        )
    
    with stat_col3:
        st.metric(
            label="🏆 Best R²",
            value=f"{best_r2:.4f}" if best_r2 else "N/A",
            delta="Excellent" if best_r2 and best_r2 > 0.85 else None
        )
    
    with stat_col4:
        st.metric(
            label="📐 Algorithms",
            value="5",
            help="PLSR, Cubist, GBRT, KRR, SVR"
        )
    
    st.markdown("---")
    
    # Current Session Status
    st.markdown("### 📋 Current Session")
    
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        st.markdown("#### Data Status")
        if st.session_state.get('data') is not None:
            metadata = st.session_state.get('metadata', {})
            st.success(f"✅ Dataset loaded: {metadata.get('filename', 'Unknown')}")
            st.info(f"📊 Samples: {metadata.get('n_samples', 0)} | Features: {metadata.get('n_features', 0)}")
            if st.session_state.get('target_col'):
                st.info(f"🎯 Target: {st.session_state.target_col}")
        else:
            st.warning("⚠️ No dataset loaded. Upload data to get started.")
    
    with col_status2:
        st.markdown("#### Training Status")
        if st.session_state.get('training_complete'):
            results = st.session_state.get('training_results', {})
            best = results.get('best_model', {})
            st.success(f"✅ Training complete!")
            st.info(f"🏆 Best: {best.get('model_name', 'N/A')} (R²: {best.get('test_r2', 0):.4f})")
            st.info(f"⏱️ Total time: {results.get('total_time', 0):.1f}s")
        else:
            st.warning("⚠️ No training completed yet.")
    
    st.markdown("---")
    
    # Recent Activity
    if st.session_state.get('training_history'):
        st.markdown("### 🕐 Recent Activity")
        
        for entry in reversed(st.session_state.training_history[-5:]):
            with st.expander(f"📁 {entry.get('run_id', 'Unknown Run')}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Time:** {entry.get('timestamp', 'N/A')[:19]}")
                with col2:
                    st.write(f"**Models:** {entry.get('n_models', 0)}")
                with col3:
                    r2 = entry.get('best_r2', 0)
                    st.write(f"**Best R²:** {r2:.4f}" if r2 else "N/A")
    
    # Getting Started Guide (if no data loaded)
    if st.session_state.get('data') is None:
        st.markdown("---")
        st.markdown("### 🎓 Getting Started")
        
        st.markdown("""
        <div style="background: #1E2329; border: 1px solid #3A3F4B; border-radius: 12px; padding: 1.5rem;">
            <h4 style="color: #4A90E2;">Welcome to Spectral Soil Modeler!</h4>
            <p style="color: #B0B0B0;">Follow these steps to get started:</p>
            <ol style="color: #FAFAFA;">
                <li><strong>Upload your spectral data</strong> - Support for CSV, XLS, XLSX formats</li>
                <li><strong>Select target property</strong> - Choose the soil property to predict</li>
                <li><strong>Configure training</strong> - Select preprocessing methods and ML algorithms</li>
                <li><strong>Start training</strong> - Let the system train all model combinations</li>
                <li><strong>Analyze results</strong> - Compare models and export best performers</li>
            </ol>
            <hr style="border-color: #3A3F4B;">
            <p style="color: #B0B0B0; font-size: 0.9rem;">
                <strong>Supported Models:</strong> PLSR, Cubist, GBRT, KRR, SVR<br>
                <strong>Preprocessing:</strong> Reflectance, Absorbance, Continuum Removal
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #808080; font-size: 0.8rem; padding: 1rem;">
        
        <p>Spectral Soil Modeler </p>
    </div>
    """, unsafe_allow_html=True)
