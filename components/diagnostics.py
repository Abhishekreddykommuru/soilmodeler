"""
Diagnostics Component
=====================
Page 5: Advanced model diagnostics, PCA analysis, and feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from modules.visualization import Visualizer
from modules.evaluation import ModelEvaluator
from modules.utils import get_performance_color


# Soil science wavelength interpretations
WAVELENGTH_INTERPRETATIONS = {
    (350, 450): {"name": "Blue/UV Region", "associated": "Iron oxides, organic matter"},
    (450, 520): {"name": "Blue-Green", "associated": "Chlorophyll, organic matter, iron oxides"},
    (520, 600): {"name": "Green-Yellow", "associated": "Vegetation, iron oxides"},
    (600, 700): {"name": "Red Region", "associated": "Chlorophyll absorption, iron oxides"},
    (700, 900): {"name": "Near-IR Edge", "associated": "Vegetation structure, cell structure"},
    (900, 1100): {"name": "NIR Plateau", "associated": "Plant structure, moisture"},
    (1100, 1300): {"name": "NIR", "associated": "Clay minerals, water"},
    (1350, 1450): {"name": "Water Band I", "associated": "Strong water absorption (O-H)"},
    (1450, 1800): {"name": "SWIR I", "associated": "Organic matter, clay minerals"},
    (1800, 2000): {"name": "Water Band II", "associated": "Water absorption, hydroxyl groups"},
    (2000, 2200): {"name": "SWIR II", "associated": "Protein, cellulose, lignin"},
    (2200, 2350): {"name": "Clay Region", "associated": "Clay minerals (Al-OH, Mg-OH bonds)"},
    (2350, 2500): {"name": "Carbonate Region", "associated": "Carbonates, clay minerals"},
}


def get_wavelength_interpretation(wavelength: float) -> dict:
    """Get soil science interpretation for a wavelength."""
    for (low, high), info in WAVELENGTH_INTERPRETATIONS.items():
        if low <= wavelength <= high:
            return info
    return {"name": "Unknown Region", "associated": "No specific association"}


def render_diagnostics():
    """Render the diagnostics page."""
    
    st.markdown("""
    <h1 style="background: linear-gradient(90deg, #667eea, #764ba2); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🔬 Advanced Diagnostics
    </h1>
    """, unsafe_allow_html=True)
    
    # Check if results exist
    if not st.session_state.get('training_complete'):
        st.warning("⚠️ No training results available. Complete a training run first.")
        if st.button("Go to Upload & Train"):
            st.session_state.current_page = "upload_train"
            st.rerun()
        return
    
    results = st.session_state.get('training_results', {})
    all_results = results.get('results', [])
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        st.error("No successful models found.")
        return
    
    # Initialize
    visualizer = Visualizer()
    evaluator = ModelEvaluator()
    
    # Create tabs
    tabs = st.tabs(["🎯 Feature Importance", "📊 PCA Analysis", "📉 Residual Diagnostics", "🔍 Model Inspection"])
    
    # Tab 1: Feature Importance
    with tabs[0]:
        render_feature_importance(successful_results, visualizer)
    
    # Tab 2: PCA Analysis
    with tabs[1]:
        render_pca_analysis()
    
    # Tab 3: Residual Diagnostics
    with tabs[2]:
        render_residual_diagnostics(successful_results, visualizer, evaluator)
    
    # Tab 4: Model Inspection
    with tabs[3]:
        render_model_inspection(successful_results, evaluator)


def render_feature_importance(results: list, visualizer: Visualizer):
    """Render feature importance analysis with beautiful wavelength visualizations."""
    
    st.markdown("### 🎯 Feature Importance Analysis")
    st.markdown("Discover which spectral wavelengths are most important for predicting soil properties.")
    
    # Model selector - show ALL models
    model_names = [r.get('model_name', 'Unknown') for r in results]
    models_with_fi = [r.get('model_name', 'Unknown') for r in results if r.get('feature_importance')]
    
    if not models_with_fi:
        st.warning("⚠️ No feature importance data available yet. This may happen if:")
        st.markdown("""
        - Models are still being processed
        - Feature importance calculation failed
        - Results were loaded from a previous session without full data
        """)
        
        # Show which models should have feature importance
        st.info("""
        **Feature Importance Methods by Model:**
        - 🌳 **GBRT, Cubist**: Native feature importance (fast)
        - 📈 **PLSR**: Coefficient magnitudes (fast)
        - ⚡ **SVR, KRR**: Permutation importance (computed during training)
        """)
        return
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select model to analyze:",
            options=models_with_fi,
            key="fi_model_select"
        )
    
    with col2:
        # Show model type info
        model_result = next((r for r in results if r.get('model_name') == selected_model), None)
        if model_result:
            model_type = model_result.get('model_type', '').upper()
            preprocessing = model_result.get('preprocessing', '').title()
            st.markdown(f"**Model:** {model_type}")
            st.markdown(f"**Preprocessing:** {preprocessing}")
    
    if selected_model and model_result and model_result.get('feature_importance'):
        importance_dict = model_result['feature_importance']
        
        st.markdown("---")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_n = st.slider("Show top N wavelengths:", min_value=10, max_value=50, value=20)
        
        with col2:
            view_type = st.radio("View type:", ["Bar Chart", "Spectral Curve", "Both"], horizontal=True)
        
        with col3:
            show_interpretation = st.checkbox("Show soil interpretations", value=True)
        
        # Parse wavelengths and importance
        wavelengths = []
        importances = []
        feature_names_sorted = []
        
        for feature, imp in importance_dict.items():
            try:
                # Try to extract wavelength number from feature name
                wl_str = feature.replace('nm', '').replace('_', '.').strip()
                wl = float(''.join(c for c in wl_str if c.isdigit() or c == '.'))
                wavelengths.append(wl)
                importances.append(imp)
                feature_names_sorted.append(feature)
            except:
                continue
        
        if wavelengths:
            # Create DataFrame for easier manipulation
            df_importance = pd.DataFrame({
                'wavelength': wavelengths,
                'importance': importances,
                'feature': feature_names_sorted
            }).sort_values('wavelength')
            
            # Top important wavelengths
            df_top = df_importance.nlargest(top_n, 'importance')
            
            # ===== BAR CHART =====
            if view_type in ["Bar Chart", "Both"]:
                st.markdown("#### 📊 Top Important Wavelengths")
                
                fig_bar = go.Figure()
                
                # Color by importance
                colors = ['#FF6B6B' if imp > 0.8 else '#FFA500' if imp > 0.5 else '#4CAF50' 
                         for imp in df_top['importance']]
                
                fig_bar.add_trace(go.Bar(
                    x=[f"{wl:.0f}nm" for wl in df_top['wavelength']],
                    y=df_top['importance'],
                    marker_color=colors,
                    text=[f"{imp:.3f}" for imp in df_top['importance']],
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>"
                ))
                
                fig_bar.update_layout(
                    title=f"Top {top_n} Important Wavelengths - {selected_model}",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Relative Importance",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=450,
                    showlegend=False
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # ===== SPECTRAL CURVE =====
            if view_type in ["Spectral Curve", "Both"]:
                st.markdown("#### 🌈 Spectral Importance Pattern")
                
                fig_spectral = go.Figure()
                
                # Main importance curve
                df_sorted = df_importance.sort_values('wavelength')
                
                fig_spectral.add_trace(go.Scatter(
                    x=df_sorted['wavelength'],
                    y=df_sorted['importance'],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.3)',
                    line=dict(color='#FF6B6B', width=2),
                    name='Importance',
                    hovertemplate="<b>%{x:.0f}nm</b><br>Importance: %{y:.4f}<extra></extra>"
                ))
                
                # Mark top peaks
                peaks = df_importance.nlargest(5, 'importance')
                fig_spectral.add_trace(go.Scatter(
                    x=peaks['wavelength'],
                    y=peaks['importance'],
                    mode='markers+text',
                    marker=dict(size=12, color='#FFD700', symbol='star'),
                    text=[f"{wl:.0f}nm" for wl in peaks['wavelength']],
                    textposition='top center',
                    textfont=dict(color='white', size=10),
                    name='Top 5 Peaks',
                    hovertemplate="<b>Peak: %{x:.0f}nm</b><br>Importance: %{y:.4f}<extra></extra>"
                ))
                
                # Add region annotations
                if show_interpretation:
                    for (low, high), info in WAVELENGTH_INTERPRETATIONS.items():
                        if low >= df_sorted['wavelength'].min() and high <= df_sorted['wavelength'].max():
                            fig_spectral.add_vrect(
                                x0=low, x1=high,
                                fillcolor="rgba(100, 100, 100, 0.1)",
                                layer="below",
                                line_width=0
                            )
                
                fig_spectral.update_layout(
                    title=f"Spectral Importance Across Wavelengths - {selected_model}",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Relative Importance",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_spectral, use_container_width=True)
            
            # ===== IMPORTANT WAVELENGTH REGIONS =====
            st.markdown("---")
            st.markdown("#### 🎯 Important Wavelength Regions")
            
            # Group wavelengths into regions
            df_top_sorted = df_top.sort_values('wavelength')
            
            # Find continuous regions
            regions = []
            current_region = {'start': None, 'end': None, 'wavelengths': [], 'importances': []}
            
            for _, row in df_top_sorted.iterrows():
                wl = row['wavelength']
                imp = row['importance']
                
                if current_region['start'] is None:
                    current_region = {'start': wl, 'end': wl, 'wavelengths': [wl], 'importances': [imp]}
                elif wl - current_region['end'] <= 50:  # Within 50nm = same region
                    current_region['end'] = wl
                    current_region['wavelengths'].append(wl)
                    current_region['importances'].append(imp)
                else:
                    regions.append(current_region)
                    current_region = {'start': wl, 'end': wl, 'wavelengths': [wl], 'importances': [imp]}
            
            if current_region['start'] is not None:
                regions.append(current_region)
            
            # Sort regions by average importance
            regions = sorted(regions, key=lambda r: np.mean(r['importances']), reverse=True)
            
            # Display top regions
            for i, region in enumerate(regions[:5], 1):
                avg_imp = np.mean(region['importances'])
                max_imp = max(region['importances'])
                
                # Get interpretation
                mid_wl = (region['start'] + region['end']) / 2
                interpretation = get_wavelength_interpretation(mid_wl)
                
                # Medal emoji
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                
                # Color based on importance
                if avg_imp > 0.7:
                    color = "#FF6B6B"
                    level = "Very High"
                elif avg_imp > 0.5:
                    color = "#FFA500"
                    level = "High"
                else:
                    color = "#4CAF50"
                    level = "Moderate"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                            border-left: 4px solid {color};
                            border-radius: 8px;
                            padding: 1rem;
                            margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.5rem;">{medal}</span>
                            <strong style="color: white; font-size: 1.1rem;">
                                {region['start']:.0f} - {region['end']:.0f} nm
                            </strong>
                            <span style="color: {color}; margin-left: 10px;">({level} Importance)</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #888;">Avg: {avg_imp:.3f} | Max: {max_imp:.3f}</span>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; color: #aaa;">
                        <strong>{interpretation['name']}</strong>: {interpretation['associated']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # ===== DETAILED TABLE =====
            st.markdown("---")
            st.markdown("#### 📋 Detailed Importance Table")
            
            with st.expander("View full table", expanded=False):
                # Create detailed table
                table_data = []
                for _, row in df_top.iterrows():
                    interp = get_wavelength_interpretation(row['wavelength'])
                    table_data.append({
                        'Rank': len(table_data) + 1,
                        'Wavelength (nm)': f"{row['wavelength']:.0f}",
                        'Importance': f"{row['importance']:.4f}",
                        'Importance (%)': f"{row['importance']*100:.2f}%",
                        'Region': interp['name'],
                        'Associated With': interp['associated']
                    })
                
                df_table = pd.DataFrame(table_data)
                st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            # ===== STATISTICS =====
            st.markdown("#### 📊 Importance Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Features", len(importance_dict))
            
            with col2:
                top10_contrib = df_importance.nlargest(10, 'importance')['importance'].sum()
                st.metric("Top 10 Contribution", f"{top10_contrib*100:.1f}%")
            
            with col3:
                top20_contrib = df_importance.nlargest(20, 'importance')['importance'].sum()
                st.metric("Top 20 Contribution", f"{top20_contrib*100:.1f}%")
            
            with col4:
                # Features needed for 50% importance
                sorted_imp = df_importance.sort_values('importance', ascending=False)
                cumsum = sorted_imp['importance'].cumsum()
                n_for_50 = (cumsum < 0.5 * cumsum.iloc[-1]).sum() + 1
                st.metric("Features for 50%", n_for_50)
        
        else:
            st.warning("Could not parse wavelengths from feature names. Showing raw importance values.")
            
            # Show simple bar chart for non-wavelength features
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            fig = go.Figure(go.Bar(
                x=[f[1] for f in sorted_features],
                y=[f[0] for f in sorted_features],
                orientation='h',
                marker_color='#FF6B6B'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} Important Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== COMPARE ACROSS MODELS =====
    st.markdown("---")
    st.markdown("### 🔀 Compare Feature Importance Across Models")
    
    models_with_fi_list = [r for r in results if r.get('feature_importance')]
    
    if len(models_with_fi_list) >= 2:
        compare_models = st.multiselect(
            "Select models to compare:",
            options=[r.get('model_name') for r in models_with_fi_list],
            default=[models_with_fi_list[0].get('model_name'), models_with_fi_list[1].get('model_name')]
        )
        
        if len(compare_models) >= 2:
            # Create heatmap of top features across models
            st.markdown("#### 🔥 Importance Heatmap")
            
            # Get top features from each model
            all_top_features = set()
            for model_name in compare_models:
                model_result = next((r for r in models_with_fi_list if r.get('model_name') == model_name), None)
                if model_result:
                    sorted_features = sorted(model_result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                    top_features = [f[0] for f in sorted_features[:15]]
                    all_top_features.update(top_features)
            
            # Create heatmap data
            heatmap_data = []
            for feature in sorted(all_top_features):
                row = {'Feature': feature}
                for model_name in compare_models:
                    model_result = next((r for r in models_with_fi_list if r.get('model_name') == model_name), None)
                    if model_result:
                        row[model_name] = model_result['feature_importance'].get(feature, 0)
                heatmap_data.append(row)
            
            df_heatmap = pd.DataFrame(heatmap_data)
            df_heatmap = df_heatmap.set_index('Feature')
            
            # Sort by average importance
            df_heatmap['avg'] = df_heatmap.mean(axis=1)
            df_heatmap = df_heatmap.sort_values('avg', ascending=True)
            df_heatmap = df_heatmap.drop('avg', axis=1)
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=df_heatmap.values,
                x=df_heatmap.columns,
                y=df_heatmap.index,
                colorscale='RdYlGn',
                text=[[f"{v:.3f}" for v in row] for row in df_heatmap.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Feature: %{y}<br>Model: %{x}<br>Importance: %{z:.4f}<extra></extra>"
            ))
            
            fig_heatmap.update_layout(
                title="Feature Importance Comparison Across Models",
                xaxis_title="Model",
                yaxis_title="Feature",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Common important features
            st.markdown("#### 🎯 Consensus Important Features")
            st.markdown("Features that appear in top 15 for multiple models:")
            
            feature_counts = {}
            for model_name in compare_models:
                model_result = next((r for r in models_with_fi_list if r.get('model_name') == model_name), None)
                if model_result:
                    sorted_features = sorted(model_result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                    for f, _ in sorted_features[:15]:
                        feature_counts[f] = feature_counts.get(f, 0) + 1
            
            common_features = [(f, c) for f, c in feature_counts.items() if c >= 2]
            common_features = sorted(common_features, key=lambda x: x[1], reverse=True)
            
            if common_features:
                for feature, count in common_features[:10]:
                    try:
                        wl = float(''.join(c for c in feature if c.isdigit() or c == '.'))
                        interp = get_wavelength_interpretation(wl)
                        st.markdown(f"- **{feature}** - Appears in {count}/{len(compare_models)} models ({interp['name']})")
                    except:
                        st.markdown(f"- **{feature}** - Appears in {count}/{len(compare_models)} models")
    else:
        st.info("Train more models to compare feature importance across different algorithms.")


def render_pca_analysis():
    """Render PCA analysis."""
    
    st.markdown("### 📊 Principal Component Analysis")
    
    if st.session_state.get('data') is None:
        st.warning("⚠️ No data available. Please upload data first.")
        return
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    df = st.session_state.data
    target_col = st.session_state.get('target_col', '')
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    
    if X.shape[1] < 3:
        st.warning("Not enough features for PCA analysis.")
        return
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    n_components = min(10, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    st.markdown("#### 📈 Explained Variance")
    
    fig_var = go.Figure()
    
    fig_var.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=pca.explained_variance_ratio_ * 100,
        name='Individual',
        marker_color='#FF6B6B'
    ))
    
    fig_var.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=np.cumsum(pca.explained_variance_ratio_) * 100,
        name='Cumulative',
        mode='lines+markers',
        marker_color='#4CAF50',
        line=dict(width=2)
    ))
    
    fig_var.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig_var, use_container_width=True)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("PC1 Variance", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
    
    with col2:
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_for_90 = np.searchsorted(cumsum, 0.9) + 1
        st.metric("PCs for 90%", n_for_90)
    
    with col3:
        st.metric("Total Variance (10 PCs)", f"{cumsum[-1]*100:.1f}%")
    
    # 2D Scatter
    st.markdown("#### 🎯 Sample Distribution (PC1 vs PC2)")
    
    if target_col and target_col in df.columns:
        y = df[target_col].values
        
        fig_scatter = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=y,
            color_continuous_scale='RdYlGn',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   'color': target_col}
        )
        
        fig_scatter.update_layout(
            title=f"Samples in PC Space (colored by {target_col})",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_residual_diagnostics(results: list, visualizer: Visualizer, evaluator: ModelEvaluator):
    """Render residual diagnostics."""
    
    st.markdown("### 📉 Residual Diagnostics")
    
    # Model selector
    model_names = [r.get('model_name', 'Unknown') for r in results]
    selected_model = st.selectbox("Select model for residual analysis:", options=model_names, key="residual_model")
    
    if selected_model:
        model_result = next((r for r in results if r.get('model_name') == selected_model), None)
        
        if model_result:
            # Get predictions - try multiple possible keys
            y_test = model_result.get('y_test') or model_result.get('y_pred_test')
            y_pred = model_result.get('y_pred_test') or model_result.get('y_pred')
            
            if y_test is None or y_pred is None:
                st.warning("⚠️ Prediction data not available for this model. This can happen when loading results from a previous session.")
                st.info("💡 Try running a new training to generate full residual diagnostics.")
                return
            
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            
            if len(y_test) == 0 or len(y_pred) == 0:
                st.warning("⚠️ Empty prediction data.")
                return
            
            residuals = y_test - y_pred
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Predicted vs Actual",
                    "Residuals vs Predicted",
                    "Residual Distribution",
                    "Q-Q Plot"
                )
            )
            
            # 1. Predicted vs Actual
            fig.add_trace(
                go.Scatter(
                    x=y_test, y=y_pred,
                    mode='markers',
                    marker=dict(color='#FF6B6B', size=8, opacity=0.7),
                    name='Predictions'
                ),
                row=1, col=1
            )
            
            # Add 1:1 line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='white', dash='dash'),
                    name='1:1 Line'
                ),
                row=1, col=1
            )
            
            # 2. Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=y_pred, y=residuals,
                    mode='markers',
                    marker=dict(color='#4CAF50', size=8, opacity=0.7),
                    name='Residuals'
                ),
                row=1, col=2
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)
            
            # 3. Residual Distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    marker_color='#FFA500',
                    name='Distribution'
                ),
                row=2, col=1
            )
            
            # 4. Q-Q Plot
            from scipy import stats
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles, y=sorted_residuals,
                    mode='markers',
                    marker=dict(color='#9C27B0', size=8, opacity=0.7),
                    name='Q-Q'
                ),
                row=2, col=2
            )
            
            # Add diagonal line for Q-Q
            fig.add_trace(
                go.Scatter(
                    x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                    y=[sorted_residuals.min(), sorted_residuals.max()],
                    mode='lines',
                    line=dict(color='white', dash='dash'),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"Residual Diagnostics - {selected_model}",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=700,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual statistics
            st.markdown("#### 📊 Residual Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Residual", f"{residuals.mean():.4f}")
            
            with col2:
                st.metric("Std Residual", f"{residuals.std():.4f}")
            
            with col3:
                st.metric("Min Residual", f"{residuals.min():.4f}")
            
            with col4:
                st.metric("Max Residual", f"{residuals.max():.4f}")
            
            # Normality test
            st.markdown("#### 🧪 Normality Test")
            
            stat, p_value = stats.shapiro(residuals[:min(5000, len(residuals))])
            
            if p_value > 0.05:
                st.success(f"✅ Residuals appear normally distributed (Shapiro-Wilk p={p_value:.4f})")
            else:
                st.warning(f"⚠️ Residuals may not be normally distributed (Shapiro-Wilk p={p_value:.4f})")


def render_model_inspection(results: list, evaluator: ModelEvaluator):
    """Render model inspection section."""
    
    st.markdown("### 🔍 Model Inspection")
    
    # Model selector
    model_names = [r.get('model_name', 'Unknown') for r in results]
    selected_model = st.selectbox("Select model to inspect:", options=model_names, key="inspect_model")
    
    if selected_model:
        model_result = next((r for r in results if r.get('model_name') == selected_model), None)
        
        if model_result:
            # Model info
            st.markdown("#### 📋 Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Model Type:** {model_result.get('model_type', 'N/A').upper()}
                
                **Preprocessing:** {model_result.get('preprocessing', 'N/A').title()}
                
                **Training Time:** {model_result.get('train_time', 0):.2f}s
                
                **Training Samples:** {model_result.get('n_train', 'N/A')}
                
                **Test Samples:** {model_result.get('n_test', 'N/A')}
                """)
            
            with col2:
                st.markdown("**Performance Metrics:**")
                
                metrics_data = {
                    'Metric': ['R² (Test)', 'RMSE', 'MAE', 'RPD', 'Correlation', 'CV Mean'],
                    'Value': [
                        f"{model_result.get('test_r2', 0):.4f}",
                        f"{model_result.get('test_rmse', 0):.4f}",
                        f"{model_result.get('test_mae', 0):.4f}",
                        f"{model_result.get('rpd', 0):.2f}",
                        f"{model_result.get('correlation', 0):.4f}",
                        f"{model_result.get('cv_mean', 0):.4f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
            
            # CV Scores distribution
            cv_scores = model_result.get('cv_scores', [])
            if cv_scores:
                st.markdown("#### 📊 Cross-Validation Scores")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                    y=cv_scores,
                    marker_color='#FF6B6B',
                    text=[f"{s:.3f}" for s in cv_scores],
                    textposition='outside'
                ))
                
                fig.add_hline(
                    y=np.mean(cv_scores),
                    line_dash="dash",
                    line_color="#4CAF50",
                    annotation_text=f"Mean: {np.mean(cv_scores):.4f}"
                )
                
                fig.update_layout(
                    title="Cross-Validation R² Scores",
                    xaxis_title="Fold",
                    yaxis_title="R² Score",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
