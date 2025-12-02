"""
Model Comparison Component
==========================
Page 3: Comprehensive model comparison with leaderboard, head-to-head, and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from modules.visualization import Visualizer
from modules.evaluation import ModelEvaluator
from modules.utils import get_performance_color, get_medal, export_results_to_excel


def render_model_comparison():
    """Render the model comparison page."""
    
    st.markdown("""
    <h1 style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 800;'>
        📊 Model Comparison & Leaderboard
    </h1>
    """, unsafe_allow_html=True)
    
    # Check if results exist
    if not st.session_state.get('training_complete'):
        st.warning("⚠️ No training results available. Complete a training run first.")
        if st.button("Go to Upload & Train", type="primary"):
            st.session_state.current_page = "upload_train"
            st.rerun()
        return
    
    results = st.session_state.get('training_results', {})
    all_results = results.get('results', [])
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        st.error("No successful models found.")
        return
    
    # Show currently loaded run info
    run_id = results.get('run_id', st.session_state.get('loaded_run_id', 'Current Session'))
    timestamp = results.get('timestamp', '')[:19] if results.get('timestamp') else ''
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; 
                display: inline-block;">
        <span style="color: white; font-weight: 500;">
            📁 Loaded Run: <strong>{run_id}</strong>
            {f' | 🕐 {timestamp}' if timestamp else ''}
            | 🤖 {len(successful_results)} models
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualizer
    visualizer = Visualizer()
    evaluator = ModelEvaluator()
    
    # Create tabs
    tabs = st.tabs(["🏆 Leaderboard", "🎯 Head-to-Head", "📈 Visualizations", "📉 Residuals", "📊 Ensemble Analysis"])
    
    # Tab 1: Leaderboard
    with tabs[0]:
        render_leaderboard(successful_results, visualizer)
    
    # Tab 2: Head-to-Head
    with tabs[1]:
        render_head_to_head(successful_results)
    
    # Tab 3: Visualizations
    with tabs[2]:
        render_visualizations(successful_results, visualizer)
    
    # Tab 4: Residuals
    with tabs[3]:
        render_residual_analysis(successful_results, visualizer, evaluator)
    
    # Tab 5: Ensemble Analysis
    with tabs[4]:
        render_ensemble_analysis(successful_results)


def render_leaderboard(results: list, visualizer: Visualizer):
    """Render the model leaderboard."""
    
    st.markdown("### 🏆 Model Leaderboard")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["test_r2", "test_rmse", "rpd", "test_mae"],
            format_func=lambda x: {"test_r2": "R²", "test_rmse": "RMSE", "rpd": "RPD", "test_mae": "MAE"}[x]
        )
    
    with col2:
        filter_prep = st.multiselect(
            "Filter by Preprocessing",
            options=list(set(r['preprocessing'] for r in results)),
            default=list(set(r['preprocessing'] for r in results))
        )
    
    with col3:
        filter_model = st.multiselect(
            "Filter by Model",
            options=list(set(r['model_type'] for r in results)),
            default=list(set(r['model_type'] for r in results))
        )
    
    # Filter and sort results
    filtered_results = [
        r for r in results 
        if r['preprocessing'] in filter_prep and r['model_type'] in filter_model
    ]
    
    ascending = sort_by in ["test_rmse", "test_mae"]  # Lower is better
    sorted_results = sorted(filtered_results, key=lambda x: x.get(sort_by, 0), reverse=not ascending)
    
    # Create leaderboard table
    st.markdown("---")
    
    # Display as styled dataframe
    leaderboard_data = []
    for i, r in enumerate(sorted_results, 1):
        leaderboard_data.append({
            'Rank': get_medal(i),
            'Model': r.get('model_name', 'N/A'),
            'Preprocessing': r.get('preprocessing', 'N/A').title(),
            'Algorithm': r.get('model_type', 'N/A').upper(),
            'R²': r.get('test_r2', 0),
            'RMSE': r.get('test_rmse', 0),
            'RPD': r.get('rpd', 0),
            'MAE': r.get('test_mae', 0),
            'CV R²': f"{r.get('cv_mean', 0):.3f}±{r.get('cv_std', 0):.3f}"
        })
    
    df = pd.DataFrame(leaderboard_data)
    
    # Style the dataframe
    def color_r2(val):
        if isinstance(val, float):
            if val >= 0.85:
                return 'background-color: rgba(76, 175, 80, 0.3)'
            elif val >= 0.75:
                return 'background-color: rgba(255, 193, 7, 0.3)'
            else:
                return 'background-color: rgba(244, 67, 54, 0.3)'
        return ''
    
    styled_df = df.style.applymap(color_r2, subset=['R²']).format({
        'R²': '{:.4f}',
        'RMSE': '{:.4f}',
        'RPD': '{:.2f}',
        'MAE': '{:.4f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Best performer summary
    if sorted_results:
        best = sorted_results[0]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%); 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: white; margin: 0;">🏆 Best by {sort_by.replace('test_', '').upper()}</h4>
            <h3 style="color: white; margin: 0.5rem 0;">{best.get('model_name', 'N/A')}</h3>
            <p style="color: rgba(255,255,255,0.9);">
                R² = {best.get('test_r2', 0):.4f} | RMSE = {best.get('test_rmse', 0):.4f} | RPD = {best.get('rpd', 0):.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export options
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        excel_data = export_results_to_excel(sorted_results)
        st.download_button(
            label="📥 Export to Excel",
            data=excel_data,
            file_name="model_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Export to CSV",
            data=csv,
            file_name="model_results.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("📊 Go to Analytics"):
            st.session_state.current_page = "analytics_export"
            st.rerun()


def render_head_to_head(results: list):
    """Render head-to-head model comparison."""
    
    st.markdown("### 🎯 Head-to-Head Model Comparison")
    
    st.info("Select two models to compare them across all metrics.")
    
    # Model selection
    model_names = [r.get('model_name', 'Unknown') for r in results]
    
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = st.selectbox("🔵 Model 1:", options=model_names, key="h2h_model1")
    
    with col2:
        # Default to second model if available
        default_idx = min(1, len(model_names) - 1)
        model2 = st.selectbox("🟣 Model 2:", options=model_names, key="h2h_model2", index=default_idx)
    
    if model1 and model2 and model1 != model2:
        # Get model data
        model1_data = next((r for r in results if r.get('model_name') == model1), {})
        model2_data = next((r for r in results if r.get('model_name') == model2), {})
        
        st.markdown("---")
        
        # Comparison metrics table
        st.markdown("### 📊 Metrics Comparison")
        
        metrics = [
            ('R² (Test)', 'test_r2', True),  # Higher is better
            ('RMSE', 'test_rmse', False),    # Lower is better
            ('RPD', 'rpd', True),
            ('MAE', 'test_mae', False),
            ('Correlation', 'correlation', True),
            ('Bias', 'bias', False),  # Closer to 0 is better
            ('CV Mean', 'cv_mean', True),
            ('CV Std', 'cv_std', False),
            ('Train Time (s)', 'train_time', False)
        ]
        
        comparison_data = []
        model1_wins = 0
        model2_wins = 0
        
        for metric_name, metric_key, higher_better in metrics:
            val1 = model1_data.get(metric_key, 0) or 0
            val2 = model2_data.get(metric_key, 0) or 0
            
            # Determine winner
            if metric_key == 'bias':
                # For bias, closer to 0 is better
                better = model1 if abs(val1) < abs(val2) else model2
            elif higher_better:
                better = model1 if val1 > val2 else model2
            else:
                better = model1 if val1 < val2 else model2
            
            if better == model1:
                model1_wins += 1
            else:
                model2_wins += 1
            
            comparison_data.append({
                'Metric': metric_name,
                model1: f"{val1:.4f}" if isinstance(val1, float) else val1,
                model2: f"{val2:.4f}" if isinstance(val2, float) else val2,
                'Difference': f"{val1 - val2:+.4f}",
                'Winner': '🔵' if better == model1 else '🟣'
            })
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Radar chart comparison
        st.markdown("### 📈 Visual Comparison")
        
        # Normalize metrics for radar chart
        radar_metrics = ['R²', 'RPD', 'Correlation']
        
        # Get normalized values
        all_r2 = [r.get('test_r2', 0) for r in results]
        all_rpd = [r.get('rpd', 0) for r in results]
        all_corr = [r.get('correlation', 0) for r in results]
        all_rmse = [r.get('test_rmse', 0) for r in results]
        
        model1_values = [
            model1_data.get('test_r2', 0),
            model1_data.get('rpd', 0) / max(all_rpd) if max(all_rpd) > 0 else 0,
            model1_data.get('correlation', 0),
            1 - (model1_data.get('test_rmse', 0) / max(all_rmse)) if max(all_rmse) > 0 else 0
        ]
        
        model2_values = [
            model2_data.get('test_r2', 0),
            model2_data.get('rpd', 0) / max(all_rpd) if max(all_rpd) > 0 else 0,
            model2_data.get('correlation', 0),
            1 - (model2_data.get('test_rmse', 0) / max(all_rmse)) if max(all_rmse) > 0 else 0
        ]
        
        categories = ['R²', 'RPD (norm)', 'Correlation', 'RMSE (inv)']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=model1_values + [model1_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=model1,
            line=dict(color='#667eea', width=2),
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=model2_values + [model2_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=model2,
            line=dict(color='#764ba2', width=2),
            fillcolor='rgba(118, 75, 162, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            title="Head-to-Head Radar Comparison",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Winner declaration
        st.markdown("### 🏆 Overall Winner")
        
        if model1_wins > model2_wins:
            winner = model1
            wins = model1_wins
            color = "#667eea"
        elif model2_wins > model1_wins:
            winner = model2
            wins = model2_wins
            color = "#764ba2"
        else:
            winner = "Tie"
            wins = model1_wins
            color = "#FFC107"
        
        st.markdown(f"""
        <div style='background: {color}40;
                    border: 2px solid {color};
                    padding: 1.5rem;
                    border-radius: 12px;
                    text-align: center;'>
            <h2 style='color: {color}; margin: 0;'>
                🏆 {winner}
            </h2>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
                Wins in {wins} out of {len(metrics)} metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Side-by-side bar chart
        st.markdown("### 📊 Bar Chart Comparison")
        
        bar_metrics = ['R²', 'RPD', 'Correlation']
        bar_vals_1 = [model1_data.get('test_r2', 0), model1_data.get('rpd', 0), model1_data.get('correlation', 0)]
        bar_vals_2 = [model2_data.get('test_r2', 0), model2_data.get('rpd', 0), model2_data.get('correlation', 0)]
        
        fig = go.Figure(data=[
            go.Bar(name=model1, x=bar_metrics, y=bar_vals_1, marker_color='#667eea'),
            go.Bar(name=model2, x=bar_metrics, y=bar_vals_2, marker_color='#764ba2')
        ])
        
        fig.update_layout(
            barmode='group',
            title="Key Metrics Comparison",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif model1 == model2:
        st.warning("⚠️ Please select two different models to compare.")


def render_visualizations(results: list, visualizer: Visualizer):
    """Render visualization charts."""
    
    st.markdown("### 📈 Performance Visualizations")
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Visualization Type",
        options=["Bar Chart", "Grouped Bars", "Heatmap", "Radar Chart", "Box Plot"]
    )
    
    # Metric selector
    metric = st.selectbox(
        "Metric",
        options=["test_r2", "test_rmse", "rpd", "test_mae"],
        format_func=lambda x: {"test_r2": "R²", "test_rmse": "RMSE", "rpd": "RPD", "test_mae": "MAE"}[x]
    )
    
    st.markdown("---")
    
    # Generate visualization
    if viz_type == "Bar Chart":
        fig = visualizer.performance_bar_chart(results, metric, f"Model Performance - {metric.upper()}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Grouped Bars":
        group_by = st.radio("Group by:", ["preprocessing", "model_type"], horizontal=True)
        fig = visualizer.grouped_bar_chart(results, metric, group_by)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Heatmap":
        fig = visualizer.performance_heatmap(results, metric)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Radar Chart":
        fig = visualizer.radar_chart(results[:5], ['test_r2', 'rpd', 'correlation'])
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        fig = visualizer.cv_scores_boxplot(results)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot section
    st.markdown("---")
    st.markdown("### 📊 Predicted vs Actual Values")
    
    model_names = [r.get('model_name', 'Unknown') for r in results]
    selected_model = st.selectbox("Select model:", options=model_names, key="scatter_model")
    
    if selected_model:
        model_result = next((r for r in results if r.get('model_name') == selected_model), None)
        
        if model_result and 'y_test' in model_result and 'y_pred_test' in model_result:
            fig = visualizer.scatter_prediction(
                model_result['y_test'],
                model_result['y_pred_test'],
                f"Predictions: {selected_model}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{model_result.get('test_r2', 0):.4f}")
            with col2:
                st.metric("RMSE", f"{model_result.get('test_rmse', 0):.4f}")
            with col3:
                st.metric("RPD", f"{model_result.get('rpd', 0):.2f}")
            with col4:
                st.metric("Bias", f"{model_result.get('bias', 0):.4f}")


def render_residual_analysis(results: list, visualizer: Visualizer, evaluator: ModelEvaluator):
    """Render residual analysis."""
    
    st.markdown("### 📉 Residual Analysis")
    
    # Model selector
    model_names = [r.get('model_name', 'Unknown') for r in results]
    selected_model = st.selectbox("Select model for residual analysis:", options=model_names, key="residual_model")
    
    if selected_model:
        model_result = next((r for r in results if r.get('model_name') == selected_model), None)
        
        if model_result:
            # Try to get y_test and y_pred from multiple possible keys
            y_test = model_result.get('y_test') or model_result.get('y_true')
            y_pred = model_result.get('y_pred_test') or model_result.get('y_pred')
            
            if y_test is None or y_pred is None:
                st.warning("⚠️ Prediction data not available for this model.")
                st.info("💡 This can happen when loading results from a previous session. Try running a new training.")
                return
            
            y_true = np.array(y_test)
            y_pred = np.array(y_pred)
            
            if len(y_true) == 0 or len(y_pred) == 0:
                st.warning("⚠️ Empty prediction data.")
                return
            
            # Residual diagnostics
            diagnostics = evaluator.calculate_residual_diagnostics(y_true, y_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals vs Predicted
                fig = visualizer.residual_plot(y_true, y_pred, "Residuals vs Predicted Values")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residual histogram
                fig = visualizer.residual_histogram(y_true, y_pred, "Residual Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Q-Q Plot
            st.markdown("#### Q-Q Plot")
            fig = visualizer.qq_plot(y_true, y_pred, "Normal Q-Q Plot")
            st.plotly_chart(fig, use_container_width=True)
            
            # Overall assessment
            quality = diagnostics.get('overall_quality', 'Unknown')
            quality_colors = {
                'Excellent': '#4CAF50',
                'Good': '#8BC34A',
                'Fair': '#FFC107',
                'Poor': '#FF6B6B'
            }
            
            st.markdown(f"""
            <div style="background: {quality_colors.get(quality, '#4A90E2')}20; 
                        border: 2px solid {quality_colors.get(quality, '#4A90E2')}; 
                        border-radius: 12px; padding: 1.5rem; text-align: center;">
                <h2 style="margin: 0; color: {quality_colors.get(quality, '#4A90E2')};">
                    Model Quality: {quality}
                </h2>
            </div>
            """, unsafe_allow_html=True)


def render_ensemble_analysis(results: list):
    """Render ensemble model analysis."""
    
    st.markdown("### 📊 Ensemble Model Analysis")
    
    st.info("""
    🎯 **Ensemble Learning**: Combine multiple models to potentially achieve better performance.
    """)
    
    # Select models for ensemble
    model_names = [r.get('model_name', 'Unknown') for r in results]
    top_models = sorted(results, key=lambda x: x.get('test_r2', 0), reverse=True)[:3]
    default_selection = [m.get('model_name') for m in top_models]
    
    selected_models = st.multiselect(
        "Select models to include in ensemble (minimum 2):",
        options=model_names,
        default=default_selection
    )
    
    if len(selected_models) >= 2:
        # Ensemble method
        ensemble_method = st.selectbox(
            "Ensemble Method",
            options=[
                "Simple Average",
                "Weighted Average (by R²)",
                "Weighted Average (by RPD)"
            ]
        )
        
        # Get selected models data
        ensemble_results = [r for r in results if r.get('model_name') in selected_models]
        
        # Calculate ensemble performance
        r2_values = [r.get('test_r2', 0) for r in ensemble_results]
        rmse_values = [r.get('test_rmse', 0) for r in ensemble_results]
        rpd_values = [r.get('rpd', 0) for r in ensemble_results]
        
        if ensemble_method == "Simple Average":
            weights = [1/len(selected_models)] * len(selected_models)
            ensemble_r2 = np.mean(r2_values)
            ensemble_rmse = np.mean(rmse_values)
            ensemble_rpd = np.mean(rpd_values)
            
        elif ensemble_method == "Weighted Average (by R²)":
            weights = np.array(r2_values) / sum(r2_values)
            ensemble_r2 = np.average(r2_values, weights=weights)
            ensemble_rmse = np.average(rmse_values, weights=weights)
            ensemble_rpd = np.average(rpd_values, weights=weights)
            
        else:  # Weighted by RPD
            weights = np.array(rpd_values) / sum(rpd_values)
            ensemble_r2 = np.average(r2_values, weights=weights)
            ensemble_rmse = np.average(rmse_values, weights=weights)
            ensemble_rpd = np.average(rpd_values, weights=weights)
        
        st.markdown("---")
        st.markdown("### 📈 Ensemble Performance")
        
        col1, col2, col3 = st.columns(3)
        
        best_r2 = max(r2_values)
        best_rmse = min(rmse_values)
        best_rpd = max(rpd_values)
        
        with col1:
            delta = ensemble_r2 - best_r2
            st.metric("Ensemble R²", f"{ensemble_r2:.4f}", f"{delta:+.4f} vs best")
        
        with col2:
            delta = ensemble_rmse - best_rmse
            st.metric("Ensemble RMSE", f"{ensemble_rmse:.4f}", f"{delta:+.4f} vs best")
        
        with col3:
            delta = ensemble_rpd - best_rpd
            st.metric("Ensemble RPD", f"{ensemble_rpd:.2f}", f"{delta:+.2f} vs best")
        
        # Model weights visualization
        st.markdown("#### Model Weights")
        
        fig = go.Figure(data=[
            go.Bar(
                x=selected_models,
                y=weights,
                marker_color='#667eea',
                text=[f"{w:.3f}" for w in weights],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Model Weights - {ensemble_method}",
            xaxis_title="Model",
            yaxis_title="Weight",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        if ensemble_r2 > best_r2:
            st.success("✅ **Recommendation**: The ensemble shows improved R² performance. Consider using this ensemble for predictions.")
        else:
            st.warning("⚠️ **Note**: The ensemble does not significantly improve over the best individual model.")
    
    else:
        st.warning("Please select at least 2 models to create an ensemble.")
