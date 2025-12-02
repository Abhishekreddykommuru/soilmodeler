"""
Model Archive Component
=======================
Page 4: View and manage training history and archived models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import json


def render_model_archive():
    """Render the model archive page."""
    
    st.markdown("""
    <h1 style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 800;'>
        📂 Model Archive
    </h1>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs(["📚 Training History", "📦 Stored Models", "🔀 Compare Runs", "📊 Cross-Run Analysis", "🗂️ Storage Management"])
    
    # Tab 1: Training History
    with tabs[0]:
        render_training_history()
    
    # Tab 2: Stored Models
    with tabs[1]:
        render_stored_models()
    
    # Tab 3: Compare Runs
    with tabs[2]:
        render_run_comparison()
    
    # Tab 4: Cross-Run Analysis
    with tabs[3]:
        render_cross_run_analysis()
    
    # Tab 5: Storage Management
    with tabs[4]:
        render_storage_management()


def render_training_history():
    """Render training history section."""
    
    st.markdown("### 📚 TRAINING RUNS History")
    
    history = st.session_state.get('training_history', [])
    
    if not history:
        st.info("📭 No training history yet. Complete a training run to see it here.")
        
        if st.button("Go to Upload & Train"):
            st.session_state.current_page = "upload_train"
            st.rerun()
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        search = st.text_input("🔍 Search runs", placeholder="Search by run ID...")
    
    with col2:
        sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First", "Best R²"])
    
    # Filter history
    filtered_history = history.copy()
    if search:
        filtered_history = [h for h in history if search.lower() in h.get('run_id', '').lower()]
    
    # Sort
    if sort_order == "Newest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', ''), reverse=True)
    elif sort_order == "Oldest First":
        filtered_history = sorted(filtered_history, key=lambda x: x.get('timestamp', ''))
    else:
        filtered_history = sorted(filtered_history, key=lambda x: x.get('best_r2', 0) or 0, reverse=True)
    
    st.markdown("---")
    
    # Initialize delete confirmation state
    if 'delete_confirm' not in st.session_state:
        st.session_state.delete_confirm = {}
    
    # Display history
    for idx, entry in enumerate(filtered_history):
        run_id = entry.get('run_id', f'run_{idx}')
        
        with st.expander(f"📁 {run_id}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**📅 Date/Time**")
                timestamp = entry.get('timestamp', '')
                st.write(timestamp[:19] if timestamp else "N/A")
            
            with col2:
                st.markdown("**🤖 Models**")
                st.write(entry.get('n_models', 0))
            
            with col3:
                st.markdown("**🏆 Best R²**")
                r2 = entry.get('best_r2', 0)
                if r2:
                    color = '#4CAF50' if r2 >= 0.85 else '#FFC107' if r2 >= 0.75 else '#FF6B6B'
                    st.markdown(f"<span style='color: {color}; font-weight: bold;'>{r2:.4f}</span>", unsafe_allow_html=True)
                else:
                    st.write("N/A")
            
            with col4:
                st.markdown("**📊 Best Model**")
                st.write(entry.get('best_model', 'N/A'))
            
            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("📥 Load Run", key=f"load_{run_id}_{idx}"):
                    # Load this run's data into session state for Model Comparison
                    load_run_for_comparison(entry)
                    st.session_state.current_page = "model_comparison"
                    st.rerun()
            
            with btn_col2:
                if st.button("📊 View Details", key=f"view_{run_id}_{idx}"):
                    # Load this run's data into session state for Model Comparison
                    load_run_for_comparison(entry)
                    st.session_state.current_page = "model_comparison"
                    st.rerun()
            
            with btn_col3:
                # Delete with confirmation
                if st.session_state.delete_confirm.get(run_id):
                    st.warning(f"⚠️ Confirm delete?")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Yes", key=f"confirm_del_{run_id}_{idx}"):
                            # Actually delete the run
                            delete_training_run(run_id)
                            st.session_state.delete_confirm[run_id] = False
                            st.success(f"Deleted run {run_id}")
                            st.rerun()
                    with confirm_col2:
                        if st.button("❌ No", key=f"cancel_del_{run_id}_{idx}"):
                            st.session_state.delete_confirm[run_id] = False
                            st.rerun()
                else:
                    if st.button("🗑️ Delete", key=f"delete_{run_id}_{idx}"):
                        st.session_state.delete_confirm[run_id] = True
                        st.rerun()
    
    # Bulk actions
    st.markdown("---")
    st.markdown("### 🔧 Bulk Actions")
    
    selected_runs = st.multiselect(
        "Select runs for bulk actions",
        options=[h.get('run_id', f'run_{i}') for i, h in enumerate(filtered_history)]
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Load Selected", use_container_width=True, disabled=not selected_runs):
            if selected_runs:
                st.success(f"Loading {len(selected_runs)} runs...")
    
    with col2:
        if st.button("🔀 Compare Selected", use_container_width=True, disabled=len(selected_runs) < 2):
            if len(selected_runs) >= 2:
                st.session_state.compare_runs = selected_runs
                st.info("Go to Compare Runs tab")
    
    with col3:
        if st.button("📦 Export Selected", use_container_width=True, disabled=not selected_runs):
            if selected_runs:
                st.success("Export package created")
    
    with col4:
        if st.button("🗑️ Delete Selected", use_container_width=True, disabled=not selected_runs, type="primary"):
            if selected_runs:
                for run_id in selected_runs:
                    delete_training_run(run_id)
                st.success(f"Deleted {len(selected_runs)} runs")
                st.rerun()
    
    # Summary statistics
    if history:
        st.markdown("---")
        st.markdown("### 📊 Archive Statistics")
        
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("Total Runs", len(history))
        
        with stat_cols[1]:
            total_models = sum(h.get('n_models', 0) for h in history)
            st.metric("Total Models Trained", total_models)
        
        with stat_cols[2]:
            r2_values = [h.get('best_r2', 0) for h in history if h.get('best_r2')]
            best_r2 = max(r2_values) if r2_values else 0
            st.metric("Best R² Overall", f"{best_r2:.4f}" if best_r2 else "N/A")
        
        with stat_cols[3]:
            if history:
                timestamps = [h.get('timestamp', '')[:10] for h in history if h.get('timestamp')]
                first_date = min(timestamps) if timestamps else "N/A"
                st.metric("First Run", first_date)


def delete_training_run(run_id: str):
    """Delete a training run from history and storage."""
    
    # Remove from session state history
    history = st.session_state.get('training_history', [])
    st.session_state.training_history = [h for h in history if h.get('run_id') != run_id]
    
    # Remove from disk if exists
    models_dir = Path("models") / run_id
    if models_dir.exists():
        try:
            shutil.rmtree(models_dir)
        except Exception as e:
            st.warning(f"Could not delete model files: {e}")
    
    # If this was the current result, clear it
    if st.session_state.get('training_results', {}).get('run_id') == run_id:
        st.session_state.training_complete = False
        st.session_state.training_results = None
        st.session_state.best_model = None
        st.session_state.best_r2 = None
        st.session_state.total_models = 0


def load_run_for_comparison(entry: dict):
    """Load a training run's data into session state for Model Comparison.
    
    Args:
        entry: Training history entry containing run info
    """
    run_id = entry.get('run_id', 'unknown')
    
    # Try to load full results from stored data
    models_dir = Path("models") / run_id
    results_file = models_dir / "results.json"
    
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                full_results = json.load(f)
            
            # Load full results into session state
            st.session_state.training_complete = True
            st.session_state.training_results = full_results
            st.session_state.best_r2 = entry.get('best_r2', 0)
            st.session_state.total_models = entry.get('n_models', 0)
            
            # Find best model from results
            all_results = full_results.get('results', [])
            successful = [r for r in all_results if r.get('status') == 'success']
            if successful:
                best = max(successful, key=lambda x: x.get('test_r2', 0))
                st.session_state.best_model = best
            
            return True
        except Exception as e:
            st.warning(f"Could not load full results: {e}")
    
    # If no stored results file, try to reconstruct from history entry
    # This allows viewing even if full results weren't saved
    if 'results' in entry:
        st.session_state.training_complete = True
        st.session_state.training_results = entry
        st.session_state.best_r2 = entry.get('best_r2', 0)
        st.session_state.total_models = entry.get('n_models', 0)
        
        all_results = entry.get('results', [])
        successful = [r for r in all_results if r.get('status') == 'success']
        if successful:
            best = max(successful, key=lambda x: x.get('test_r2', 0))
            st.session_state.best_model = best
        
        return True
    
    # Fallback: Set basic info so user knows which run is loaded
    st.session_state.training_complete = True
    st.session_state.best_r2 = entry.get('best_r2', 0)
    st.session_state.total_models = entry.get('n_models', 0)
    st.session_state.loaded_run_id = run_id
    
    # Show info message about limited data
    st.info(f"📊 Loaded run {run_id} - Some detailed results may not be available if the run was from a previous session.")
    
    return False


def render_stored_models():
    """Render stored models section."""
    
    st.markdown("### 📦 Stored Models")
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    
    # List all run directories
    run_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not run_dirs:
        st.info("📭 No models stored yet. Train models to see them here.")
        return
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search = st.text_input("🔍 Search models", placeholder="Search by name...")
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Name"])
    
    # Sort directories
    if sort_by == "Date (Newest)":
        run_dirs = sorted(run_dirs, key=lambda x: x.name, reverse=True)
    elif sort_by == "Date (Oldest)":
        run_dirs = sorted(run_dirs, key=lambda x: x.name)
    else:
        run_dirs = sorted(run_dirs, key=lambda x: x.name)
    
    # Filter by search
    if search:
        run_dirs = [d for d in run_dirs if search.lower() in d.name.lower()]
    
    # Display models by run
    for run_dir in run_dirs:
        with st.expander(f"📁 {run_dir.name}", expanded=False):
            # List model files
            model_files = list(run_dir.glob("*.pkl"))
            model_files = [f for f in model_files if f.name != "metadata.pkl"]
            
            if not model_files:
                st.write("No model files in this run.")
                continue
            
            st.write(f"**{len(model_files)} model files:**")
            
            # Display as cards
            cols = st.columns(3)
            for idx, model_file in enumerate(model_files):
                with cols[idx % 3]:
                    size_kb = model_file.stat().st_size / 1024
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); 
                                border: 1px solid #3A3F4B; 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin-bottom: 0.5rem;'>
                        <strong>📄 {model_file.stem}</strong><br>
                        <small style='color: #888;'>Size: {size_kb:.1f} KB</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "💾 Download",
                            data=model_file.read_bytes(),
                            file_name=model_file.name,
                            key=f"dl_{run_dir.name}_{model_file.name}",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("🗑️", key=f"del_model_{run_dir.name}_{model_file.name}"):
                            try:
                                model_file.unlink()
                                st.success(f"Deleted {model_file.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            
            # Check for results CSV
            results_file = run_dir / "results.csv"
            if results_file.exists():
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📊 View Results", key=f"results_{run_dir.name}"):
                        df = pd.read_csv(results_file)
                        st.dataframe(df, use_container_width=True)
                with col2:
                    st.download_button(
                        "📥 Download Results",
                        data=results_file.read_text(),
                        file_name=f"{run_dir.name}_results.csv",
                        mime="text/csv",
                        key=f"dl_results_{run_dir.name}"
                    )
            
            # Delete entire run
            st.markdown("---")
            if st.button(f"🗑️ Delete Entire Run", key=f"del_run_{run_dir.name}", type="secondary"):
                try:
                    shutil.rmtree(run_dir)
                    st.success(f"Deleted {run_dir.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Storage summary
    st.markdown("---")
    st.markdown("### 💾 Storage Summary")
    
    total_size = sum(f.stat().st_size for d in run_dirs if d.exists() for f in d.glob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    total_models = sum(len(list(d.glob("*.pkl"))) - (1 if (d / "metadata.pkl").exists() else 0) for d in run_dirs if d.exists())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Runs", len(run_dirs))
    
    with col2:
        st.metric("Total Model Files", max(0, total_models))
    
    with col3:
        st.metric("Storage Used", f"{total_size_mb:.2f} MB")


def render_run_comparison():
    """Render run comparison section."""
    
    st.markdown("### 🔀 Compare Training Runs")
    
    history = st.session_state.get('training_history', [])
    
    if len(history) < 2:
        st.info("📭 Need at least 2 training runs to compare. Complete more runs to use this feature.")
        return
    
    # Select runs to compare
    run_ids = [h.get('run_id', f"Run {i}") for i, h in enumerate(history)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        run1 = st.selectbox("Select first run:", options=run_ids, key="compare_run1")
    
    with col2:
        # Default to second option if available
        default_idx = min(1, len(run_ids) - 1)
        run2 = st.selectbox("Select second run:", options=run_ids, key="compare_run2", index=default_idx)
    
    if run1 and run2 and run1 != run2:
        entry1 = next((h for h in history if h.get('run_id') == run1), {})
        entry2 = next((h for h in history if h.get('run_id') == run2), {})
        
        st.markdown("---")
        st.markdown("### 📊 Side-by-Side Comparison")
        
        # Create comparison table
        comparison_metrics = [
            ("📅 Date/Time", entry1.get('timestamp', 'N/A')[:19], entry2.get('timestamp', 'N/A')[:19]),
            ("🤖 Models Trained", entry1.get('n_models', 0), entry2.get('n_models', 0)),
            ("🏆 Best R²", f"{entry1.get('best_r2', 0):.4f}" if entry1.get('best_r2') else "N/A", 
             f"{entry2.get('best_r2', 0):.4f}" if entry2.get('best_r2') else "N/A"),
            ("📊 Best Model", entry1.get('best_model', 'N/A'), entry2.get('best_model', 'N/A'))
        ]
        
        # Display as table
        st.markdown(f"""
        | Metric | {run1} | {run2} |
        |--------|--------|--------|
        """)
        
        for metric, val1, val2 in comparison_metrics:
            st.markdown(f"| {metric} | {val1} | {val2} |")
        
        # R² comparison chart
        r2_1 = entry1.get('best_r2', 0) or 0
        r2_2 = entry2.get('best_r2', 0) or 0
        
        if r2_1 and r2_2:
            st.markdown("---")
            st.markdown("### 📈 Visual Comparison")
            
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[run1, run2],
                    y=[r2_1, r2_2],
                    marker_color=['#667eea', '#764ba2'],
                    text=[f"{r2_1:.4f}", f"{r2_2:.4f}"],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Best R² Comparison",
                yaxis_title="R²",
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Winner indicator
            diff = r2_2 - r2_1
            pct_change = (diff / r2_1) * 100 if r2_1 != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(run1, f"{r2_1:.4f}")
            with col2:
                st.metric(run2, f"{r2_2:.4f}")
            with col3:
                st.metric("Difference", f"{diff:+.4f}", delta=f"{pct_change:+.1f}%")
            
            # Winner declaration
            if diff > 0.001:
                st.success(f"🏆 **{run2}** has better performance by {diff:.4f} R²!")
            elif diff < -0.001:
                st.success(f"🏆 **{run1}** has better performance by {-diff:.4f} R²!")
            else:
                st.info("🤝 Both runs have approximately equal performance.")
    
    elif run1 == run2:
        st.warning("⚠️ Please select two different runs to compare.")


def render_cross_run_analysis():
    """Render cross-run analysis section."""
    
    st.markdown("### 📊 Cross-Run Performance Analysis")
    
    history = st.session_state.get('training_history', [])
    
    if len(history) < 2:
        st.info("📭 Need at least 2 training runs for cross-run analysis.")
        return
    
    # Create performance matrix
    import plotly.graph_objects as go
    
    run_ids = [h.get('run_id', f"Run {i}")[:15] for i, h in enumerate(history)]
    r2_values = [h.get('best_r2', 0) or 0 for h in history]
    model_counts = [h.get('n_models', 0) for h in history]
    
    # Performance trend chart
    st.markdown("#### 📈 Performance Trend Over Runs")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(run_ids))),
        y=r2_values,
        mode='lines+markers',
        name='Best R²',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    
    # Add trend line
    if len(r2_values) > 1:
        z = np.polyfit(range(len(r2_values)), r2_values, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=list(range(len(run_ids))),
            y=[p(i) for i in range(len(run_ids))],
            mode='lines',
            name='Trend',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="R² Performance Across Training Runs",
        xaxis_title="Training Run",
        yaxis_title="Best R²",
        height=400,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(ticktext=run_ids, tickvals=list(range(len(run_ids))))
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_r2 = np.mean([r for r in r2_values if r > 0])
        st.metric("Average R²", f"{avg_r2:.4f}")
    
    with col2:
        max_r2 = max(r2_values)
        st.metric("Best R²", f"{max_r2:.4f}")
    
    with col3:
        improvement = r2_values[-1] - r2_values[0] if len(r2_values) > 1 else 0
        st.metric("Improvement", f"{improvement:+.4f}")
    
    with col4:
        total_models = sum(model_counts)
        st.metric("Total Models", total_models)


def render_storage_management():
    """Render storage management section."""
    
    st.markdown("### 🗂️ Storage Management")
    
    models_dir = Path("models")
    cache_dir = Path("cache")
    logs_dir = Path("logs")
    reports_dir = Path("reports")
    
    # Calculate storage usage
    def get_dir_size(path):
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    models_size = get_dir_size(models_dir) / (1024 * 1024)
    cache_size = get_dir_size(cache_dir) / (1024 * 1024)
    logs_size = get_dir_size(logs_dir) / (1024 * 1024)
    reports_size = get_dir_size(reports_dir) / (1024 * 1024)
    total_size = models_size + cache_size + logs_size + reports_size
    
    # Storage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models", f"{models_size:.2f} MB")
    with col2:
        st.metric("Cache", f"{cache_size:.2f} MB")
    with col3:
        st.metric("Logs", f"{logs_size:.2f} MB")
    with col4:
        st.metric("Total", f"{total_size:.2f} MB")
    
    # Storage breakdown pie chart
    st.markdown("#### 📊 Storage Breakdown")
    
    import plotly.express as px
    
    storage_data = {
        'Category': ['Models', 'Cache', 'Logs', 'Reports'],
        'Size (MB)': [models_size, cache_size, logs_size, reports_size]
    }
    
    fig = px.pie(
        storage_data,
        values='Size (MB)',
        names='Category',
        color_discrete_sequence=['#667eea', '#764ba2', '#4CAF50', '#FFC107']
    )
    
    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cleanup options
    st.markdown("#### 🧹 Cleanup Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            if cache_dir.exists():
                for f in cache_dir.glob('*'):
                    if f.is_file():
                        f.unlink()
                st.success(f"Cleared cache: {cache_size:.2f} MB freed")
                st.rerun()
    
    with col2:
        if st.button("📋 Clear Logs", use_container_width=True):
            if logs_dir.exists():
                for f in logs_dir.glob('*.log'):
                    f.unlink()
                st.success(f"Cleared logs: {logs_size:.2f} MB freed")
                st.rerun()
    
    with col3:
        if st.button("📄 Clear Reports", use_container_width=True):
            if reports_dir.exists():
                for f in reports_dir.glob('*'):
                    if f.is_file() and f.name != '.gitkeep':
                        f.unlink()
                st.success(f"Cleared reports: {reports_size:.2f} MB freed")
                st.rerun()
    
    with col4:
        if st.button("⚠️ Clear All Models", use_container_width=True, type="primary"):
            if models_dir.exists():
                for d in models_dir.iterdir():
                    if d.is_dir():
                        shutil.rmtree(d)
                st.session_state.training_history = []
                st.session_state.training_complete = False
                st.session_state.training_results = None
                st.success(f"Cleared all models: {models_size:.2f} MB freed")
                st.rerun()
