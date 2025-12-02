"""
Analytics & Export Component
============================
Page 6: Generate reports, export results, and download models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import json
import zipfile
import joblib
from modules.utils import export_results_to_excel, get_medal, get_performance_color


def render_analytics_export():
    """Render the analytics and export page."""
    
    st.markdown("""
    <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 800;'>
        📈 Analytics & Export
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
    
    # Create tabs
    tabs = st.tabs(["📊 Summary Report", "📥 Export Data", "💾 Download Models", "📄 Generate PDF"])
    
    # Tab 1: Summary Report
    with tabs[0]:
        render_summary_report(results, successful_results)
    
    # Tab 2: Export Data
    with tabs[1]:
        render_export_data(successful_results)
    
    # Tab 3: Download Models
    with tabs[2]:
        render_download_models(successful_results)
    
    # Tab 4: Generate PDF
    with tabs[3]:
        render_pdf_report(results, successful_results)


def render_summary_report(results: dict, successful_results: list):
    """Render summary report."""
    
    st.markdown("### 📊 Training Summary Report")
    
    # Overview section
    st.markdown("#### 📋 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", results.get('total_models', 0))
    
    with col2:
        st.metric("Successful", results.get('successful_models', 0))
    
    with col3:
        st.metric("Failed", results.get('failed_models', 0))
    
    with col4:
        st.metric("Training Time", f"{results.get('total_time', 0):.1f}s")
    
    st.markdown("---")
    
    # Best model highlight
    best = results.get('best_model', {})
    if best:
        st.markdown("#### 🏆 Best Model")
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%); 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: white; margin: 0;">{best.get('model_name', 'N/A')}</h3>
            <div style="display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
                <div>
                    <span style="color: rgba(255,255,255,0.7);">R²</span><br>
                    <span style="color: white; font-size: 1.5rem; font-weight: bold;">
                        {best.get('test_r2', 0):.4f}
                    </span>
                </div>
                <div>
                    <span style="color: rgba(255,255,255,0.7);">RMSE</span><br>
                    <span style="color: white; font-size: 1.5rem; font-weight: bold;">
                        {best.get('test_rmse', 0):.4f}
                    </span>
                </div>
                <div>
                    <span style="color: rgba(255,255,255,0.7);">RPD</span><br>
                    <span style="color: white; font-size: 1.5rem; font-weight: bold;">
                        {best.get('rpd', 0):.2f}
                    </span>
                </div>
                <div>
                    <span style="color: rgba(255,255,255,0.7);">Correlation</span><br>
                    <span style="color: white; font-size: 1.5rem; font-weight: bold;">
                        {best.get('correlation', 0):.4f}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance by preprocessing
    st.markdown("#### 📈 Performance by Preprocessing Method")
    
    prep_stats = {}
    for r in successful_results:
        prep = r.get('preprocessing', 'Unknown')
        if prep not in prep_stats:
            prep_stats[prep] = {'r2_values': [], 'rmse_values': []}
        prep_stats[prep]['r2_values'].append(r.get('test_r2', 0))
        prep_stats[prep]['rmse_values'].append(r.get('test_rmse', 0))
    
    prep_summary = []
    for prep, stats in prep_stats.items():
        prep_summary.append({
            'Preprocessing': prep.title(),
            'Avg R²': np.mean(stats['r2_values']),
            'Max R²': np.max(stats['r2_values']),
            'Avg RMSE': np.mean(stats['rmse_values']),
            'Min RMSE': np.min(stats['rmse_values']),
            'N Models': len(stats['r2_values'])
        })
    
    prep_df = pd.DataFrame(prep_summary).sort_values('Avg R²', ascending=False)
    st.dataframe(prep_df, use_container_width=True, hide_index=True)
    
    # Performance by model type
    st.markdown("#### 🤖 Performance by Model Type")
    
    model_stats = {}
    for r in successful_results:
        model = r.get('model_type', 'Unknown')
        if model not in model_stats:
            model_stats[model] = {'r2_values': [], 'rmse_values': []}
        model_stats[model]['r2_values'].append(r.get('test_r2', 0))
        model_stats[model]['rmse_values'].append(r.get('test_rmse', 0))
    
    model_summary = []
    for model, stats in model_stats.items():
        model_summary.append({
            'Model': model.upper(),
            'Avg R²': np.mean(stats['r2_values']),
            'Max R²': np.max(stats['r2_values']),
            'Avg RMSE': np.mean(stats['rmse_values']),
            'Min RMSE': np.min(stats['rmse_values']),
            'N Configs': len(stats['r2_values'])
        })
    
    model_df = pd.DataFrame(model_summary).sort_values('Avg R²', ascending=False)
    st.dataframe(model_df, use_container_width=True, hide_index=True)


def render_export_data(successful_results: list):
    """Render data export options."""
    
    st.markdown("### 📥 Export Data")
    
    # Data selection
    st.markdown("#### Select data to export:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_metrics = st.checkbox("Performance Metrics", value=True)
        include_cv = st.checkbox("Cross-Validation Scores", value=True)
    
    with col2:
        include_predictions = st.checkbox("Predictions (y_test, y_pred)", value=False)
        include_metadata = st.checkbox("Model Metadata", value=True)
    
    st.markdown("---")
    
    # Prepare export data
    export_data = []
    
    for r in successful_results:
        row = {
            'model_name': r.get('model_name', ''),
            'preprocessing': r.get('preprocessing', ''),
            'model_type': r.get('model_type', ''),
        }
        
        if include_metrics:
            row.update({
                'test_r2': r.get('test_r2', 0),
                'test_rmse': r.get('test_rmse', 0),
                'test_mae': r.get('test_mae', 0),
                'rpd': r.get('rpd', 0),
                'correlation': r.get('correlation', 0),
                'bias': r.get('bias', 0),
            })
        
        if include_cv:
            row.update({
                'cv_mean': r.get('cv_mean', 0),
                'cv_std': r.get('cv_std', 0),
            })
        
        if include_metadata:
            row.update({
                'n_train': r.get('n_train', 0),
                'n_test': r.get('n_test', 0),
                'train_time': r.get('train_time', 0),
            })
        
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Preview
    st.markdown("#### 📋 Export Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Download buttons
    st.markdown("#### ⬇️ Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        excel_data = export_results_to_excel(export_data)
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Predictions export
    if include_predictions:
        st.markdown("---")
        st.markdown("#### 📊 Export Predictions")
        
        model_names = [r.get('model_name', 'Unknown') for r in successful_results]
        selected_model = st.selectbox("Select model for predictions export:", options=model_names)
        
        if selected_model:
            model_result = next((r for r in successful_results if r.get('model_name') == selected_model), None)
            
            if model_result and 'y_test' in model_result and 'y_pred_test' in model_result:
                pred_df = pd.DataFrame({
                    'Actual': model_result['y_test'],
                    'Predicted': model_result['y_pred_test'],
                    'Residual': np.array(model_result['y_test']) - np.array(model_result['y_pred_test'])
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                csv_pred = pred_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Predictions ({selected_model})",
                    data=csv_pred,
                    file_name=f"predictions_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )


def render_download_models(successful_results: list):
    """Render model download options with ZIP functionality."""
    
    st.markdown("### 💾 Download Trained Models")
    
    st.info("""
    Download trained model files (.pkl) for deployment or further analysis. 
    Models can be loaded using Python's `joblib.load()` function.
    """)
    
    # Model selection table
    st.markdown("#### Select models to download:")
    
    model_data = []
    for r in successful_results:
        model_data.append({
            'Model': r.get('model_name', 'Unknown'),
            'R²': f"{r.get('test_r2', 0):.4f}",
            'RMSE': f"{r.get('test_rmse', 0):.4f}",
            'RPD': f"{r.get('rpd', 0):.2f}"
        })
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    # Selection
    model_names = [r.get('model_name', 'Unknown') for r in successful_results]
    selected_models = st.multiselect(
        "Select models:",
        options=model_names,
        default=[successful_results[0].get('model_name', 'Unknown')] if successful_results else []
    )
    
    st.markdown("---")
    
    # Individual downloads
    if selected_models:
        st.markdown(f"#### ⬇️ Download Selected ({len(selected_models)} models)")
        
        for model_name in selected_models:
            model_result = next((r for r in successful_results if r.get('model_name') == model_name), None)
            
            if model_result:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 **{model_name}**")
                
                with col2:
                    r2 = model_result.get('test_r2', 0)
                    st.write(f"R² = {r2:.4f}")
                
                with col3:
                    # Create model info JSON for download
                    model_info = {
                        'model_name': model_name,
                        'preprocessing': model_result.get('preprocessing'),
                        'model_type': model_result.get('model_type'),
                        'test_r2': model_result.get('test_r2'),
                        'test_rmse': model_result.get('test_rmse'),
                        'rpd': model_result.get('rpd'),
                        'correlation': model_result.get('correlation'),
                        'cv_mean': model_result.get('cv_mean'),
                        'cv_std': model_result.get('cv_std'),
                        'n_train': model_result.get('n_train'),
                        'n_test': model_result.get('n_test'),
                        'train_time': model_result.get('train_time')
                    }
                    
                    st.download_button(
                        label="💾",
                        data=json.dumps(model_info, indent=2),
                        file_name=f"{model_name.replace(' ', '_')}_info.json",
                        mime="application/json",
                        key=f"dl_{model_name}"
                    )
        
        st.markdown("---")
        
        # Batch download as ZIP
        st.markdown("#### 📦 Batch Download")
        
        if st.button("📥 Download All Selected as ZIP", use_container_width=True, type="primary"):
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add model info files
                for model_name in selected_models:
                    model_result = next((r for r in successful_results if r.get('model_name') == model_name), None)
                    
                    if model_result:
                        model_info = {
                            'model_name': model_name,
                            'preprocessing': model_result.get('preprocessing'),
                            'model_type': model_result.get('model_type'),
                            'test_r2': model_result.get('test_r2'),
                            'test_rmse': model_result.get('test_rmse'),
                            'rpd': model_result.get('rpd'),
                            'correlation': model_result.get('correlation'),
                            'cv_mean': model_result.get('cv_mean'),
                            'cv_std': model_result.get('cv_std'),
                            'n_train': model_result.get('n_train'),
                            'n_test': model_result.get('n_test'),
                            'train_time': model_result.get('train_time')
                        }
                        
                        # Add info JSON
                        zip_file.writestr(
                            f"{model_name.replace(' ', '_')}_info.json",
                            json.dumps(model_info, indent=2)
                        )
                        
                        # Add predictions if available
                        if 'y_test' in model_result and 'y_pred_test' in model_result:
                            pred_df = pd.DataFrame({
                                'Actual': model_result['y_test'],
                                'Predicted': model_result['y_pred_test']
                            })
                            zip_file.writestr(
                                f"{model_name.replace(' ', '_')}_predictions.csv",
                                pred_df.to_csv(index=False)
                            )
                
                # Add summary CSV
                summary_data = []
                for model_name in selected_models:
                    model_result = next((r for r in successful_results if r.get('model_name') == model_name), None)
                    if model_result:
                        summary_data.append({
                            'model_name': model_name,
                            'preprocessing': model_result.get('preprocessing'),
                            'model_type': model_result.get('model_type'),
                            'test_r2': model_result.get('test_r2'),
                            'test_rmse': model_result.get('test_rmse'),
                            'rpd': model_result.get('rpd')
                        })
                
                summary_df = pd.DataFrame(summary_data)
                zip_file.writestr("summary.csv", summary_df.to_csv(index=False))
                
                # Add README
                readme_content = f"""# Model Export Package
                
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Models: {len(selected_models)}

## Contents
- summary.csv: Summary of all models
- *_info.json: Detailed info for each model
- *_predictions.csv: Predictions for each model

## Usage
```python
import json

# Load model info
with open('model_info.json', 'r') as f:
    info = json.load(f)
    print(f"R²: {{info['test_r2']}}")
```
"""
                zip_file.writestr("README.md", readme_content)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download ZIP Package",
                data=zip_buffer.getvalue(),
                file_name=f"models_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            st.success(f"✅ ZIP package ready with {len(selected_models)} models!")
    
    else:
        st.warning("Select at least one model to download.")
    
    # Usage instructions
    st.markdown("---")
    st.markdown("#### 📖 Usage Instructions")
    
    st.code("""
# Load model info
import json

with open('model_info.json', 'r') as f:
    info = json.load(f)

print(f"Model: {info['model_name']}")
print(f"R²: {info['test_r2']}")
print(f"RMSE: {info['test_rmse']}")

# If you have the actual model file (.pkl):
import joblib
model = joblib.load('model_file.pkl')
predictions = model.predict(X_new)
    """, language="python")


def render_pdf_report(results: dict, successful_results: list):
    """Render PDF report generation."""
    
    st.markdown("### 📄 Generate PDF Report")
    
    st.info("""
    Generate a comprehensive report containing all training results and analysis.
    """)
    
    # Report options
    st.markdown("#### 📋 Report Contents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_summary = st.checkbox("Executive Summary", value=True)
        include_leaderboard = st.checkbox("Model Leaderboard", value=True)
        include_charts = st.checkbox("Performance Charts", value=True)
    
    with col2:
        include_diagnostics = st.checkbox("Diagnostic Analysis", value=True)
        include_recommendations = st.checkbox("Recommendations", value=True)
        include_appendix = st.checkbox("Technical Appendix", value=False)
    
    # Report customization
    st.markdown("#### 🎨 Customization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", value="Spectral Soil Modeling - Training Report")
        author = st.text_input("Author", value="")
    
    with col2:
        date = st.date_input("Report Date", value=datetime.now())
    
    st.markdown("---")
    
    # Generate reports
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Text Report", use_container_width=True):
            report_text = generate_text_report(results, successful_results, report_title, author, str(date))
            
            st.download_button(
                label="📥 Download Text Report",
                data=report_text,
                file_name=f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col2:
        if st.button("📝 Generate Markdown", use_container_width=True):
            md_content = generate_markdown_report(results, successful_results, report_title, author, str(date))
            
            st.download_button(
                label="📥 Download Markdown",
                data=md_content,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    with col3:
        if st.button("📊 Generate HTML", use_container_width=True):
            html_content = generate_html_report(results, successful_results, report_title, author, str(date))
            
            st.download_button(
                label="📥 Download HTML",
                data=html_content,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
    
    # Preview
    st.markdown("---")
    st.markdown("#### 📋 Report Preview")
    
    best = results.get('best_model', {})
    
    with st.expander("View Report Preview", expanded=True):
        st.markdown(f"""
# {report_title}

**Author:** {author}  
**Date:** {date}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Models | {results.get('total_models', 0)} |
| Successful | {results.get('successful_models', 0)} |
| Best R² | {best.get('test_r2', 0):.4f} |
| Best Model | {best.get('model_name', 'N/A')} |

---

## Top 5 Models

| Rank | Model | R² | RMSE | RPD |
|------|-------|-----|------|-----|
""")
        
        sorted_results = sorted(successful_results, key=lambda x: x.get('test_r2', 0), reverse=True)[:5]
        for i, r in enumerate(sorted_results, 1):
            st.markdown(f"| {get_medal(i)} | {r.get('model_name', 'N/A')} | {r.get('test_r2', 0):.4f} | {r.get('test_rmse', 0):.4f} | {r.get('rpd', 0):.2f} |")


def generate_text_report(results: dict, successful_results: list, title: str, author: str, date: str) -> str:
    """Generate a text report."""
    
    best = results.get('best_model', {})
    
    report = f"""
{'='*60}
{title.upper()}
{'='*60}

Author: {author}
Date: {date}

{'='*60}
EXECUTIVE SUMMARY
{'='*60}

Total Models Trained: {results.get('total_models', 0)}
Successful Models: {results.get('successful_models', 0)}
Failed Models: {results.get('failed_models', 0)}
Total Training Time: {results.get('total_time', 0):.1f} seconds

Best Model: {best.get('model_name', 'N/A')}
Best R²: {best.get('test_r2', 0):.4f}
Best RMSE: {best.get('test_rmse', 0):.4f}
Best RPD: {best.get('rpd', 0):.2f}

{'='*60}
MODEL LEADERBOARD
{'='*60}

{'Rank':<6}{'Model':<40}{'R²':<12}{'RMSE':<12}{'RPD':<10}
{'-'*80}
"""
    
    sorted_results = sorted(successful_results, key=lambda x: x.get('test_r2', 0), reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        report += f"{i:<6}{r.get('model_name', 'N/A')[:38]:<40}{r.get('test_r2', 0):<12.4f}{r.get('test_rmse', 0):<12.4f}{r.get('rpd', 0):<10.2f}\n"
    
    report += f"""
{'='*60}
RECOMMENDATIONS
{'='*60}

Based on the training results, the recommended model for deployment is:
{best.get('model_name', 'N/A')}

This model achieved:
- R² = {best.get('test_r2', 0):.4f}
- RMSE = {best.get('test_rmse', 0):.4f}
- RPD = {best.get('rpd', 0):.2f}

{'='*60}
Report generated by Spectral Soil Modeler

{'='*60}
"""
    
    return report


def generate_markdown_report(results: dict, successful_results: list, title: str, author: str, date: str) -> str:
    """Generate a markdown report."""
    
    best = results.get('best_model', {})
    
    report = f"""# {title}

**Author:** {author}  
**Date:** {date}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Models | {results.get('total_models', 0)} |
| Successful | {results.get('successful_models', 0)} |
| Best R² | {best.get('test_r2', 0):.4f} |
| Best Model | {best.get('model_name', 'N/A')} |

---

## Best Model Details

- **Name:** {best.get('model_name', 'N/A')}
- **R²:** {best.get('test_r2', 0):.4f}
- **RMSE:** {best.get('test_rmse', 0):.4f}
- **RPD:** {best.get('rpd', 0):.2f}

---

## Model Leaderboard

| Rank | Model | Preprocessing | Algorithm | R² | RMSE | RPD |
|------|-------|---------------|-----------|-----|------|-----|
"""
    
    sorted_results = sorted(successful_results, key=lambda x: x.get('test_r2', 0), reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        report += f"| {i} | {r.get('model_name', 'N/A')} | {r.get('preprocessing', 'N/A')} | {r.get('model_type', 'N/A').upper()} | {r.get('test_r2', 0):.4f} | {r.get('test_rmse', 0):.4f} | {r.get('rpd', 0):.2f} |\n"
    
    report += """
---

## Recommendations

Based on the analysis, we recommend using the top-performing model for deployment.

---

*Report generated by Spectral Soil Modeler - *
"""
    
    return report


def generate_html_report(results: dict, successful_results: list, title: str, author: str, date: str) -> str:
    """Generate an HTML report."""
    
    best = results.get('best_model', {})
    sorted_results = sorted(successful_results, key=lambda x: x.get('test_r2', 0), reverse=True)
    
    # Build table rows
    table_rows = ""
    for i, r in enumerate(sorted_results, 1):
        table_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{r.get('model_name', 'N/A')}</td>
            <td>{r.get('preprocessing', 'N/A')}</td>
            <td>{r.get('model_type', 'N/A').upper()}</td>
            <td>{r.get('test_r2', 0):.4f}</td>
            <td>{r.get('test_rmse', 0):.4f}</td>
            <td>{r.get('rpd', 0):.2f}</td>
        </tr>
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .best-model {{ background: linear-gradient(135deg, #4CAF50, #66BB6A); color: white; padding: 20px; border-radius: 12px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-size: 12px; opacity: 0.8; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .footer {{ text-align: center; color: #888; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p><strong>Author:</strong> {author} | <strong>Date:</strong> {date}</p>
        
        <h2>📋 Executive Summary</h2>
        <table>
            <tr><td><strong>Total Models</strong></td><td>{results.get('total_models', 0)}</td></tr>
            <tr><td><strong>Successful</strong></td><td>{results.get('successful_models', 0)}</td></tr>
            <tr><td><strong>Training Time</strong></td><td>{results.get('total_time', 0):.1f}s</td></tr>
        </table>
        
        <h2>🏆 Best Model</h2>
        <div class="best-model">
            <h3 style="margin:0">{best.get('model_name', 'N/A')}</h3>
            <div class="metric">
                <div class="metric-label">R²</div>
                <div class="metric-value">{best.get('test_r2', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{best.get('test_rmse', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">RPD</div>
                <div class="metric-value">{best.get('rpd', 0):.2f}</div>
            </div>
        </div>
        
        <h2>📈 Model Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Preprocessing</th>
                    <th>Algorithm</th>
                    <th>R²</th>
                    <th>RMSE</th>
                    <th>RPD</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Report generated by Spectral Soil Modeler</p>
            <p></p>
        </div>
    </div>
</body>
</html>
"""
    
    return html
