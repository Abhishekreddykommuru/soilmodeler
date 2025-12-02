"""
Visualization Module
====================
Creates all visualizations for the Spectral Soil Modeler.
Uses Plotly for interactive charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Creates visualizations for model analysis."""
    
    # Color schemes
    COLORS = {
        'primary': '#4A90E2',
        'secondary': '#50C878',
        'accent': '#FFB84D',
        'danger': '#FF6B6B',
        'info': '#2196F3',
        'success': '#4CAF50',
        'warning': '#FFC107',
        'dark': '#262B33',
        'light': '#FAFAFA'
    }
    
    PREPROCESSING_COLORS = {
        'reflectance': '#4A90E2',
        'absorbance': '#50C878',
        'continuum_removal': '#FFB84D'
    }
    
    MODEL_COLORS = {
        'plsr': '#667eea',
        'cubist': '#4fc3f7',
        'gbrt': '#81c784',
        'krr': '#ffb74d',
        'svr': '#e57373'
    }
    
    def __init__(self):
        # Common layout settings
        self.layout_template = {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#FAFAFA', 'size': 12},
            'xaxis': {
                'gridcolor': 'rgba(255,255,255,0.1)',
                'zerolinecolor': 'rgba(255,255,255,0.2)'
            },
            'yaxis': {
                'gridcolor': 'rgba(255,255,255,0.1)',
                'zerolinecolor': 'rgba(255,255,255,0.2)'
            }
        }
    
    def performance_bar_chart(
        self, 
        results: List[Dict], 
        metric: str = 'test_r2',
        title: str = "Model Performance Comparison"
    ) -> go.Figure:
        """Create a bar chart comparing model performance."""
        df = pd.DataFrame(results)
        df = df[df['status'] == 'success'] if 'status' in df.columns else df
        df = df.sort_values(metric, ascending=False)
        
        # Create color mapping
        colors = [self.PREPROCESSING_COLORS.get(p, '#4A90E2') for p in df['preprocessing']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['model_name'] if 'model_name' in df.columns else df.index,
            y=df[metric],
            marker_color=colors,
            text=[f"{v:.3f}" for v in df[metric]],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' + f'{metric}: ' + '%{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title=metric.upper().replace('_', ' '),
            **self.layout_template
        )
        
        return fig
    
    def grouped_bar_chart(
        self, 
        results: List[Dict], 
        metric: str = 'test_r2',
        group_by: str = 'preprocessing'
    ) -> go.Figure:
        """Create a grouped bar chart."""
        df = pd.DataFrame(results)
        df = df[df['status'] == 'success'] if 'status' in df.columns else df
        
        fig = go.Figure()
        
        if group_by == 'preprocessing':
            groups = df['preprocessing'].unique()
            models = df['model_type'].unique()
            
            for model in models:
                model_data = df[df['model_type'] == model]
                fig.add_trace(go.Bar(
                    name=model.upper(),
                    x=model_data['preprocessing'],
                    y=model_data[metric],
                    marker_color=self.MODEL_COLORS.get(model, '#4A90E2'),
                    text=[f"{v:.3f}" for v in model_data[metric]],
                    textposition='outside'
                ))
        else:
            groups = df['model_type'].unique()
            preps = df['preprocessing'].unique()
            
            for prep in preps:
                prep_data = df[df['preprocessing'] == prep]
                fig.add_trace(go.Bar(
                    name=prep.title(),
                    x=prep_data['model_type'],
                    y=prep_data[metric],
                    marker_color=self.PREPROCESSING_COLORS.get(prep, '#4A90E2'),
                    text=[f"{v:.3f}" for v in prep_data[metric]],
                    textposition='outside'
                ))
        
        fig.update_layout(
            title=f"Performance by {group_by.title()}",
            barmode='group',
            xaxis_title=group_by.title(),
            yaxis_title=metric.upper().replace('_', ' '),
            **self.layout_template
        )
        
        return fig
    
    def performance_heatmap(
        self, 
        results: List[Dict], 
        metric: str = 'test_r2'
    ) -> go.Figure:
        """Create a heatmap of model performance."""
        df = pd.DataFrame(results)
        df = df[df['status'] == 'success'] if 'status' in df.columns else df
        
        # Pivot for heatmap
        pivot = df.pivot(index='preprocessing', columns='model_type', values=metric)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            hovertemplate='Preprocessing: %{y}<br>Model: %{x}<br>' + f'{metric}: ' + '%{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Performance Heatmap - {metric.upper()}",
            xaxis_title="Model Type",
            yaxis_title="Preprocessing",
            **self.layout_template
        )
        
        return fig
    
    def scatter_prediction(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        title: str = "Predicted vs Actual Values",
        show_confidence: bool = True
    ) -> go.Figure:
        """Create a scatter plot of predicted vs actual values."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate residuals for coloring
        residuals = np.abs(y_pred - y_true)
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(
                size=8,
                color=residuals,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Residual"),
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<br>Error: %{marker.color:.3f}<extra></extra>'
        ))
        
        # Add 1:1 line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        padding = (max_val - min_val) * 0.05
        
        fig.add_trace(go.Scatter(
            x=[min_val - padding, max_val + padding],
            y=[min_val - padding, max_val + padding],
            mode='lines',
            line=dict(color='#FF6B6B', dash='dash', width=2),
            name='1:1 Line'
        ))
        
        # Add confidence bands if requested
        if show_confidence:
            std_residual = np.std(residuals)
            fig.add_trace(go.Scatter(
                x=[min_val - padding, max_val + padding],
                y=[min_val - padding + 1.96 * std_residual, max_val + padding + 1.96 * std_residual],
                mode='lines',
                line=dict(color='rgba(176,176,176,0.5)', dash='dot', width=1),
                name='+95% CI',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[min_val - padding, max_val + padding],
                y=[min_val - padding - 1.96 * std_residual, max_val + padding - 1.96 * std_residual],
                mode='lines',
                line=dict(color='rgba(176,176,176,0.5)', dash='dot', width=1),
                name='-95% CI',
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            showlegend=True,
            **self.layout_template
        )
        
        # Make axes equal
        fig.update_xaxes(range=[min_val - padding, max_val + padding])
        fig.update_yaxes(range=[min_val - padding, max_val + padding])
        
        return fig
    
    def residual_plot(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        title: str = "Residual Analysis"
    ) -> go.Figure:
        """Create a residual plot."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_pred - y_true
        
        fig = go.Figure()
        
        # Scatter of residuals
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color=np.abs(residuals),
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="|Residual|"),
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line=dict(color='#FF6B6B', dash='dash', width=2))
        
        # Add ±2σ lines
        std_res = np.std(residuals)
        fig.add_hline(y=2*std_res, line=dict(color='rgba(255,193,7,0.5)', dash='dot', width=1))
        fig.add_hline(y=-2*std_res, line=dict(color='rgba(255,193,7,0.5)', dash='dot', width=1))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            **self.layout_template
        )
        
        return fig
    
    def residual_histogram(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        title: str = "Residual Distribution"
    ) -> go.Figure:
        """Create a histogram of residuals."""
        residuals = np.array(y_pred).flatten() - np.array(y_true).flatten()
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color=self.COLORS['primary'],
            opacity=0.7,
            name='Residuals'
        ))
        
        # Normal distribution overlay
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        from scipy import stats
        y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
        y_norm = y_norm * len(residuals) * (residuals.max() - residuals.min()) / 30
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            line=dict(color=self.COLORS['danger'], width=2),
            name='Normal Distribution'
        ))
        
        # Mean line
        fig.add_vline(x=np.mean(residuals), line=dict(color=self.COLORS['success'], width=2))
        
        fig.update_layout(
            title=title,
            xaxis_title="Residual Value",
            yaxis_title="Frequency",
            showlegend=True,
            **self.layout_template
        )
        
        return fig
    
    def qq_plot(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        title: str = "Q-Q Plot"
    ) -> go.Figure:
        """Create a Q-Q plot for residual normality check."""
        from scipy import stats
        
        residuals = np.array(y_pred).flatten() - np.array(y_true).flatten()
        
        # Calculate theoretical quantiles
        sorted_residuals = np.sort(residuals)
        n = len(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        
        fig = go.Figure()
        
        # Q-Q scatter
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            marker=dict(size=8, color=self.COLORS['primary'], opacity=0.7),
            name='Residuals'
        ))
        
        # Reference line
        fig.add_trace(go.Scatter(
            x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
            y=[sorted_residuals.min(), sorted_residuals.max()],
            mode='lines',
            line=dict(color=self.COLORS['danger'], dash='dash', width=2),
            name='Reference Line'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            showlegend=True,
            **self.layout_template
        )
        
        return fig
    
    def feature_importance_chart(
        self, 
        importance: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """Create a feature importance bar chart."""
        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_items) if sorted_items else ([], [])
        
        # Normalize values
        max_val = max(values) if values else 1
        normalized = [v / max_val for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=list(features)[::-1],
            x=list(normalized)[::-1],
            orientation='h',
            marker=dict(
                color=list(normalized)[::-1],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f"{v:.3f}" for v in list(normalized)[::-1]],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Relative Importance",
            yaxis_title="Feature",
            height=max(400, top_n * 25),
            **self.layout_template
        )
        
        return fig
    
    def target_distribution(
        self, 
        target: np.ndarray,
        title: str = "Target Variable Distribution"
    ) -> go.Figure:
        """Create a histogram of target variable."""
        target = np.array(target).flatten()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=target,
            nbinsx=30,
            marker_color=self.COLORS['secondary'],
            opacity=0.7,
            name='Target'
        ))
        
        # Add mean and median lines
        fig.add_vline(
            x=np.mean(target), 
            line=dict(color=self.COLORS['danger'], width=2, dash='solid'),
            annotation_text=f"Mean: {np.mean(target):.3f}"
        )
        fig.add_vline(
            x=np.median(target),
            line=dict(color=self.COLORS['warning'], width=2, dash='dash'),
            annotation_text=f"Median: {np.median(target):.3f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Target Value",
            yaxis_title="Frequency",
            **self.layout_template
        )
        
        return fig
    
    def cv_scores_boxplot(
        self, 
        results: List[Dict],
        title: str = "Cross-Validation Score Distribution"
    ) -> go.Figure:
        """Create a box plot of cross-validation scores."""
        fig = go.Figure()
        
        for r in results:
            if 'cv_scores' in r and r['cv_scores']:
                fig.add_trace(go.Box(
                    y=r['cv_scores'],
                    name=r.get('model_name', 'Model'),
                    marker_color=self.PREPROCESSING_COLORS.get(r.get('preprocessing', ''), '#4A90E2'),
                    boxpoints='all',
                    jitter=0.3
                ))
        
        fig.update_layout(
            title=title,
            yaxis_title="R² Score",
            **self.layout_template
        )
        
        return fig
    
    def radar_chart(
        self, 
        results: List[Dict],
        metrics: List[str] = ['test_r2', 'rpd', 'correlation'],
        title: str = "Multi-Metric Comparison"
    ) -> go.Figure:
        """Create a radar chart comparing multiple metrics."""
        fig = go.Figure()
        
        # Normalize metrics to 0-1 range
        for r in results[:5]:  # Limit to 5 models
            values = []
            for m in metrics:
                val = r.get(m, 0)
                if m == 'test_rmse':  # Invert RMSE (lower is better)
                    val = 1 - min(val, 1)
                values.append(min(max(val, 0), 1))
            
            # Close the radar
            values.append(values[0])
            categories = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=r.get('model_name', 'Model'),
                opacity=0.5
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor='rgba(0,0,0,0)'
            ),
            title=title,
            showlegend=True,
            **self.layout_template
        )
        
        return fig
