"""
Model Evaluation Module
=======================
Calculates performance metrics and diagnostic statistics for trained models.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics."""
    
    def __init__(self):
        pass
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return self._empty_metrics()
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 2:
            return self._empty_metrics()
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # RPD (Ratio of Performance to Deviation)
        std_true = np.std(y_true)
        rpd = std_true / rmse if rmse > 0 else 0
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Correlation
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        # Residuals
        residuals = y_pred - y_true
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'rpd': rpd,
            'mape': mape,
            'correlation': correlation,
            'correlation_p_value': p_value,
            'bias': bias,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'n_samples': len(y_true)
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'r2': 0,
            'rmse': float('inf'),
            'mae': float('inf'),
            'rpd': 0,
            'mape': float('inf'),
            'correlation': 0,
            'correlation_p_value': 1,
            'bias': 0,
            'residuals_mean': 0,
            'residuals_std': 0,
            'n_samples': 0
        }
    
    def calculate_residual_diagnostics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive residual diagnostics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of diagnostic results
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_pred - y_true
        
        # Basic statistics
        diagnostics = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'range': np.max(residuals) - np.min(residuals),
            'median': np.median(residuals),
            'q1': np.percentile(residuals, 25),
            'q3': np.percentile(residuals, 75),
            'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Normality tests
        if len(residuals) >= 8:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for large samples
            diagnostics['shapiro_stat'] = shapiro_stat
            diagnostics['shapiro_p'] = shapiro_p
            diagnostics['shapiro_pass'] = shapiro_p > 0.05
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
            diagnostics['ks_stat'] = ks_stat
            diagnostics['ks_p'] = ks_p
            diagnostics['ks_pass'] = ks_p > 0.05
            
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(residuals)
            diagnostics['jb_stat'] = jb_stat
            diagnostics['jb_p'] = jb_p
            diagnostics['jb_pass'] = jb_p > 0.05
        else:
            diagnostics['shapiro_pass'] = True
            diagnostics['ks_pass'] = True
            diagnostics['jb_pass'] = True
        
        # Outlier detection
        q1, q3 = diagnostics['q1'], diagnostics['q3']
        iqr = diagnostics['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
        
        diagnostics['n_outliers'] = np.sum(outlier_mask)
        diagnostics['outlier_percentage'] = (np.sum(outlier_mask) / len(residuals)) * 100
        diagnostics['outlier_indices'] = np.where(outlier_mask)[0].tolist()
        
        # Z-score outliers
        z_scores = np.abs(stats.zscore(residuals))
        diagnostics['n_zscore_outliers'] = np.sum(z_scores > 3)
        
        # Homoscedasticity check (simple version)
        n = len(residuals)
        mid = n // 2
        var_first_half = np.var(residuals[:mid])
        var_second_half = np.var(residuals[mid:])
        diagnostics['variance_ratio'] = var_first_half / var_second_half if var_second_half > 0 else 1
        diagnostics['homoscedastic'] = 0.5 < diagnostics['variance_ratio'] < 2.0
        
        # Overall assessment
        normality_pass = diagnostics.get('shapiro_pass', True) or diagnostics.get('ks_pass', True)
        low_outliers = diagnostics['outlier_percentage'] < 5
        
        if normality_pass and low_outliers and diagnostics['homoscedastic']:
            diagnostics['overall_quality'] = 'Excellent'
        elif normality_pass and low_outliers:
            diagnostics['overall_quality'] = 'Good'
        elif low_outliers:
            diagnostics['overall_quality'] = 'Fair'
        else:
            diagnostics['overall_quality'] = 'Poor'
        
        return diagnostics
    
    def get_performance_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Get interpretations of performance metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary of interpretations
        """
        interpretations = {}
        
        # R² interpretation
        r2 = metrics.get('r2', 0)
        if r2 >= 0.90:
            interpretations['r2'] = "Excellent - Very strong predictive power"
        elif r2 >= 0.80:
            interpretations['r2'] = "Good - Strong predictive power"
        elif r2 >= 0.70:
            interpretations['r2'] = "Moderate - Acceptable for some applications"
        elif r2 >= 0.50:
            interpretations['r2'] = "Fair - Weak predictive power"
        else:
            interpretations['r2'] = "Poor - Model needs improvement"
        
        # RPD interpretation
        rpd = metrics.get('rpd', 0)
        if rpd >= 3.0:
            interpretations['rpd'] = "Excellent - Suitable for quantitative predictions"
        elif rpd >= 2.5:
            interpretations['rpd'] = "Very Good - Good for quantitative predictions"
        elif rpd >= 2.0:
            interpretations['rpd'] = "Good - Suitable for semi-quantitative predictions"
        elif rpd >= 1.5:
            interpretations['rpd'] = "Fair - Rough screening possible"
        else:
            interpretations['rpd'] = "Poor - Not suitable for predictions"
        
        # Bias interpretation
        bias = abs(metrics.get('bias', 0))
        rmse = metrics.get('rmse', 1)
        if bias < 0.1 * rmse:
            interpretations['bias'] = "Excellent - Nearly unbiased predictions"
        elif bias < 0.3 * rmse:
            interpretations['bias'] = "Good - Minor systematic bias"
        else:
            interpretations['bias'] = "Fair - Notable systematic bias"
        
        # Correlation interpretation
        corr = abs(metrics.get('correlation', 0))
        if corr >= 0.95:
            interpretations['correlation'] = "Very strong linear relationship"
        elif corr >= 0.80:
            interpretations['correlation'] = "Strong linear relationship"
        elif corr >= 0.60:
            interpretations['correlation'] = "Moderate linear relationship"
        else:
            interpretations['correlation'] = "Weak linear relationship"
        
        return interpretations
    
    def compare_models(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and generate insights.
        
        Args:
            results: List of model result dictionaries
            
        Returns:
            Comparison summary
        """
        if not results:
            return {}
        
        # Extract metrics for comparison
        r2_values = [r.get('test_r2', 0) for r in results]
        rmse_values = [r.get('test_rmse', float('inf')) for r in results]
        rpd_values = [r.get('rpd', 0) for r in results]
        
        # Find best models
        best_r2_idx = np.argmax(r2_values)
        best_rmse_idx = np.argmin(rmse_values)
        best_rpd_idx = np.argmax(rpd_values)
        
        # Group by preprocessing
        prep_groups = {}
        for r in results:
            prep = r.get('preprocessing', 'unknown')
            if prep not in prep_groups:
                prep_groups[prep] = []
            prep_groups[prep].append(r.get('test_r2', 0))
        
        prep_avg = {k: np.mean(v) for k, v in prep_groups.items()}
        best_prep = max(prep_avg, key=prep_avg.get)
        
        # Group by model
        model_groups = {}
        for r in results:
            model = r.get('model_type', 'unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(r.get('test_r2', 0))
        
        model_avg = {k: np.mean(v) for k, v in model_groups.items()}
        best_model = max(model_avg, key=model_avg.get)
        
        return {
            'best_overall': results[best_r2_idx],
            'best_by_r2': results[best_r2_idx],
            'best_by_rmse': results[best_rmse_idx],
            'best_by_rpd': results[best_rpd_idx],
            'best_preprocessing': best_prep,
            'best_model_type': best_model,
            'preprocessing_rankings': prep_avg,
            'model_rankings': model_avg,
            'r2_stats': {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'min': np.min(r2_values),
                'max': np.max(r2_values)
            },
            'rmse_stats': {
                'mean': np.mean(rmse_values),
                'std': np.std(rmse_values),
                'min': np.min(rmse_values),
                'max': np.max(rmse_values)
            }
        }
