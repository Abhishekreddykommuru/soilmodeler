"""
Models Module
=============
Defines all machine learning models for soil property prediction.
- PLSR: Partial Least Squares Regression
- Cubist: Rule-based regression (implemented as Gradient Boosting)
- GBRT: Gradient Boosting Regression Trees
- KRR: Kernel Ridge Regression
- SVR: Support Vector Regression
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelFactory:
    """Factory class for creating ML models."""
    
    MODELS = {
        'plsr': {
            'name': 'PLSR',
            'full_name': 'Partial Least Squares Regression',
            'description': 'Linear model that finds latent variables (components) with maximum covariance with target',
            'icon': '📈'
        },
        'cubist': {
            'name': 'Cubist',
            'full_name': 'Cubist Rule-Based Regression',
            'description': 'Rule-based regression using decision trees with linear models at leaves',
            'icon': '🌳'
        },
        'gbrt': {
            'name': 'GBRT',
            'full_name': 'Gradient Boosting Regression Trees',
            'description': 'Ensemble of decision trees trained sequentially to correct errors',
            'icon': '🚀'
        },
        'krr': {
            'name': 'KRR',
            'full_name': 'Kernel Ridge Regression',
            'description': 'Ridge regression with kernel trick for non-linear relationships',
            'icon': '🔮'
        },
        'svr': {
            'name': 'SVR',
            'full_name': 'Support Vector Regression',
            'description': 'Support vector machine for regression with epsilon-insensitive loss',
            'icon': '⚡'
        }
    }
    
    # Default hyperparameters
    DEFAULT_PARAMS = {
        'plsr': {
            'n_components': 10,
            'scale': True,
            'max_iter': 500
        },
        'cubist': {  # Using GradientBoosting as Cubist approximation
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        },
        'gbrt': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'min_samples_split': 5,
            'random_state': 42
        },
        'krr': {
            'alpha': 1.0,
            'kernel': 'rbf',
            'gamma': None  # Will use 'scale'
        },
        'svr': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
    }
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_model(
        self, 
        model_type: str, 
        params: Optional[Dict[str, Any]] = None,
        include_scaler: bool = True
    ):
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ('plsr', 'cubist', 'gbrt', 'krr', 'svr')
            params: Custom hyperparameters (optional)
            include_scaler: Whether to include StandardScaler in pipeline
            
        Returns:
            Model or Pipeline instance
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get parameters
        model_params = self.DEFAULT_PARAMS[model_type].copy()
        if params:
            model_params.update(params)
        
        # Create base model
        if model_type == 'plsr':
            model = PLSRegression(
                n_components=min(model_params['n_components'], 50),
                scale=model_params['scale'],
                max_iter=model_params['max_iter']
            )
        
        elif model_type == 'cubist':
            # Using GradientBoosting as Cubist approximation
            # Cubist uses rule-based regression, similar to boosted trees
            model = GradientBoostingRegressor(
                n_estimators=model_params['n_estimators'],
                max_depth=model_params['max_depth'],
                learning_rate=model_params['learning_rate'],
                subsample=model_params['subsample'],
                random_state=model_params['random_state'],
                loss='squared_error'
            )
        
        elif model_type == 'gbrt':
            model = GradientBoostingRegressor(
                n_estimators=model_params['n_estimators'],
                max_depth=model_params['max_depth'],
                learning_rate=model_params['learning_rate'],
                subsample=model_params['subsample'],
                min_samples_split=model_params['min_samples_split'],
                random_state=model_params['random_state']
            )
        
        elif model_type == 'krr':
            model = KernelRidge(
                alpha=model_params['alpha'],
                kernel=model_params['kernel'],
                gamma=model_params['gamma']
            )
        
        elif model_type == 'svr':
            model = SVR(
                kernel=model_params['kernel'],
                C=model_params['C'],
                epsilon=model_params['epsilon'],
                gamma=model_params['gamma']
            )
        
        # Wrap in pipeline with scaler if needed
        if include_scaler and model_type not in ['plsr']:  # PLSR has its own scaling
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        
        return model
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a model type."""
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        info = self.MODELS[model_type].copy()
        info['default_params'] = self.DEFAULT_PARAMS[model_type].copy()
        return info
    
    def get_all_models(self) -> Dict[str, Dict]:
        """Get information about all available models."""
        return {k: self.get_model_info(k) for k in self.MODELS.keys()}
    
    def get_feature_importance(self, model, feature_names: list, X_train=None, y_train=None) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from a trained model.
        
        Args:
            model: Trained model or pipeline
            feature_names: List of feature names
            X_train: Training features (needed for permutation importance)
            y_train: Training targets (needed for permutation importance)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            # Handle pipelines
            actual_model = model
            if hasattr(model, 'named_steps'):
                actual_model = model.named_steps.get('model', model)
            
            importance = None
            importance_type = None
            
            # GradientBoosting models (GBRT, Cubist)
            if hasattr(actual_model, 'feature_importances_'):
                importance = actual_model.feature_importances_
                importance_type = 'native'
            
            # PLSR - use coefficient magnitudes
            elif hasattr(actual_model, 'coef_'):
                coef = actual_model.coef_
                if len(coef.shape) > 1:
                    importance = np.abs(coef).mean(axis=0)
                else:
                    importance = np.abs(coef)
                importance_type = 'coefficients'
            
            # SVR/KRR - use permutation importance
            elif X_train is not None and y_train is not None:
                from sklearn.inspection import permutation_importance
                
                # Calculate permutation importance (use fewer repeats for speed)
                perm_result = permutation_importance(
                    model, X_train, y_train, 
                    n_repeats=5, 
                    random_state=42,
                    n_jobs=-1,
                    scoring='r2'
                )
                importance = perm_result.importances_mean
                importance_type = 'permutation'
            
            if importance is not None:
                # Ensure proper shape
                if len(importance) != len(feature_names):
                    return None
                
                # Handle negative values (from permutation importance)
                importance = np.maximum(importance, 0)
                
                # Normalize to 0-1
                if importance.max() > 0:
                    importance = importance / importance.max()
                
                # Create dictionary
                return dict(zip(feature_names, importance))
            
            return None
            
        except Exception as e:
            return None


class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.params = ModelFactory.DEFAULT_PARAMS.get(model_type, {}).copy()
    
    def update(self, **kwargs):
        """Update parameters."""
        self.params.update(kwargs)
        return self
    
    def get(self, key: str, default=None):
        """Get a parameter value."""
        return self.params.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.params.copy()
