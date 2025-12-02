"""
Training Orchestrator Module
============================
Handles automated training of all model combinations.
Manages training progress, cross-validation, and results collection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

from .preprocessing import SpectralPreprocessor
from .models import ModelFactory
from .evaluation import ModelEvaluator


class TrainingOrchestrator:
    """Orchestrates the automated training pipeline."""
    
    def __init__(self, models_dir: str = "models"):
        self.preprocessor = SpectralPreprocessor()
        self.model_factory = ModelFactory()
        self.evaluator = ModelEvaluator()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Training state
        self.results = []
        self.trained_models = {}
        self.training_start_time = None
        self.current_run_id = None
    
    def train_all_combinations(
        self,
        df: pd.DataFrame,
        target_col: str,
        preprocessing_methods: List[str],
        model_types: List[str],
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        custom_params: Optional[Dict[str, Dict]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train all combinations of preprocessing and models.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            preprocessing_methods: List of preprocessing methods
            model_types: List of model types
            test_size: Fraction for test set
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            custom_params: Custom parameters per model type
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary with training results
        """
        self.training_start_time = time.time()
        self.current_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.results = []
        self.trained_models = {}
        
        # Create run directory
        run_dir = self.models_dir / self.current_run_id
        run_dir.mkdir(exist_ok=True)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Calculate total combinations
        total_combinations = len(preprocessing_methods) * len(model_types)
        current_combination = 0
        
        # Train each combination
        for prep_method in preprocessing_methods:
            # Preprocess data
            X, y, feature_names = self.preprocessor.preprocess_dataframe(
                df, target_col, prep_method
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            for model_type in model_types:
                current_combination += 1
                
                # Update progress
                if progress_callback:
                    progress_callback({
                        'current': current_combination,
                        'total': total_combinations,
                        'preprocessing': prep_method,
                        'model': model_type,
                        'status': 'training'
                    })
                
                try:
                    # Get custom parameters if provided
                    params = None
                    if custom_params and model_type in custom_params:
                        params = custom_params[model_type]
                    
                    # Create and train model
                    model = self.model_factory.create_model(model_type, params)
                    
                    train_start = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - train_start
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Evaluate
                    train_metrics = self.evaluator.calculate_metrics(y_train, y_pred_train)
                    test_metrics = self.evaluator.calculate_metrics(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X, y, 
                        cv=cv_folds, 
                        scoring='r2'
                    )
                    
                    # Feature importance
                    feature_importance = self.model_factory.get_feature_importance(
                        model, feature_names, X_train, y_train
                    )
                    
                    # Create result entry
                    result = {
                        'run_id': self.current_run_id,
                        'preprocessing': prep_method,
                        'model_type': model_type,
                        'model_name': f"{model_type.upper()}_{prep_method}",
                        'train_r2': train_metrics['r2'],
                        'test_r2': test_metrics['r2'],
                        'train_rmse': train_metrics['rmse'],
                        'test_rmse': test_metrics['rmse'],
                        'train_mae': train_metrics['mae'],
                        'test_mae': test_metrics['mae'],
                        'rpd': test_metrics['rpd'],
                        'correlation': test_metrics['correlation'],
                        'bias': test_metrics['bias'],
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'cv_scores': cv_scores.tolist(),
                        'train_time': train_time,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'feature_importance': feature_importance,
                        'y_test': y_test.tolist(),
                        'y_pred_test': y_pred_test.tolist(),
                        'y_train': y_train.tolist(),
                        'y_pred_train': y_pred_train.tolist(),
                        'status': 'success',
                        'error': None
                    }
                    
                    # Save model
                    model_filename = f"{prep_method}_{model_type}.pkl"
                    model_path = run_dir / model_filename
                    joblib.dump(model, model_path)
                    result['model_path'] = str(model_path)
                    
                    # Store trained model
                    model_key = f"{prep_method}_{model_type}"
                    self.trained_models[model_key] = model
                    
                except Exception as e:
                    result = {
                        'run_id': self.current_run_id,
                        'preprocessing': prep_method,
                        'model_type': model_type,
                        'model_name': f"{model_type.upper()}_{prep_method}",
                        'test_r2': 0,
                        'test_rmse': float('inf'),
                        'status': 'failed',
                        'error': str(e)
                    }
                
                self.results.append(result)
                
                # Update progress with result
                if progress_callback:
                    progress_callback({
                        'current': current_combination,
                        'total': total_combinations,
                        'preprocessing': prep_method,
                        'model': model_type,
                        'status': 'completed',
                        'result': result
                    })
        
        # Calculate summary
        total_time = time.time() - self.training_start_time
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['test_r2'])
        else:
            best_result = None
        
        summary = {
            'run_id': self.current_run_id,
            'total_models': len(self.results),
            'successful_models': len(successful_results),
            'failed_models': len(self.results) - len(successful_results),
            'total_time': total_time,
            'avg_time_per_model': total_time / len(self.results) if self.results else 0,
            'best_model': best_result,
            'results': self.results,
            'run_dir': str(run_dir)
        }
        
        # Save metadata
        self._save_run_metadata(run_dir, summary, df, target_col)
        
        return summary
    
    def _save_run_metadata(
        self, 
        run_dir: Path, 
        summary: Dict, 
        df: pd.DataFrame, 
        target_col: str
    ):
        """Save run metadata to disk."""
        metadata = {
            'run_id': summary['run_id'],
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,
            'target_column': target_col,
            'total_models': summary['total_models'],
            'successful_models': summary['successful_models'],
            'total_time': summary['total_time'],
            'best_r2': summary['best_model']['test_r2'] if summary['best_model'] else None,
            'best_model': summary['best_model']['model_name'] if summary['best_model'] else None
        }
        
        # Save as pickle
        joblib.dump(metadata, run_dir / 'metadata.pkl')
        
        # Save results as CSV
        results_df = pd.DataFrame(summary['results'])
        results_df.to_csv(run_dir / 'results.csv', index=False)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        # Select relevant columns for display
        cols = [
            'preprocessing', 'model_type', 'test_r2', 'test_rmse', 
            'rpd', 'test_mae', 'correlation', 'bias', 
            'cv_mean', 'cv_std', 'train_time', 'status'
        ]
        
        df = pd.DataFrame(self.results)
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols]
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get ranked leaderboard of models."""
        df = self.get_results_dataframe()
        if df.empty:
            return df
        
        # Filter successful models
        df = df[df['status'] == 'success'].copy()
        
        # Sort by R²
        df = df.sort_values('test_r2', ascending=False).reset_index(drop=True)
        
        # Add rank
        df.insert(0, 'rank', range(1, len(df) + 1))
        
        return df
    
    def get_model_by_name(self, model_name: str):
        """Get a trained model by name."""
        for key, model in self.trained_models.items():
            if model_name in key or key in model_name:
                return model
        return None
    
    def load_run(self, run_dir: str) -> Dict[str, Any]:
        """Load a previous training run."""
        run_path = Path(run_dir)
        
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Load metadata
        metadata_path = run_path / 'metadata.pkl'
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
        else:
            metadata = {}
        
        # Load results
        results_path = run_path / 'results.csv'
        if results_path.exists():
            results_df = pd.read_csv(results_path)
            self.results = results_df.to_dict('records')
        
        # Load models
        self.trained_models = {}
        for model_file in run_path.glob('*.pkl'):
            if model_file.name != 'metadata.pkl':
                model_key = model_file.stem
                self.trained_models[model_key] = joblib.load(model_file)
        
        return {
            'metadata': metadata,
            'results': self.results,
            'models': list(self.trained_models.keys())
        }
