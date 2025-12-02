"""
Data Loader Module
==================
Handles loading of spectral data files with smart format detection.
Supports CSV, XLS, XLSX with automatic fallback for misnamed files.
"""

import pandas as pd
import numpy as np
import io
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and validation of spectral soil datasets."""
    
    SUPPORTED_FORMATS = ['.csv', '.xls', '.xlsx']
    MAX_FILE_SIZE_MB = 500
    
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.load_message = ""
    
    def load_file(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load a file with smart format detection.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Tuple of (DataFrame or None, status message)
        """
        if uploaded_file is None:
            return None, "No file uploaded"
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            return None, f"File too large: {file_size_mb:.1f}MB (max {self.MAX_FILE_SIZE_MB}MB)"
        
        filename = uploaded_file.name.lower()
        file_bytes = uploaded_file.getvalue()
        
        # Smart loading with fallback
        df = None
        methods_tried = []
        
        # Try based on extension first
        if filename.endswith('.csv'):
            df, msg = self._try_csv(file_bytes)
            methods_tried.append('CSV')
            if df is None:
                df, msg = self._try_excel(file_bytes)
                methods_tried.append('Excel')
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df, msg = self._try_excel(file_bytes)
            methods_tried.append('Excel')
            if df is None:
                df, msg = self._try_csv(file_bytes)
                methods_tried.append('CSV')
        else:
            # Unknown extension - try all
            df, msg = self._try_csv(file_bytes)
            methods_tried.append('CSV')
            if df is None:
                df, msg = self._try_excel(file_bytes)
                methods_tried.append('Excel')
        
        if df is not None:
            # Validate and clean data
            df = self._clean_data(df)
            self.data = df
            self.metadata = self._extract_metadata(df, uploaded_file.name)
            return df, f"✅ Successfully loaded {uploaded_file.name}"
        else:
            return None, f"❌ Could not load file. Tried: {', '.join(methods_tried)}"
    
    def _try_csv(self, file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], str]:
        """Try to load as CSV with various encodings."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding=encoding,
                        sep=sep,
                        engine='python'
                    )
                    if len(df.columns) > 1 and len(df) > 0:
                        return df, f"Loaded as CSV (encoding: {encoding}, separator: {repr(sep)})"
                except Exception:
                    continue
        return None, "CSV loading failed"
    
    def _try_excel(self, file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], str]:
        """Try to load as Excel file."""
        engines = ['openpyxl', 'xlrd']
        
        for engine in engines:
            try:
                df = pd.read_excel(io.BytesIO(file_bytes), engine=engine)
                if len(df.columns) > 1 and len(df) > 0:
                    return df, f"Loaded as Excel (engine: {engine})"
            except Exception:
                continue
        return None, "Excel loading failed"
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the dataframe."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Convert column names to strings
        df.columns = [str(col).strip() for col in df.columns]
        
        # Try to convert all columns to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _extract_metadata(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Extract metadata from the loaded dataframe."""
        metadata = {
            'filename': filename,
            'n_samples': len(df),
            'n_columns': len(df.columns),
            'n_features': len(df.columns) - 1,  # Assuming last column is target
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'duplicates': df.duplicated().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Detect spectral range if column names look like wavelengths
        numeric_cols = []
        for col in df.columns[:-1]:  # Exclude last column (target)
            try:
                val = float(str(col).replace('nm', '').replace('Band', '').strip())
                numeric_cols.append(val)
            except:
                pass
        
        if len(numeric_cols) > 10:
            metadata['spectral_range'] = f"{min(numeric_cols):.0f}-{max(numeric_cols):.0f} nm"
            metadata['spectral_resolution'] = f"~{(max(numeric_cols) - min(numeric_cols)) / len(numeric_cols):.1f} nm"
        else:
            metadata['spectral_range'] = "Not detected"
            metadata['spectral_resolution'] = "N/A"
        
        # Data quality score
        quality_score = 10
        if metadata['missing_percentage'] > 0:
            quality_score -= min(3, metadata['missing_percentage'])
        if metadata['duplicates'] > 0:
            quality_score -= min(2, metadata['duplicates'] / len(df) * 10)
        
        metadata['quality_score'] = max(0, quality_score)
        metadata['quality_label'] = self._get_quality_label(metadata['quality_score'])
        
        return metadata
    
    def _get_quality_label(self, score: float) -> str:
        """Get quality label based on score."""
        if score >= 9:
            return "Excellent"
        elif score >= 7:
            return "Good"
        elif score >= 5:
            return "Fair"
        else:
            return "Poor"
    
    def get_target_statistics(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Calculate statistics for the target column."""
        if target_col not in df.columns:
            return {}
        
        target = df[target_col].dropna()
        
        stats = {
            'mean': target.mean(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'median': target.median(),
            'range': target.max() - target.min(),
            'cv': (target.std() / target.mean()) * 100 if target.mean() != 0 else 0,
            'q1': target.quantile(0.25),
            'q3': target.quantile(0.75),
            'iqr': target.quantile(0.75) - target.quantile(0.25),
            'skewness': target.skew(),
            'kurtosis': target.kurtosis(),
            'n_samples': len(target)
        }
        
        # Detect outliers using IQR method
        q1, q3 = stats['q1'], stats['q3']
        iqr = stats['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = target[(target < lower_bound) | (target > upper_bound)]
        stats['n_outliers'] = len(outliers)
        stats['outlier_percentage'] = (len(outliers) / len(target)) * 100
        stats['outlier_indices'] = outliers.index.tolist()
        
        return stats
    
    def validate_for_training(self, df: pd.DataFrame, target_col: str) -> Tuple[bool, str]:
        """Validate if the data is suitable for training."""
        errors = []
        
        # Check minimum samples
        if len(df) < 30:
            errors.append(f"Too few samples: {len(df)} (minimum 30 recommended)")
        
        # Check if target column exists
        if target_col not in df.columns:
            errors.append(f"Target column '{target_col}' not found")
        
        # Check for all numeric data
        feature_cols = [col for col in df.columns if col != target_col]
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            errors.append(f"Dataset contains {missing} missing values")
        
        # Check for constant features
        constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]
        if constant_cols:
            errors.append(f"Found {len(constant_cols)} constant features (zero variance)")
        
        if errors:
            return False, "Validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        
        return True, "✅ Data validation passed"
