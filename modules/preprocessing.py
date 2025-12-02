"""
Spectral Preprocessing Module
==============================
Implements spectral preprocessing methods for soil spectroscopy.
- Reflectance (raw data)
- Absorbance (log10(1/R) transformation)
- Continuum Removal (convex hull normalization)
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SpectralPreprocessor:
    """Handles spectral data preprocessing for soil analysis."""
    
    METHODS = {
        'reflectance': 'Reflectance (Raw)',
        'absorbance': 'Absorbance (log₁₀(1/R))',
        'continuum_removal': 'Continuum Removal'
    }
    
    def __init__(self):
        self.wavelengths = None
        self.preprocessing_history = []
    
    def preprocess(self, X: np.ndarray, method: str) -> np.ndarray:
        """
        Apply preprocessing to spectral data.
        
        Args:
            X: Spectral data (n_samples, n_bands)
            method: Preprocessing method ('reflectance', 'absorbance', 'continuum_removal')
            
        Returns:
            Preprocessed spectral data
        """
        X = np.array(X, dtype=float)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        if method == 'reflectance':
            return self._reflectance(X)
        elif method == 'absorbance':
            return self._absorbance(X)
        elif method == 'continuum_removal':
            return self._continuum_removal(X)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def _reflectance(self, X: np.ndarray) -> np.ndarray:
        """
        Reflectance - return raw spectral data.
        This is the baseline preprocessing (no transformation).
        """
        return X.copy()
    
    def _absorbance(self, X: np.ndarray) -> np.ndarray:
        """
        Absorbance transformation: A = log10(1/R)
        
        Converts reflectance to apparent absorbance, which has
        a more linear relationship with concentration (Beer-Lambert law).
        """
        # Avoid division by zero and log of zero
        X_safe = np.clip(X, 1e-10, 1.0)
        
        # Calculate absorbance
        absorbance = np.log10(1.0 / X_safe)
        
        # Handle any remaining infinities
        absorbance = np.nan_to_num(absorbance, nan=0.0, posinf=0.0, neginf=0.0)
        
        return absorbance
    
    def _continuum_removal(self, X: np.ndarray) -> np.ndarray:
        """
        Continuum Removal - normalizes spectra using convex hull.
        
        This method:
        1. Fits a convex hull to the upper envelope of the spectrum
        2. Divides the spectrum by the continuum line
        3. Highlights absorption features
        """
        n_samples, n_bands = X.shape
        X_cr = np.zeros_like(X)
        
        for i in range(n_samples):
            spectrum = X[i, :]
            X_cr[i, :] = self._continuum_remove_single(spectrum)
        
        return X_cr
    
    def _continuum_remove_single(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply continuum removal to a single spectrum."""
        n_bands = len(spectrum)
        wavelengths = np.arange(n_bands)
        
        # Handle edge cases
        if np.all(spectrum == 0) or np.all(np.isnan(spectrum)):
            return spectrum
        
        # Ensure positive values for convex hull
        spectrum = np.maximum(spectrum, 1e-10)
        
        try:
            # Create points for convex hull (wavelength, reflectance)
            points = np.column_stack((wavelengths, spectrum))
            
            # Add boundary points to ensure proper hull
            min_y = np.min(spectrum) * 0.9
            points_extended = np.vstack([
                points,
                [0, min_y],
                [n_bands - 1, min_y]
            ])
            
            # Compute convex hull
            hull = ConvexHull(points_extended)
            
            # Get upper envelope (continuum)
            hull_vertices = hull.vertices
            hull_points = points_extended[hull_vertices]
            
            # Filter to get only upper envelope points
            upper_points = hull_points[hull_points[:, 1] >= min_y + 0.01]
            
            if len(upper_points) < 2:
                # Fallback: use linear interpolation between endpoints
                continuum = np.linspace(spectrum[0], spectrum[-1], n_bands)
            else:
                # Sort by wavelength
                upper_points = upper_points[np.argsort(upper_points[:, 0])]
                
                # Interpolate continuum
                f = interp1d(
                    upper_points[:, 0], 
                    upper_points[:, 1],
                    kind='linear',
                    fill_value='extrapolate'
                )
                continuum = f(wavelengths)
            
            # Ensure continuum is at least as large as spectrum
            continuum = np.maximum(continuum, spectrum * 0.999)
            
            # Divide spectrum by continuum
            cr_spectrum = spectrum / continuum
            
            # Clip to valid range
            cr_spectrum = np.clip(cr_spectrum, 0, 1)
            
        except Exception:
            # Fallback: simple normalization
            cr_spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        
        return cr_spectrum
    
    def get_method_description(self, method: str) -> str:
        """Get description of preprocessing method."""
        descriptions = {
            'reflectance': """
**Reflectance (Raw Data)**
- No transformation applied
- Uses original spectral measurements
- Suitable for baseline comparison
- Preserves absolute reflectance values
            """,
            'absorbance': """
**Absorbance Transformation**
- Formula: A = log₁₀(1/R)
- Converts reflectance to apparent absorbance
- Linear relationship with concentration (Beer-Lambert law)
- Enhances subtle absorption features
            """,
            'continuum_removal': """
**Continuum Removal**
- Normalizes spectra using convex hull envelope
- Highlights absorption features
- Removes background reflectance variations
- Best for identifying specific absorption bands
            """
        }
        return descriptions.get(method, "No description available")
    
    def preprocess_dataframe(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        method: str
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Preprocess a dataframe, separating features and target.
        
        Args:
            df: DataFrame with spectral data
            target_col: Name of target column
            method: Preprocessing method
            
        Returns:
            Tuple of (X_preprocessed, y, feature_names)
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Apply preprocessing
        X_preprocessed = self.preprocess(X, method)
        
        return X_preprocessed, y, feature_cols
