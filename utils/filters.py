"""
Custom data preprocessing filters for machine learning pipelines.

This module contains custom transformer classes that follow scikit-learn's
BaseEstimator and TransformerMixin interfaces for data preprocessing tasks
including correlation filtering, skewness filtering, and outlier handling.

Classes:
    CorrelationFilter: Removes highly correlated features
    SkewnessFilter: Removes features with high skewness
    OutlierFilter: Handles outliers using IQR-based methods

Author: Amir Bhattarai
Date: 2025-05-30
Version: 1.0
"""

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with high correlation to reduce multicollinearity.
    
    This transformer identifies and removes features that have a correlation
    coefficient above the specified threshold with other features.
    
    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold above which features will be removed.
        Must be between 0 and 1.
    
    Attributes
    ----------
    to_drop_ : list
        List of feature names to be dropped after fitting.
    threshold : float
        The correlation threshold used for filtering.
    
    Examples
    --------
    >>> from filters import CorrelationFilter
    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.1, 3.1], 'C': [4, 5, 6]})
    >>> filter = CorrelationFilter(threshold=0.9)
    >>> X_filtered = filter.fit_transform(X)
    """
    
    def __init__(self, threshold=0.95):
        """
        Initialize the CorrelationFilter.
        
        Parameters
        ----------
        threshold : float, default=0.95
            Correlation threshold for feature removal.
        """
        self.threshold = threshold
        self.to_drop_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer by identifying highly correlated features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored in this transformer).
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Compute correlation matrix
        corr_matrix = pd.DataFrame(X).corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        self.to_drop_ = [
            column for column in upper.columns 
            if any(upper[column] > self.threshold)
        ]
        
        return self
    
    def transform(self, X):
        """
        Transform the data by removing highly correlated features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with correlated features removed.
        """
        return pd.DataFrame(X).drop(columns=self.to_drop_, axis=1)


class SkewnessFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with high skewness to improve model performance.
    
    This transformer identifies and removes features that have absolute
    skewness values above the specified threshold.
    
    Parameters
    ----------
    threshold : float, default=0.75
        Skewness threshold above which features will be removed.
    
    Attributes
    ----------
    to_drop_ : list
        List of feature names to be dropped after fitting.
    threshold : float
        The skewness threshold used for filtering.
    
    Examples
    --------
    >>> from filters import SkewnessFilter
    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 1, 1, 100, 200]})
    >>> filter = SkewnessFilter(threshold=1.0)
    >>> X_filtered = filter.fit_transform(X)
    """
    
    def __init__(self, threshold=0.75):
        """
        Initialize the SkewnessFilter.
        
        Parameters
        ----------
        threshold : float, default=0.75
            Skewness threshold for feature removal.
        """
        self.threshold = threshold
        self.to_drop_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer by identifying highly skewed features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored in this transformer).
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Calculate skewness of features
        skewness = pd.DataFrame(X).skew()
        
        # Find features with absolute skewness above threshold
        self.to_drop_ = skewness[
            abs(skewness) > self.threshold
        ].index.tolist()
        
        return self
    
    def transform(self, X):
        """
        Transform the data by removing highly skewed features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with skewed features removed.
        """
        return pd.DataFrame(X).drop(columns=self.to_drop_, axis=1)


class OutlierFilter(BaseEstimator, TransformerMixin):
    """
    Handle outliers using IQR-based method with median imputation and clipping.
    
    This transformer handles outliers in two steps:
    1. Replace outliers with median values
    2. Apply conservative winsorization by clipping extreme values
    
    Parameters
    ----------
    columns : list
        List of column names to process for outlier handling.
    
    Attributes
    ----------
    columns_ : list
        Stored column names for transformation.
    
    Examples
    --------
    >>> from filters import OutlierFilter
    >>> import pandas as pd
    >>> X = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [1, 2, 3, 4]})
    >>> filter = OutlierFilter(columns=['A', 'B'])
    >>> X_filtered = filter.fit_transform(X)
    """
    
    def __init__(self, columns):
        """
        Initialize the OutlierFilter.
        
        Parameters
        ----------
        columns : list
            List of column names to process.
        """
        self.columns = columns
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no fitting required for this transformer).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored in this transformer).
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.columns_ = self.columns
        return self
    
    def transform(self, X):
        """
        Transform the data by handling outliers using IQR method.
        
        The transformation process:
        1. Calculate Q1, Q3, and IQR for each feature
        2. Define outlier bounds using 1.5 * IQR rule
        3. Replace outliers with median values
        4. Recalculate bounds and apply clipping for conservative winsorization
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with outliers handled.
        """
        X_clean = pd.DataFrame(X, columns=self.columns_)
        
        for column in X_clean.columns:
            # Step 1: Impute outliers with median
            Q1 = X_clean[column].quantile(0.25)
            Q3 = X_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers and replace with median
            outliers_mask = (
                (X_clean[column] < lower_bound) | 
                (X_clean[column] > upper_bound)
            )
            median = X_clean[column].median()
            X_clean.loc[outliers_mask, column] = median
            
            # Step 2: Recalculate bounds and apply conservative clipping
            Q1 = X_clean[column].quantile(0.25)
            Q3 = X_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Apply winsorization (clipping)
            X_clean[column] = np.clip(
                X_clean[column], lower_bound, upper_bound
            )
        
        return X_clean