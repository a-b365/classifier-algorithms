# Third library imports
import pandas as pd
import numpy as np

# Relative imports
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Compute Correlation Matrix
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_, axis=1)
    
class SkewnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Calculate skewness of features
        skewness = pd.DataFrame(X).skew()
        # Find features with skewness > threshold
        self.to_drop_ = skewness[abs(skewness) > self.threshold].index.tolist()
        return self

    def transform(self, X):
        # Drop the skewed features
        return pd.DataFrame(X).drop(columns=self.to_drop_, axis=1)


class OutlierFilter(BaseEstimator, TransformerMixin):
    def __init__(self, impute_first=True):
        self.impute_first = impute_first

    def fit(self, X, y=None):
        # No fitting required for this transformer
        return self

    def transform(self, X):
        X_clean = X.copy()
        
        for column in X_clean.columns:
            # Step 1: Impute Outliers with Median
            Q1 = X_clean[column].quantile(0.25)
            Q3 = X_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_mask = (X_clean[column] < lower_bound) | (X_clean[column] > upper_bound)
            median = X_clean[column].median()
            X_clean.loc[outliers_mask.to_list(), column] = median

            # Step 2: Recalculate bounds and Clip (for conservative winsorization)
            Q1 = X_clean[column].quantile(0.25)
            Q3 = X_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            X_clean[column] = np.clip(X_clean[column], lower_bound, upper_bound)

        return X_clean
