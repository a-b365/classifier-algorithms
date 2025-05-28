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