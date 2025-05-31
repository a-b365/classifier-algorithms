"""
Logistic Regression Machine Learning Pipeline for Binary Classification.

This module implements a comprehensive machine learning pipeline for binary
classification using logistic regression. The pipeline includes data preprocessing,
feature selection, hyperparameter tuning, and model evaluation.

Key Features:
- Robust data preprocessing with multiple filters
- Automated feature selection and engineering
- Hyperparameter optimization using RandomizedSearchCV
- Comprehensive model evaluation metrics
- Cross-validation for reliable performance estimation

Dependencies:
- scikit-learn
- pandas
- numpy
- imbalanced-learn
- prettytable

Author: Amir Bhattarai
Date: 2025-05-30
Version: 1.0
"""

# Standard library imports
import os
import sys
import warnings
from typing import Tuple, Dict, Any

# Third-party imports
import scipy
import numpy as np
import pandas as pd
from scipy import stats
from prettytable import PrettyTable

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import (
    SelectKBest, 
    mutual_info_classif, 
    VarianceThreshold, 
    SelectFromModel
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Local imports
try:
    from metrics import evaluate_metrics
    from filters import CorrelationFilter, OutlierFilter, SkewnessFilter
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    print("Please ensure 'metrics.py' and 'filters.py' are in the same directory.")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class LogisticRegressionPipeline:
    """
    A comprehensive logistic regression pipeline for binary classification.
    
    This class encapsulates the entire machine learning workflow including
    data preprocessing, feature engineering, model training, hyperparameter
    tuning, and evaluation.
    
    Attributes
    ----------
    pipeline : imblearn.pipeline.Pipeline
        The complete preprocessing and modeling pipeline.
    best_params_ : dict
        Best hyperparameters found during random search.
    cv_scores_ : dict
        Cross-validation scores for different metrics.
    
    Examples
    --------
    >>> pipeline = LogisticRegressionPipeline()
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
    >>> pipeline.evaluate(X_test, y_test)
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the LogisticRegressionPipeline.
        
        Parameters
        ----------
        random_state : int, default=42
            Random state for reproducibility.
        """
        self.random_state = random_state
        self.pipeline = None
        self.best_params_ = None
        self.cv_scores_ = None
        
    def _create_pipeline(self, feature_columns):
        """
        Create the preprocessing and modeling pipeline.
        
        Parameters
        ----------
        feature_columns : list
            List of feature column names.
        
        Returns
        -------
        pipeline : imblearn.pipeline.Pipeline
            Complete preprocessing and modeling pipeline.
        """
        pipeline_steps = [
            # Data imputation - handle missing values
            ("imputer", SimpleImputer(
                missing_values=np.nan, 
                strategy="median"
            )),
            
            # Outlier handling - robust outlier treatment
            # ('outlier_filter', OutlierFilter(feature_columns)),
            
            # Feature scaling - robust to outliers
            ("scaler", RobustScaler()),
            
            # Remove highly skewed features
            # ('skewness_filter', SkewnessFilter(threshold=1.0)),
            
            # Remove low variance features
            # ("variance_threshold", VarianceThreshold(threshold=0.01)),
            
            # Remove highly correlated features
            # ('correlation_filter', CorrelationFilter(threshold=0.95)),
            
            # Handle class imbalance
            # ('smote', SMOTE(random_state=self.random_state)),
            
            # Feature selection based on mutual information
            # ('feature_selection', SelectKBest(
            #     score_func=mutual_info_classif, 
            #     k=50
            # )),
            
            # Final model
            ("classifier", LogisticRegression(
                penalty='l2',
                solver='saga',
                max_iter=4085,
                C=0.001,
                random_state=self.random_state,
                class_weight="balanced"
            ))
        ]
        
        return ImbPipeline(pipeline_steps)
    
    def fit(self, X_train, y_train):
        """
        Fit the pipeline to training data.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.
        y_train : array-like of shape (n_samples,)
            Training target values.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        print("Creating and fitting the pipeline...")
        
        # Create pipeline
        self.pipeline = self._create_pipeline(X_train.columns.tolist())
        
        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        
        print("Pipeline fitted successfully!")
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted pipeline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        
        Returns
        -------
        predictions : array-like of shape (n_samples,)
            Predicted class labels.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before making predictions.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted pipeline.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict probabilities for.
        
        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before making predictions.")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the model performance on training and test sets.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.
        y_train : array-like of shape (n_samples,)
            Training target values.
        X_test : array-like of shape (n_samples, n_features)
            Test features.
        y_test : array-like of shape (n_samples,)
            Test target values.
        
        Returns
        -------
        results : dict
            Dictionary containing evaluation metrics for both sets.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before evaluation.")
        
        # Make predictions
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        
        # Evaluate metrics
        train_metrics = evaluate_metrics(y_train, y_train_pred)
        test_metrics = evaluate_metrics(y_test, y_test_pred)
        
        # Create results table
        table = PrettyTable([
            "Dataset", "Accuracy", "AUC ROC", "Sensitivity", 
            "Specificity", "F1-score"
        ])
        table.add_row(["Training"] + list(train_metrics))
        table.add_row(["Test"] + list(test_metrics))
        
        print("\nModel Evaluation Results:")
        print(table)
        
        return {
            'train': dict(zip(['accuracy', 'auc', 'sensitivity', 'specificity', 'f1'], train_metrics)),
            'test': dict(zip(['accuracy', 'auc', 'sensitivity', 'specificity', 'f1'], test_metrics))
        }
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5, n_iter=5):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.
        y_train : array-like of shape (n_samples,)
            Training target values.
        cv_folds : int, default=5
            Number of cross-validation folds.
        n_iter : int, default=5
            Number of parameter combinations to try.
        
        Returns
        -------
        self : object
            Returns the instance itself with optimized parameters.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before hyperparameter tuning.")
        
        print("Starting hyperparameter tuning...")
        
        # Define parameter grid for tuning
        param_grid = {
            'classifier__max_iter':stats.randint(2500, 5000),
            'classifier__C': [0.001, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__class_weight': [None, "balanced"]
        }
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )
        
        random_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params_ = random_search.best_params_
        
        print("Hyperparameter tuning completed!")
        print(f"Best parameters: {self.best_params_}")
        print(f"Best cross-validation F1 score: {random_search.best_score_:.4f}")
        
        # Update pipeline with best parameters
        self.pipeline.set_params(**self.best_params_)
        self.pipeline.fit(X_train, y_train)
        
        return self


def load_and_preprocess_data(train_path, test_path, blinded_path):
    """
    Load and preprocess training and test datasets.
    
    Parameters
    ----------
    train_path : str
        Path to the training dataset CSV file.
    test_path : str
        Path to the test dataset CSV file.
    blinded_path : str
        Path to the blinded test dataset CSV file.
    
    Returns
    -------
    X_train, y_train, X_test, y_test, X_blinded : tuple
        Preprocessed training and test features and targets.
    """
    try:
        # Load datasets
        print("Loading datasets...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        blinded_data = pd.read_csv(blinded_path)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Blinded data shape: {blinded_data.shape}")
        
        # Separate features and targets
        X_train = train_data.drop(columns=["CLASS"])
        y_train = train_data["CLASS"]
        X_test = test_data.drop(columns=["CLASS"])
        y_test = test_data["CLASS"]
        X_blinded = blinded_data
        
        # Handle infinite values
        print("Handling infinite values...")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_blinded = X_blinded.replace([np.inf, -np.inf], np.nan)
        
        print("Data preprocessing completed!")
        return X_train, y_train, X_test, y_test, X_blinded
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def main():
    """
    Main function to execute the complete machine learning pipeline.
    
    This function orchestrates the entire workflow:
    1. Load and preprocess data
    2. Create and fit the pipeline
    3. Evaluate initial performance
    4. Perform hyperparameter tuning
    5. Evaluate final performance
    """
    try:
        # Get data paths from environment variables
        data_path = os.environ.get("DATA_PATH")
        results_path = os.environ.get("RESULTS_PATH")
        if data_path is None:
            raise ValueError("DATA_PATH environment variable not set")
        
        train_path = os.path.join(data_path, "train_set.csv")
        test_path = os.path.join(data_path, "test_set.csv")
        blinded_path = os.path.join(data_path, "blinded_test_set.csv")
        
        # Load and preprocess data
        X_train, y_train, X_test, y_test, X_blinded = load_and_preprocess_data(
            train_path, test_path, blinded_path
        )
        
        # Initialize and fit the pipeline
        print("\n" + "="*60)
        print("INITIAL MODEL TRAINING")
        print("="*60)
        
        pipeline = LogisticRegressionPipeline(random_state=42)
        pipeline.fit(X_train.drop(columns="ID"), y_train)

        # Predict class probabilities for each dataset
        datasets = {
            'train': X_train,
            'test': X_test, 
            'blinded': X_blinded
        }

        for dataset_name, dataset in datasets.items():
            # Extract IDs before prediction
            ids = dataset["ID"]
            
            # Get predictions without ID column
            class_probabilities = pipeline.predict_proba(dataset.drop(columns="ID"))
            
            # Create DataFrame with class probabilities
            proba_df = pd.DataFrame(class_probabilities)
            
            # Add ID as the first column
            proba_df.insert(0, 'ID', ids.values)
            
            # Save to CSV with descriptive filename
            output_path = os.path.join(results_path, f"proba_{dataset_name}.csv")
            proba_df.to_csv(output_path, index=False)

        # Initial evaluation
        print("\nInitial Model Performance:")
        # initial_results = pipeline.evaluate(X_train.drop(columns="ID"), y_train, X_test.drop(columns="ID"), y_test)
        
        # Hyperparameter tuning
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)

    #     pipeline.hyperparameter_tuning(X_train.drop(columns="ID"), y_train, cv_folds=5, n_iter=20)
        
    #     # Final evaluation
    #     print("\n" + "="*60)
    #     print("FINAL MODEL PERFORMANCE")
    #     print("="*60)
        
    #     final_results = pipeline.evaluate(X_train.drop(columns="ID"), y_train, X_test.drop(columns="ID"), y_test)
        
    #     # Performance comparison
    #     print("\n" + "="*60)
    #     print("PERFORMANCE COMPARISON")
    #     print("="*60)
        
    #     comparison_table = PrettyTable([
    #         "Model", "Test Accuracy", "Test F1-Score", "Test AUC"
    #     ])
    #     comparison_table.add_row([
    #         "Initial",
    #         f"{initial_results['test']['accuracy']:.4f}",
    #         f"{initial_results['test']['f1']:.4f}",
    #         f"{initial_results['test']['auc']:.4f}"
    #     ])
    #     comparison_table.add_row([
    #         "Tuned",
    #         f"{final_results['test']['accuracy']:.4f}",
    #         f"{final_results['test']['f1']:.4f}",
    #         f"{final_results['test']['auc']:.4f}"
    #     ])
    #     print(comparison_table)
        
    #     print("\nPipeline execution completed successfully!")
        
    except KeyError as e:
        print(f"Environment variable error: {e}")
        print("Please set the DATA_PATH environment variable.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check that the data files exist at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()