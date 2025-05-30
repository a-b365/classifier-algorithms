# Tabular-data classification

## Overview

This project implements a comprehensive machine learning pipeline designed to handle high-dimensional datasets with limited sample sizes. The primary challenge addressed is the "curse of dimensionality" where the number of features (3,238) significantly exceeds the number of samples (315).

## Problem Statement

- **Dataset**: Binary classification with 3,238 features and 315 samples
- **Challenge**: High dimensionality leading to overfitting and poor generalization
- **Goal**: Develop robust models that perform well on unseen test data
- **Models**: Logistic Regression, Random Forest, and Decision Tree classifiers

## Key Features

### 🔧 Robust Preprocessing Pipeline
- **Missing Value Handling**: Intelligent imputation using median values
- **Outlier Detection**: IQR-based outlier detection and clipping
- **Feature Filtering**: Removal of low-variance and highly correlated features
- **Normalization**: Robust scaling to handle outlier-resistant preprocessing

### 🎯 Advanced Feature Engineering
- **Multiple Selection Methods**: Mutual information and model-based selection
- **Power Transformation**: Yeo-Johnson transformation for normality

### 🤖 Model Training & Evaluation
- **Multiple Algorithms**: Logistic Regression, Random Forest, Decision Tree
- **Hyperparameter Tuning**: Grid and Randomized search with cross-validation
- **Class Imbalance Handling**: SMOTE Oversampling technique
- **Comprehensive Evaluation**: Accuracy, AUC-ROC, sensitivity, specificity, F1-score

### 📊 Visualization & Analysis
- **Model Comparison**: Side-by-side performance comparison
- **Confusion Matrices**: Visual representation of classification results
- **Feature Importance**: Analysis of most predictive features

## Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy imbalanced-learn
```

### Required Libraries
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- imbalanced-learn >= 0.8.0

## Usage

### Quick Start

```python
from ml_pipeline import main_pipeline

# Define your data paths
TRAIN_PATH = "path/to/train.csv"
TEST_PATH = "path/to/test.csv"

# Run the complete pipeline
best_model, results, trainer = main_pipeline(TRAIN_PATH, TEST_PATH)
```

### Custom Configuration

```python
from ml_pipeline import ModelTrainer, create_pipeline

# Initialize trainer
trainer = ModelTrainer(cv_folds=5, scoring='roc_auc')

# Create custom pipeline
pipeline = create_pipeline(
    model_type='logistic',
    use_pca=True,
    n_components=50,
    use_smote=False
)

# Train model
model = trainer.train_model(X_train, y_train, model_type='logistic')

# Evaluate
results = trainer.evaluate_model(model, X_test, y_test, 'Custom_Model')
```

## Data Format

### Expected CSV Structure
```
ID,feature_1,feature_2,...,feature_n,CLASS
1,0.123,0.456,...,0.789,0
2,0.234,0.567,...,0.890,1
...
```

### Requirements
- **ID Column**: Unique identifier for each sample
- **Feature Columns**: Numerical features (3,238 expected)
- **CLASS Column**: Binary target variable (0 or 1)
- **Missing Values**: Handled automatically
- **Infinite Values**: Automatically converted to NaN and imputed

## Methodology

### 1. Data Preprocessing
- **Infinite Value Handling**: Replace ±∞ with NaN
- **Missing Value Imputation**: Median-based imputation
- **Variance Filtering**: Remove low-variance features (threshold: 0.01)
- **Correlation Filtering**: Remove highly correlated features (threshold: 0.95)

### 2. Feature Engineering
- **Outlier Treatment**: IQR-based detection and clipping
- **Power Transformation**: Yeo-Johnson for normality
- **Scaling**: Robust scaling for outlier resistance
- **Feature Selection**: Multiple methods (mutual info, RFE, model-based)

### 3. Model Training
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Class Imbalance**: Balanced class weights
- **Regularization**: Strong regularization to prevent overfitting

### 4. Evaluation Metrics
- **Primary**: ROC-AUC score (handles class imbalance)
- **Secondary**: Accuracy, Precision, Recall, F1-score
- **Visualization**: Confusion matrices and performance plots

## Best Practices for High-Dimensional Data

### ✅ Recommended Approaches
1. **Aggressive Feature Selection**: Start with 10-50 features
2. **Strong Regularization**: Use L1/L2 penalties
3. **Cross-Validation**: Always use stratified CV
4. **Ensemble Methods**: Random Forest with max_features='sqrt'
5. **PCA**: Consider for linear dimensionality reduction

### ❌ Common Pitfalls to Avoid
1. **No Feature Selection**: Using all 3,238 features
2. **Data Leakage**: Applying transforms before train/test split
3. **Overfitting**: Complex models without regularization
4. **SMOTE Overuse**: Synthetic data may not help generalization
5. **Ignoring Class Balance**: Not accounting for imbalanced classes

## Results Interpretation

### Model Performance Expectations
- **Good Performance**: Accuracy > 0.75, AUC > 0.80
- **Acceptable Performance**: Accuracy > 0.65, AUC > 0.70
- **Poor Performance**: Accuracy < 0.60, AUC < 0.65

### Feature Importance Analysis
```python
# Get feature importance from trained Random Forest
rf_model = trainer.trained_models['rf_pca_False_smote_False']
feature_importance = rf_model.named_steps['classifier'].feature_importances_

# Plot top 20 features
plt