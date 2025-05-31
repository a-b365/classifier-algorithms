# Tabular-data classification

## Overview

This project implements a comprehensive machine learning pipeline designed to handle a high-dimensional datasets with limited sample size. The primary challenge addressed is the "curse of dimensionality" where the number of features (3238) significantly exceeds the number of samples (315).

---

## Features

- **Robust Processing Pipeline**: Missing value handling, outlier detection, feature filtering, and scaling
- **Advanced Feature Engineering**: Feature selection and power transformation
- **Model Training & Evaluation**: Multiple algorithms with tuning and comprehensive metrics
- **Performance Analysis**: Side-by-side performance comparison

---

## Technical Approach

**Core Methods:**
- Median imputation and IQR-based outlier handling (feature-wise)
- Normalize skewed features, remove near-constant and highly correlated features
- Feature selection based on mutual information
- Hyperparameter tuning and Stratified Cross Validation

**Stack:**
- `scikit-learn`, `scipy`, `imblearn`
- Visualization: `matplotlib`, `seaborn`
- Helper libs: `numpy`, `pandas`

---

## Project Structure

```
classifier-algorithms/
├── utils/
│   ├── decision_tree.py           # Decision Tree Classifier
│   ├── filters.py                 # Feature Selectors
│   ├── logistic_regression.py     # Logistic Regression Classifier
│   ├── metrics.py                 # Evaluation Metrics
│   ├── plots.py                   # Matplotlib & Seaborn helpers
│   └── random_forest.py           # Random Forest Classifier
├── notebooks/
│   └── notebook.ipynb             # Interactive Jupyter notebook
├── results
│   └── *.csv                      # Class probabilties for each dataset 
├── docs/
│   └── report.pdf                 # Methodology and technical documentation
├── requirements.txt               # Python dependencies
├── env.ps1                        # Windows setup script
└── readme.md                      # Project overview and instructions
```

---

## 🚀 Setup Instructions

### 🔧 Prerequisites

- Python 3.8+
- Git (to clone the repo)
- PowerShell (for Windows users)

### 💻 Installation

1. Clone the repository

    ```bash
    git clone https://github.com/a-b365/classifier-algorithms.git
    cd classifier-algorithms
    ```

2. Run the powershell script to add the environment variables:

    ```powershell
    .\env.ps1
    ```

3. Create and activate a virtual environment:

  - On macOS/Linux:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

  - On Windows:
    ```powershell
    .\venv\Scripts\activate
    ```

4. Install Dependencies

    Run:
    ```
    pip install -r requirements.txt
    ```
---

## Data Format

### CSV Structure
```
ID,feature_1,feature_2,...,feature_3238,CLASS
ID_1,18281.541667,18432.0,...,0.061710,0
ID_2,20010.083333,20100.0,...,0.090548,1
...
```

---

## Example Usage

```python

    from logisitic_regression import LogisticRegressionPipeline

    if name == "__main__":
      pipeline = LogisticRegressionPipeline()
      pipeline.fit(X_train, y_train)
      predictions = pipeline.predict(X_test)
      pipeline.evaluate(X_test, y_test)
```
---

## Outputs

- **Model Evaluation**: Tabulate accuracy, AUC ROC, sensitivity, specificity, F1-score
- **Best Parameters**: Provides appropriate parameters based on hyperparameter tuning
- **Best CV Score**: Provides the best score obtained during cross-valdiation.
- **Class Probabilities**: Class probabilities for each dataset in `results/`
---

## Methodology

**Preprocessing:**
- Exploratory Data Analysis
- Infinite Handling
- Median Imputation

**Feature Engineering**
- Outlier Treatment
- Power Transformation (optional)
- Robust Scaling
- Skewness Filtering (optional)
- Variance Filtering (optional)
- Correlation Filtering
- SMOTE Oversampling
- Feature Selection

**Model Training and Evaluation**
- StratifiedKFold Cross Validation
- RandomizedSearch Hyperparameter Tuning
- Performance Metrics

---

## Usage

- Run the desired Python module directly from the command line.
- This will run one classifier at a time:
  - Logistic Regression: Produces all classification metrics and the comparison
  - Decision Tree: Generates initial performance measures and final performance
  - Random Forest: It produces the same results as other classifiers
- Note: The results of all classifiers are in similar format.

---

## Notes

  - A detailed project report is available in the docs/ folder
  - Run the Jupyter notebook inside the notebooks/ folder for an interactive walkthrough of the implementation
  - Blinded test set results can be found in the results/

---
