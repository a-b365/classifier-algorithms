# Standard library imports
import os

# Third-party imports 
import numpy as np
import pandas as pd

# Relative imports
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer


if __name__ == "__main__":
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scalar", StandardScaler()),
        # ("selector_i", VarianceThreshold(threshold=0.01)),
        # ("selector_ii", SelectKBest(score_func=f_classif, k=100)),
        ("model", GradientBoostingClassifier(learning_rate=0.01, max_depth=3, min_samples_split=5, n_estimators=100, subsample=0.8))
    ])

    TRAIN_DATA = os.environ["ORIGINAL_DATA_PATH"] + "/train_set.csv"
    TEST_DATA = os.environ["ORIGINAL_DATA_PATH"] + "/test_set.csv"
    
    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)

    X_train = train_data.drop(columns=["ID", "CLASS"])
    y_train = train_data["CLASS"]
    X_test = test_data.drop(columns=["ID", "CLASS"])
    y_test = test_data["CLASS"]

    X_clean = X_train.replace([np.inf, -np.inf], np.nan)

    print(pipeline.fit(X_clean, y_train).score(X_test, y_test))
    # mask = pipeline.fit(X_clean, y).named_steps["selector_i"].get_support(indices=True)
    # selected_features = X_clean.columns[mask]
    # print(f"Reduced from {X_clean.shape[1]} to {len(mask)} features.")
    

