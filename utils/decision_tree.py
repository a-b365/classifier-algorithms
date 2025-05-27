# Standard library imports
import os

# Third-party imports 
import numpy as np
import pandas as pd

# Relative imports
from prettytable import PrettyTable
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score


if __name__ == "__main__":
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scalar", StandardScaler()),
        ("selector_i", VarianceThreshold(threshold=0.01)),
        ("selector_ii", SelectKBest(score_func=f_classif, k=100)),
        ("model", DecisionTreeClassifier())
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

    pipeline.fit(X_clean, y_train)
    y_train_pred = pipeline.predict(X_clean)
    y_test_pred = pipeline.predict(X_test)

    myTable = PrettyTable(["Accuracy", "AUC ROC", "Sensitivity", "F1-score"])
    myTable.add_divider()
    myTable.add_row([accuracy_score(y_train, y_train_pred), roc_auc_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), f1_score(y_train, y_train_pred)])
    myTable.add_divider()
    myTable.add_row([accuracy_score(y_test, y_test_pred), roc_auc_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred)])

    print(myTable)
    # mask = pipeline.fit(X_clean, y).named_steps["selector_i"].get_support(indices=True)
    # selected_features = X_clean.columns[mask]
    # print(f"Reduced from {X_clean.shape[1]} to {len(mask)} features.")
    

