# Standard library imports
import os

# Third-party imports 
import numpy as np
import pandas as pd

# Relative imports
from prettytable import PrettyTable
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


if __name__ == "__main__":
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scaler", StandardScaler()),
        ("selector_i", VarianceThreshold(threshold=0.01)),
        # ("selector_ii", SelectKBest(score_func=f_classif, k=100)),
        ("model", DecisionTreeClassifier(class_weight="balanced"))
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

    print("Metrics before hyperparameter tuning")
    pipeline.fit(X_clean, y_train)
    y_train_pred = pipeline.predict(X_clean)
    y_test_pred = pipeline.predict(X_test)

    myTable = PrettyTable(["Accuracy", "AUC ROC", "Sensitivity", "F1-score"])
    myTable.add_divider()
    myTable.add_row([accuracy_score(y_train, y_train_pred), roc_auc_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), f1_score(y_train, y_train_pred)])
    myTable.add_divider()
    myTable.add_row([accuracy_score(y_test, y_test_pred), roc_auc_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred)])
    print(myTable)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {
        'model__criterion': ['gini', 'entropy', 'log_loss'],
        'model__max_depth': [None, 5, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': [None, 'sqrt', 'log2'],
    }

    print("Starting grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=4)
    grid_search.fit(X_clean, y_train)
    best_params = grid_search.best_params_

    print("Metrics after hyperparameter tuning")
    pipeline.set_params(model__criterion = best_params["model__criterion"],
                        model__max_depth = best_params["model__max_depth"],
                        model__min_samples_split = best_params["model__min_samples_split"],
                        model__min_samples_leaf = best_params["model__min_samples_leaf"],
                        model__max_features = best_params["model__max_features"])
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
    

