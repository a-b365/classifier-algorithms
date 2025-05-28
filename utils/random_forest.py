# Standard library imports
import os

# Third-party imports 
import numpy as np
import pandas as pd

# Relative imports
from scipy.stats import randint
from prettytable import PrettyTable
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# Local imports
from metrics import evaluate_metrics


if __name__ == "__main__":
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scaler", StandardScaler()),
        ("selector_i", VarianceThreshold(threshold=0.01)),
        ("selector_ii", SelectKBest(score_func=f_classif, k=600)),
        ("model", RandomForestClassifier(n_estimators=100))
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

    myTable = PrettyTable(["Accuracy", "AUC ROC", "Sensitivity", "Specificity", "F1-score"])
    myTable.add_divider()
    acc, auc, recall, specificity, f1 = evaluate_metrics(y_train, y_train_pred)
    myTable.add_row([acc, auc, recall, specificity, f1])
    myTable.add_divider()
    acc, auc, recall, specificity, f1 = evaluate_metrics(y_test, y_test_pred)
    myTable.add_row([acc, auc, recall, specificity, f1])
    print(myTable)


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {

            'model__n_estimators': randint(100, 300),
            'model__max_depth': randint(5, 50),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 20),
            'model__max_features': ['sqrt', 'log2']
    }

    print("Starting randomized search for hyperparameter tuning...")
    random_search = RandomizedSearchCV(pipeline, param_distributions=params, n_iter=10, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    random_search.fit(X_clean, y_train)
    best_params = random_search.best_params_

    print("Selecting the best parameters...")
    pipeline.set_params(model__n_estimators = best_params["model__n_estimators"], 
                        model__max_depth = best_params["model__max_depth"], 
                        model__min_samples_split = best_params["model__min_samples_split"], 
                        model__min_samples_leaf = best_params["model__min_samples_leaf"], 
                        model__max_features = best_params["model__max_features"])
    
    pipeline.fit(X_clean, y_train)
    y_train_pred = pipeline.predict(X_clean)
    y_test_pred = pipeline.predict(X_test)

    print("Applying the params...")
    print("Evaluating the metrics...")
    myTable = PrettyTable(["Accuracy", "AUC ROC", "Sensitivity", "Specificity", "F1-score"])
    myTable.add_divider()
    acc, auc, recall, specificity, f1 = evaluate_metrics(y_train, y_train_pred)
    myTable.add_row([acc, auc, recall, specificity, f1])
    myTable.add_divider()
    acc, auc, recall, specificity, f1 = evaluate_metrics(y_test, y_test_pred)
    myTable.add_row([acc, auc, recall, specificity, f1])
    print(myTable)
    

    

