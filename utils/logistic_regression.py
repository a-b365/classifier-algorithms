# Standard library imports
import os

# Third-party imports 
import numpy as np
import pandas as pd

# Relative imports
from prettytable import PrettyTable
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, f1_score

# Local imports
from metrics import evaluate_metrics


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

if __name__ == "__main__":
    
    base_model = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42, class_weight="balanced")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("scaler", StandardScaler()),
        ("var_thresh", VarianceThreshold(threshold=0.01)),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("select_model", SelectFromModel(estimator=base_model, threshold="median")),
        ("model", base_model)
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

    f1_scorer = make_scorer(f1_score, average='macro')

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fit & evaluate
    scores = cross_val_score(pipeline, X_clean, y_train, cv=cv, scoring=f1_scorer)

    print("Mean F1 (Cross-Val):", np.mean(scores))
    print("Std F1 (Cross-Val):", np.std(scores))

    # pipeline.fit(X_clean, y_train)
    # y_train_pred = pipeline.predict(X_clean)
    # y_test_pred = pipeline.predict(X_test)

    # myTable = PrettyTable(["Accuracy", "AUC ROC", "Sensitivity", "Specificity", "F1-score"])
    # myTable.add_divider()
    # acc, auc, recall, specificity, f1 = evaluate_metrics(y_train, y_train_pred)
    # myTable.add_row([acc, auc, recall, specificity, f1])
    # myTable.add_divider()
    # acc, auc, recall, specificity, f1 = evaluate_metrics(y_test, y_test_pred)
    # myTable.add_row([acc, auc, recall, specificity, f1])
    # print(myTable)

    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # params = {
    #         'model__C': [0.001, 0.1, 1, 10, 100],
    #         'model__penalty': ['l1', 'l2'],
    #         'model__solver': ['lbfgs', 'saga']
    # }

    # print("Starting grid search for hyperparameter tuning...")
    # grid_search = GridSearchCV(pipeline, param_grid=params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    # grid_search.fit(X_clean, y_train)
    # best_params = grid_search.best_params_

    # print("Selecting the best parameters...")
    # pipeline.set_params(model__C = best_params["model__C"], 
    #                     model__penalty = best_params["model__penalty"], 
    #                     model__solver = best_params["model__solver"])
    # pipeline.fit(X_clean, y_train)
    # y_train_pred = pipeline.predict(X_clean)
    # y_test_pred = pipeline.predict(X_test)


    # print("Applying the params...")
    # print("Evaluating the metrics...")
    # myTable = PrettyTable(["Accuracy", "AUC ROC", "Sensitivity", "Specificity", "F1-score"])
    # myTable.add_divider()
    # acc, auc, recall, specificity, f1 = evaluate_metrics(y_train, y_train_pred)
    # myTable.add_row([acc, auc, recall, specificity, f1])
    # myTable.add_divider()
    # acc, auc, recall, specificity, f1 = evaluate_metrics(y_test, y_test_pred)
    # myTable.add_row([acc, auc, recall, specificity, f1])
    # print(myTable)