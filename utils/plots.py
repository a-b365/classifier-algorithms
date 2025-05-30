import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_labels(df):
    labels = df['CLASS'].value_counts().reset_index()
    labels.plot(kind="bar", color="skyblue")
    plt.show()


def missing_row(df):
    missing_per_row = df.isna().sum(axis=1)
    missing_per_row = missing_per_row.value_counts()
    missing_per_row.plot(kind="bar", color="red")
    plt.show()

def missing_feature(df):
    missing_per_column = df.isna().sum(axis=0)
    missing_per_column = missing_per_column[missing_per_column > 0]
    missing_per_column.plot(kind="bar")
    plt.show()

def show_dtypes(df):
    plt.figure(figsize=(2, 8))
    feature_dtypes = df.dtypes.value_counts()
    feature_dtypes.plot(kind="bar", color="orange")
    plt.title("Data types of the features")
    plt.show()

def show_boxplot(df, start, stop):
    plt.figure(figsize=(20, 30))
    plt.boxplot(df.iloc[:, start:stop], 
            positions=range(start, stop, 1), 
            tick_labels=df.iloc[:, start:stop].columns.to_list(),
            boxprops=dict(color='blue'), 
            whiskerprops=dict(color='red'), 
            capprops=dict(color='green'), 
            medianprops=dict(color='orange'), 
            flierprops=dict(markerfacecolor='red', marker='o')
        )
    plt.show()

def show_inf(df):
    # Count infinities

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    inf_per_feature = np.isinf(df).sum(axis=0).value_counts()
    inf_per_feature.plot(kind="bar", color="gold", ax=ax[0])
    ax[0].set_xlabel("inf value counts")
    ax[0].set_ylabel("features")

    inf_per_row = np.isinf(df).sum(axis=1).value_counts()
    inf_per_row.plot(kind="bar", color="gold", ax=ax[1])
    ax[1].set_xlabel("inf value counts")
    ax[1].set_ylabel("rows")

    plt.tight_layout()
    plt.show()

def inf_features(df):
    inf_mask = np.isinf(df)
    inf_summary = inf_mask.sum(axis=0)
    features = inf_summary[inf_summary > 0].index.to_list()
    plt.bar(features, height=inf_summary[inf_summary > 0].values, color="pink")
    plt.show()

def visualize_feature(df):
    
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    sns.scatterplot(df, ax=axes[0])
    axes[0].set_title("Scatter plot")
    sns.boxplot(x=df, ax=axes[1])
    axes[1].set_title("Box plot")

    plt.tight_layout(pad=0.5)
    plt.show()


if __name__ == "__main__":

    TRAIN_DATA = os.environ["ORIGINAL_DATA_PATH"] + "/train_set.csv"
    TEST_DATA = os.environ["ORIGINAL_DATA_PATH"] + "/test_set.csv"

    train_data = pd.read_csv(TRAIN_DATA)
    test_data = pd.read_csv(TEST_DATA)

    # show_labels(train_data)
    # missing_row(train_data)
    # missing_feature(train_data)
    # show_dtypes(train_data)

    X_train = train_data.drop(columns=["ID", "CLASS"])
    y_train = train_data["CLASS"]

    X_test = test_data.drop(columns=["ID", "CLASS"])
    y_test = test_data["CLASS"]

    #show_boxplot(X_train, 0, 10)
    # show_inf(X_train)

    #inf_features(X_train)
    visualize_feature(X_train["Feature_1"])