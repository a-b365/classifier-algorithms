# Third party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q3 - 1.5 * IQR
    upper_bound = Q1 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)


def impute_outliers(data):
    for column in data.columns:
        outliers = detect_outliers_iqr(data[column])
        median_value = data.loc[np.logical_not(outliers.to_list()), column].median()
        data.loc[outliers.to_list(), column] = median_value
    return data


def visualize_feature(data):
    
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    sns.scatterplot(data, ax=axes[0])
    axes[0].set_title("Original feature")
    sns.boxplot(x=data, ax=axes[1])
    axes[1].set_title("Box plot")

    plt.tight_layout(pad=0.5)
    plt.show()
