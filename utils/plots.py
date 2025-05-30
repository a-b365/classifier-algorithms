"""
Data visualization utilities for exploratory data analysis.

This module provides functions for visualizing various aspects of datasets
including class distributions, missing values, data types, outliers, and
feature distributions. All functions are designed to work with pandas DataFrames.

Functions:
    show_labels: Visualize class distribution
    missing_row: Show missing values per row
    missing_feature: Show missing values per feature
    show_dtypes: Display data types distribution
    show_boxplot: Create box plots for feature ranges
    show_inf: Visualize infinity values distribution
    inf_features: Show features with infinity values
    visualize_feature: Create scatter and box plots for individual features

Author: Amir Bhattarai
Date: 2025-05-30
Version: 1.0
"""

# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def show_labels(df):
    """
    Display the distribution of class labels in a bar chart.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'CLASS' column with target labels.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'CLASS': [0, 1, 0, 1, 1]})
    >>> show_labels(df)
    """
    labels = df['CLASS'].value_counts().reset_index()
    labels.plot(kind="bar", color="skyblue")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()


def missing_row(df):
    """
    Visualize the distribution of missing values per row.
    
    Shows how many rows have 0, 1, 2, etc. missing values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [1, 2, np.nan]})
    >>> missing_row(df)
    """
    missing_per_row = df.isna().sum(axis=1)
    missing_per_row = missing_per_row.value_counts()
    missing_per_row.plot(kind="bar", color="red")
    plt.title("Distribution of Missing Values per Row")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Number of Rows")
    plt.xticks(rotation=0)
    plt.show()


def missing_feature(df):
    """
    Visualize missing values per feature/column.
    
    Only shows features that have at least one missing value.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [1, 2, 3], 'C': [np.nan, 2, np.nan]})
    >>> missing_feature(df)
    """
    missing_per_column = df.isna().sum(axis=0)
    missing_per_column = missing_per_column[missing_per_column > 0]
    
    if len(missing_per_column) > 0:
        missing_per_column.plot(kind="bar")
        plt.title("Missing Values per Feature")
        plt.xlabel("Features")
        plt.ylabel("Number of Missing Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        print("No missing values found in the dataset.")
    
    plt.show()


def show_dtypes(df):
    """
    Display the distribution of data types in the DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [1.1, 2.2, 3.3]})
    >>> show_dtypes(df)
    """
    plt.figure(figsize=(8, 6))
    feature_dtypes = df.dtypes.value_counts()
    feature_dtypes.plot(kind="bar", color="orange")
    plt.title("Distribution of Data Types")
    plt.xlabel("Data Types")
    plt.ylabel("Number of Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_boxplot(df, start, stop):
    """
    Create box plots for a range of features to visualize distributions and outliers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the features.
    start : int
        Starting column index (inclusive).
    stop : int
        Ending column index (exclusive).
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [1, 2, 3, 4], 'C': [10, 20, 30, 40]})
    >>> show_boxplot(df, 0, 3)
    """
    plt.figure(figsize=(20, 8))
    plt.boxplot(
        df.iloc[:, start:stop],
        positions=range(start, stop, 1),
        tick_labels=df.iloc[:, start:stop].columns.tolist(),
        boxprops=dict(color='blue'),
        whiskerprops=dict(color='red'),
        capprops=dict(color='green'),
        medianprops=dict(color='orange'),
        flierprops=dict(markerfacecolor='red', marker='o')
    )
    plt.title(f"Box Plots for Features {start} to {stop-1}")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_inf(df):
    """
    Visualize the distribution of infinity values in the dataset.
    
    Creates two subplots:
    1. Distribution of infinity values per feature
    2. Distribution of infinity values per row
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, np.inf, 3], 'B': [1, 2, np.inf]})
    >>> show_inf(df)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Infinity values per feature
    inf_per_feature = np.isinf(df).sum(axis=0).value_counts()
    inf_per_feature.plot(kind="bar", color="gold", ax=axes[0])
    axes[0].set_title("Distribution of Infinity Values per Feature")
    axes[0].set_xlabel("Number of Infinity Values")
    axes[0].set_ylabel("Number of Features")
    
    # Infinity values per row
    inf_per_row = np.isinf(df).sum(axis=1).value_counts()
    inf_per_row.plot(kind="bar", color="gold", ax=axes[1])
    axes[1].set_title("Distribution of Infinity Values per Row")
    axes[1].set_xlabel("Number of Infinity Values")
    axes[1].set_ylabel("Number of Rows")
    
    plt.tight_layout()
    plt.show()


def inf_features(df):
    """
    Show which specific features contain infinity values and their counts.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to analyze.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, np.inf, 3], 'B': [1, 2, 3], 'C': [np.inf, np.inf, 3]})
    >>> inf_features(df)
    """
    inf_mask = np.isinf(df)
    inf_summary = inf_mask.sum(axis=0)
    features_with_inf = inf_summary[inf_summary > 0]
    
    if len(features_with_inf) > 0:
        plt.figure(figsize=(12, 6))
        plt.bar(
            features_with_inf.index.tolist(),
            features_with_inf.values,
            color="pink"
        )
        plt.title("Features with Infinity Values")
        plt.xlabel("Features")
        plt.ylabel("Number of Infinity Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        print("No infinity values found in the dataset.")
    
    plt.show()


def visualize_feature(feature_series):
    """
    Create scatter and box plots for a single feature.
    
    Parameters
    ----------
    feature_series : pandas.Series
        A single feature/column from the DataFrame.
    
    Returns
    -------
    None
        Displays the plot directly.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    >>> visualize_feature(df['A'])
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    sns.scatterplot(x=range(len(feature_series)), y=feature_series, ax=axes[0])
    axes[0].set_title(f"Scatter Plot - {feature_series.name}")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Value")
    
    # Box plot
    sns.boxplot(x=feature_series, ax=axes[1])
    axes[1].set_title(f"Box Plot - {feature_series.name}")
    axes[1].set_xlabel("Value")
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function for testing visualization functions.
    
    Reads data from environment variables and demonstrates various
    visualization capabilities.
    """
    try:
        # Get data paths from environment variables
        train_data_path = os.environ["ORIGINAL_DATA_PATH"] + "/train_set.csv"
        test_data_path = os.environ["ORIGINAL_DATA_PATH"] + "/test_set.csv"
        
        # Load data
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        
        print("Data loaded successfully!")
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Prepare features
        X_train = train_data.drop(columns=["ID", "CLASS"])
        y_train = train_data["CLASS"]
        X_test = test_data.drop(columns=["ID", "CLASS"])
        y_test = test_data["CLASS"]
        
        # Example visualizations (uncomment to use)
        # show_labels(train_data)
        # missing_row(train_data)
        # missing_feature(train_data)
        # show_dtypes(train_data)
        # show_boxplot(X_train, 0, 10)
        # show_inf(X_train)
        # inf_features(X_train)
        
        # Visualize a specific feature
        if "Feature_1" in X_train.columns:
            visualize_feature(X_train["Feature_1"])
        else:
            print("Feature_1 not found in the dataset.")
            
    except KeyError as e:
        print(f"Environment variable not found: {e}")
        print("Please set ORIGINAL_DATA_PATH environment variable.")
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()