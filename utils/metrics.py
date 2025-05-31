"""
Performance Evaluation Module

This module provides functions for calculating comprehensive performance metrics
for binary classification tasks including accuracy, AUC, recall, specificity,
precision, and F1-score.

Author: Amir Bhattarai
Date: May 31, 2025
Version: 1.0.0
"""

from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_metrics(y_true, y_pred, name="Set"):
    """
    Calculate binary classification performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
    name : str, default="Set"
        Name identifier for the dataset (currently unused)
        
    Returns:
    --------
    tuple
        A tuple containing (accuracy, auc, recall, specificity, f1)
    """
    # Extract confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate accuracy: (correct predictions) / (total predictions)
    acc = (tp + tn) / (tn + fp + fn + tp)
    
    # Calculate AUC score using ROC curve
    auc = roc_auc_score(y_true, y_pred)
    
    # Calculate recall (sensitivity): true positive rate
    recall = tp / (tp + fn)
    
    # Calculate specificity: true negative rate
    specificity = tn / (tn + fp)
    
    # Calculate precision: positive predictive value
    precision = tp / (tp + fp)
    
    # Calculate F1 score: harmonic mean of precision and recall
    f1 = (2 * precision * recall) / (precision + recall)
    
    return acc, auc, recall, specificity, f1