from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_metrics(y_true, y_pred, name="Set"):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    auc = roc_auc_score(y_true, y_pred)
    recall = tp / (tp+fn)
    specificity = tn / (tn+fp)
    precision = tp / (tp+fp)
    f1 = (2 * precision * recall) /  (precision + recall)
    return acc, auc, recall, specificity, f1