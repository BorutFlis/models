import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def compute_binary_classification_metrics(y_true, y_pred, y_proba):
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # aka Recall
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Return as a pandas Series
    return pd.Series({
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1
    })

