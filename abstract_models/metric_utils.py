import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, auc, roc_curve
)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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


def plot_multiple_roc_curves(model_results_dict, title="ROC Curves for Multiple Models"):
    """
    Plots ROC curves for multiple models.

    Parameters:
    -----------
    model_results_dict : dict
        Dictionary with keys as model names and values as (y_test, y_proba) tuples.
        y_proba should be the predicted probability for the positive class.

    title : str
        Title for the plot.
    """

    plt.figure(figsize=(10, 8))

    for model_name, model_results in model_results_dict.items():
        y_test = model_results["y_test"]
        y_proba = model_results["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    # Random guess baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()