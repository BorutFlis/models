import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score
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
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = None

    try:
        # Confusion matrix for specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
    except ValueError:
        specificity = None

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


def binary_metrics_by_age_cohort(
    y_test,
    y_proba,
    age,
    *,
    start_age: int = 40,
    end_age: int = 100,
    cohort_size: int = 10,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> pd.DataFrame:
    """
    Compute binary classification metrics stratified by 10-year age cohorts.

    Cohorts are half-open bins like [40,50), [50,60), ..., [100,110).
    Includes ages within [start_age, end_age] (inclusive).

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        True labels (binary).
    y_proba : array-like of shape (n_samples,)
        Predicted probability for the positive class.
    age : array-like of shape (n_samples,)
        Age for each example.
    start_age, end_age : int
        Inclusive age range to evaluate (default 40..100 inclusive).
    cohort_size : int
        Size of cohort bins (default 10).
    threshold : float
        Decision threshold for converting probabilities to class predictions.
    positive_label : int
        The label considered "positive" (default 1).

    Returns
    -------
    pd.DataFrame
        One row per cohort with metrics and counts.
    """
    y_test = np.asarray(y_test)
    y_proba = np.asarray(y_proba, dtype=float)
    age = np.asarray(age, dtype=float)

    if not (len(y_test) == len(y_proba) == len(age)):
        raise ValueError("y_test, y_proba, and age must have the same length.")

    # Filter to requested age range (inclusive)
    mask = (age >= start_age) & (age <= end_age) & np.isfinite(age)
    y_test_f = y_test[mask]
    y_proba_f = y_proba[mask]
    age_f = age[mask].astype(int)

    if y_test_f.size == 0:
        # Return empty but well-formed output
        cols = [
            "cohort", "age_min", "age_max", "n", "n_pos", "n_neg",
            "prevalence", "roc_auc", "pr_auc", "log_loss", "brier",
            "accuracy", "balanced_accuracy", "precision", "recall", "f1",
            "specificity", "tn", "fp", "fn", "tp",
        ]
        return pd.DataFrame(columns=cols)

    # Define bins: [40,50),...,[100,110) so that age=100 is included
    edges = np.arange(start_age, end_age + cohort_size + 1, cohort_size)  # 40..110 step 10
    cohort_labels = [f"{lo}-{lo+cohort_size-1}" for lo in edges[:-1]]
    cohorts = pd.cut(
        age_f,
        bins=edges,
        right=False,              # left-closed, right-open
        labels=cohort_labels,
        include_lowest=True,
    )

    y_pred_f = (y_proba_f >= threshold).astype(int)
    # Map positives if labels are not {0,1}
    # We assume y_test is already binary; if not, user should encode beforehand.
    if positive_label != 1:
        # Convert to {0,1} for sklearn metrics that assume positive_label=1 by default
        y_test_bin = (y_test_f == positive_label).astype(int)
    else:
        y_test_bin = y_test_f.astype(int)

    rows = []
    for label in cohort_labels:
        idx = (cohorts == label)
        if idx.sum() == 0:
            # Keep cohorts with no samples (optional); here we include with NaNs.
            lo = int(label.split("-")[0])
            hi = int(label.split("-")[1])
            rows.append({
                "cohort": label, "age_min": lo, "age_max": hi,
                "n": 0, "n_pos": 0, "n_neg": 0, "prevalence": np.nan,
                "roc_auc": np.nan, "pr_auc": np.nan, "log_loss": np.nan, "brier": np.nan,
                "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan, "specificity": np.nan,
                "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
            })
            continue

        yt = y_test_bin[idx]
        yp = y_proba_f[idx]
        yhat = y_pred_f[idx]

        n = int(yt.size)
        n_pos = int((yt == 1).sum())
        n_neg = int((yt == 0).sum())
        prev = n_pos / n if n else np.nan

        # Confusion matrix with fixed label order to guarantee 2x2
        tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()

        # Some metrics require both classes present; handle gracefully
        roc_auc = roc_auc_score(yt, yp) if (n_pos > 0 and n_neg > 0) else np.nan
        pr_auc = average_precision_score(yt, yp) if (n_pos > 0 and n_neg > 0) else np.nan

        # log_loss requires both classes unless labels are provided
        ll = log_loss(yt, yp, labels=[0, 1])
        brier = brier_score_loss(yt, yp)

        acc = accuracy_score(yt, yhat)
        bacc = balanced_accuracy_score(yt, yhat)
        prec = precision_score(yt, yhat, zero_division=0)
        rec = recall_score(yt, yhat, zero_division=0)
        f1 = f1_score(yt, yhat, zero_division=0)

        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        lo = int(label.split("-")[0])
        hi = int(label.split("-")[1])

        rows.append({
            "cohort": label,
            "age_min": lo,
            "age_max": hi,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": prev,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": ll,
            "brier": brier,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "specificity": spec,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    return pd.DataFrame(rows)


def mean_std_metrics_output(
        results_df: pd.DataFrame,
        columns=('Accuracy', 'AUC', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1 Score'),
        groupby_col="Model"
):
    output_df = (
        results_df.groupby(groupby_col)[list(columns)].mean().round(2).astype(str) + " ± " + \
        results_df.groupby(groupby_col)[list(columns)].std().round(2).astype(str)
    )
    return output_df