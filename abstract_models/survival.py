import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def cross_validate_coxph(
    df,
    duration_col,
    event_col,
    n_splits=5,
    shuffle=True,
    random_state=42,
    penalizer=0.0
):
    """
    Perform K-fold cross-validation for a lifelines CoxPHFitter.

    Returns
    -------
    results_df : pd.DataFrame
        Original row index, fold assignment, observed time/event,
        and out-of-fold Cox model predictions for every sample.

    fold_scores : pd.DataFrame
        Training and test concordance index for each fold.
    """

    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None
    )

    # Store out-of-fold predictions
    all_predictions = []

    # Store fold-level performance
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # -------------------------
        # Fit Cox model
        # -------------------------
        cph = CoxPHFitter(penalizer=penalizer)

        cph.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col
        )

        # -------------------------
        # Predictions
        # -------------------------

        # Partial hazard:
        # higher value = higher estimated risk
        train_risk = cph.predict_partial_hazard(train_df)
        test_risk = cph.predict_partial_hazard(test_df)

        # Log partial hazard is often convenient for downstream analysis
        test_log_risk = cph.predict_log_partial_hazard(test_df)

        # -------------------------
        # Concordance
        # -------------------------
        #
        # concordance_index expects:
        # higher prediction = longer survival
        #
        # Cox partial hazard means:
        # higher prediction = higher risk / shorter survival
        #
        # Therefore use NEGATIVE risk.
        train_cindex = concordance_index(
            train_df[duration_col],
            -train_risk,
            train_df[event_col]
        )

        test_cindex = concordance_index(
            test_df[duration_col],
            -test_risk,
            test_df[event_col]
        )

        fold_scores.append({
            "fold": fold,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_cindex": train_cindex,
            "test_cindex": test_cindex
        })

        # -------------------------
        # Save individual test predictions
        # -------------------------
        fold_predictions = pd.DataFrame({
            "original_index": test_df.index,
            "fold": fold,
            "observed_time": test_df[duration_col].values,
            "event": test_df[event_col].values,
            "predicted_partial_hazard": np.asarray(test_risk).ravel(),
            "predicted_log_partial_hazard": np.asarray(test_log_risk).ravel()
        })

        all_predictions.append(fold_predictions)

    # Combine predictions from all held-out folds.
    # Each individual should appear exactly once.
    results_df = pd.concat(all_predictions, ignore_index=True)

    fold_scores = pd.DataFrame(fold_scores)

    return results_df, fold_scores
