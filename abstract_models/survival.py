import numpy as np
import pandas as pd
import math

from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def chunk_cindex_calculations(df:pd.DataFrame, chunk: int = 100, time_diff_threshold: int = None):
    total_concordant = 0
    total_discordant = 0
    total_tied_risk = 0
    total_tied_time = 0

    for i in range(0, len(df), chunk):
        pair_df = get_unique_pairs(df.iloc[i:], 0, chunk)
        if time_diff_threshold is not None:
            pair_df = pair_df.loc[pair_df["time_diff"] > time_diff_threshold]
        concordant, discordant, tied_risk, tied_time = count_concordant_discordant_pairs(pair_df)
        total_concordant += concordant
        total_discordant += discordant
        total_tied_risk += tied_risk
        total_tied_time += tied_time
    return total_concordant, total_discordant, total_tied_risk, total_tied_time


def get_unique_pairs(df: pd.DataFrame, start_i=None,chunk=None) -> pd.DataFrame:
    assert df["patid"].duplicated().sum() == 0

    if start_i is not None and chunk is not None:
        first_df = df.iloc[start_i : start_i + chunk]
        pair_df = pd.merge(first_df, df, how="cross")
        unique_pair_df = pd.concat([
            pair_df.loc[~pair_df["patid_y"].isin(first_df["patid"].unique())],
            pair_df.loc[pair_df["patid_y"].isin(first_df["patid"].unique())].query("patid_x > patid_y")
        ])
        assert len(unique_pair_df) == math.comb(chunk, 2) + (len(df) - chunk) * chunk

    else:
        pair_df = pd.merge(df, df, how="cross")
        unique_pair_df = pair_df.loc[pair_df["patid_x"].gt(pair_df["patid_y"])]

        assert len(unique_pair_df) == math.comb(len(df), 2)
    unique_pair_df["time_diff"] = (unique_pair_df["days_to_event_x"] - unique_pair_df["days_to_event_y"]).abs()
    return unique_pair_df


def count_concordant_discordant_pairs(combinations_df: pd.DataFrame):
    # examples with both censored can't be compared
    combinations_df = combinations_df.drop(combinations_df.index[combinations_df["death_event_x"].eq(0) & combinations_df["death_event_y"].eq(0)])

    event_combinations_df = combinations_df.loc[combinations_df["death_event_x"].eq(1) & combinations_df["death_event_y"].eq(1)]

    concordant_x = event_combinations_df["days_to_event_x"].gt(event_combinations_df["days_to_event_y"]) & event_combinations_df["estimate_x"].lt(event_combinations_df["estimate_y"])
    discordant_x = event_combinations_df["days_to_event_x"].gt(event_combinations_df["days_to_event_y"]) & event_combinations_df["estimate_x"].gt(event_combinations_df["estimate_y"])

    concordant_y = event_combinations_df["days_to_event_y"].gt(event_combinations_df["days_to_event_x"]) & event_combinations_df["estimate_y"].lt(event_combinations_df["estimate_x"])
    discordant_y = event_combinations_df["days_to_event_y"].gt(event_combinations_df["days_to_event_x"]) & event_combinations_df["estimate_y"].gt(event_combinations_df["estimate_x"])


    tied_risk = event_combinations_df["estimate_x"].eq(event_combinations_df["estimate_y"]).sum()
    tied_time = event_combinations_df["days_to_event_x"].eq(event_combinations_df["days_to_event_y"]).sum()

    concordant = concordant_x.sum() + concordant_y.sum()
    discordant = discordant_x.sum() + discordant_y.sum()

    event_vs_censored_combinations_df = combinations_df.loc[
        (combinations_df["death_event_y"].eq(1) & combinations_df["death_event_x"].eq(0)) |
        (combinations_df["death_event_y"].eq(0) & combinations_df["death_event_x"].eq(1))
    ]
    drop_too_short_censoring = (
            (combinations_df["death_event_y"].eq(1) & combinations_df["death_event_x"].eq(0) & combinations_df["days_to_event_y"].gt(combinations_df["days_to_event_x"])) |
            (combinations_df["death_event_y"].eq(0) & combinations_df["death_event_x"].eq(1) & combinations_df[
                "days_to_event_x"].gt(combinations_df["days_to_event_y"]))
    )
    event_vs_censored_combinations_df = event_vs_censored_combinations_df.drop(combinations_df.index[drop_too_short_censoring])

    concordant_event_vs_censored = (
            event_vs_censored_combinations_df["death_event_x"].eq(1) &
            event_vs_censored_combinations_df["estimate_x"].gt(event_vs_censored_combinations_df["estimate_y"])
    ).sum()
    concordant_event_vs_censored += (
            event_vs_censored_combinations_df["death_event_y"].eq(1) &
            event_vs_censored_combinations_df["estimate_y"].gt(event_vs_censored_combinations_df["estimate_x"])
    ).sum()
    concordant += concordant_event_vs_censored

    discordant_event_vs_censored = (
            event_vs_censored_combinations_df["death_event_x"].eq(1) &
            event_vs_censored_combinations_df["estimate_y"].gt(event_vs_censored_combinations_df["estimate_x"])).sum()
    discordant_event_vs_censored += (
            event_vs_censored_combinations_df["death_event_y"].eq(1) &
            event_vs_censored_combinations_df["estimate_x"].gt(event_vs_censored_combinations_df["estimate_y"])).sum()
    discordant += discordant_event_vs_censored

    tied_risk_event_vs_censored = event_vs_censored_combinations_df["estimate_x"].eq(event_vs_censored_combinations_df["estimate_y"]).sum()
    tied_risk += tied_risk_event_vs_censored

    assert (concordant + discordant + tied_time + tied_risk) == combinations_df.shape[0] - drop_too_short_censoring.sum()

    return concordant, discordant, tied_risk, tied_time


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
