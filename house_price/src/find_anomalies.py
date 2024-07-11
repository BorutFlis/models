import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, anderson, kstest


def load_shuffle():
    shuffle_dir = "../house_price_data/interim/shuffled"
    gather_dfs = []
    for f in os.listdir(shuffle_dir):
        if f[-3:] == "csv":
            df = pd.read_csv(os.path.join(shuffle_dir, f), index_col=0)
            gather_dfs.append(df.assign(src=f))
    return pd.concat(gather_dfs)


def identify_unique_sources(data, source_column='src'):
    """
    Identify unique sources in the dataset.

    :param data: DataFrame containing the data.
    :param source_column: Name of the column containing source identifiers.
    :return: List of unique sources.
    """
    return data[source_column].unique()


def outliers_by_center_total_iqr(df, col):
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    for multiplier_step in [1.5, 2, 2.5, 3]:
        lower = q25 - iqr * multiplier_step
        upper = q75 + iqr * multiplier_step

        outliers = df[col].gt(upper) | df[col].lt(lower)
        print(f"{multiplier_step} ")
        print(df.assign(outliers=outliers).groupby("src")["outliers"].sum())


def outliers_by_center_individual_iqr(df_full, col, iqr_multiplier=1.5):
    for center in df_full["src"].unique():
        df = df_full.query(f"src =='{center}'")
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        upper = q75 + iqr * iqr_multiplier
        lower = q25 - iqr * iqr_multiplier
        n_outliers = df[col].gt(upper).sum() + df[col].lt(lower).sum()
        print(f"{center}: {n_outliers}")


def outliers_per_attr(df_floats: pd.DataFrame):
    q25 = df_floats.quantile(0.25)
    q75 = df_floats.quantile(0.75)
    iqr = q75 - q25
    for multiplier_step in [1.5, 2, 2.5, 3]:
        lower = q25 - iqr * multiplier_step
        upper = q75 + iqr * multiplier_step

        n_outliers = (df_floats.gt(upper).sum() + df_floats.lt(lower).sum()).sort_values(ascending=False)
        print(f"{multiplier_step} ")
        print(n_outliers)


def calculate_correlations(data, source_column='src'):
    """
    Calculate correlation matrices for each subset in the data, excluding pairs with less than 10% common non-NA values.
    """
    unique_sources = identify_unique_sources(data, source_column)
    # Preparing a dictionary to hold correlation matrices for each source
    gather_correlations = {}
    for source in unique_sources:
        source_data = data[data[source_column] == source].select_dtypes(include=['number'])
        gather_correlations[source] = source_data.corr(min_periods=int(len(source_data) * 0.1))
    return gather_correlations


def compare_correlations(correlations):
    """
    Compare correlation coefficients across subsets, excluding NaN correlations and focusing on unique column pairs.
    """
    numerical_columns = correlations[next(iter(correlations))].columns
    pairs = [(col1, col2) for i, col1 in enumerate(numerical_columns) for col2 in numerical_columns[i + 1:]]
    variance_data = []

    for col1, col2 in pairs:
        corr_values = [correlations[source].at[col1, col2] for source in correlations if
                       not pd.isna(correlations[source].at[col1, col2])]
        if len(corr_values) == len(correlations):  # Ensure all sources have valid correlations
            std_dev = np.std(corr_values)
            mean_correlation = np.mean(corr_values)
            deviations = {source: abs(correlations[source].at[col1, col2] - mean_correlation) for source in
                          correlations}
            standout_source = max(deviations, key=deviations.get)
            variance_data.append(((col1, col2), std_dev, standout_source))

    variance_df = pd.DataFrame(variance_data, columns=['Column Pair', 'Standard Deviation', 'Standout Source'])
    variance_df.sort_values(by='Standard Deviation', ascending=False, inplace=True)

    return variance_df

auc_intervals = [0.9, 0.8, 0.7]


def run_adversarial_validation_by_center(df: pd.DataFrame):
    #'dataset_label'
    # df = df.drop(to_check.apply(lambda x: tuple(x), axis=1).values, axis=1)

    # all the columns that have more than 50 % NA
    filter_relevant = df.groupby("src").apply(lambda x: (~x.isna()).sum() / len(x)).gt(0.5).all(axis=0).loc[lambda x: x].index

    X = df.loc[:, filter_relevant]
    X = X.drop("src", axis=1).select_dtypes(include=["int", "float"])
    X = X.fillna(X.median())
    gather_results = []
    for src in df["src"].unique():
        y = df["src"].where(df["src"].eq(src), "OTH")
        result_series = run_iterative_adversarial_validation(X, y, 0.55)
        gather_results.append(result_series.to_frame(src))
    return pd.concat(gather_results, axis=1)


def run_iterative_adversarial_validation(X, y, exit_auc_score=0.55, output=False):
    """
    Perform adversarial validation, removing the most informative feature iteratively,
    until the model's AUC score is below a defined threshold.

    Args:
    X (DataFrame): Input features.
    y (Series): Target variable indicating train/test membership.
    exit_auc_score (float): The AUC threshold at which to stop removing features.

    Returns:
    List[str]: The names of the features removed to reach a non-informative model.
    """
    removed_features = []
    current_auc_score = 1.0
    gather_step_report = []

    while current_auc_score >= exit_auc_score and X.shape[1] > 1:
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        clf.fit(X_train, y_train)

        # Predict probabilities for the validation set
        probs = clf.predict_proba(X_val)[:, 1]

        # Calculate the ROC AUC score
        current_auc_score = roc_auc_score(y_val, probs)

        if output:
            print(f"Current ROC AUC Score: {current_auc_score}")

        importances = clf.feature_importances_
        most_important_feature = X.columns[np.argmax(importances)]
        gather_step_report.append({
            "most_important": most_important_feature,
            "auc": current_auc_score,
            "n_removed": len(removed_features),
            "n_remaining": len(X.columns)
        })
        if current_auc_score >= exit_auc_score:
            # Get feature importances and remove the most important feature

            X = X.drop(most_important_feature, axis=1)
            removed_features.append(most_important_feature)
            if output:
                print(f"Removed feature: {most_important_feature}")
    result_df = pd.DataFrame(gather_step_report)
    agg_results = {}
    agg_results["most_important"] = result_df["most_important"].iat[0]
    agg_results["start_auc"] = result_df["auc"].iat[0]
    for auc_step in auc_intervals:
        n_removed, n_remaining = result_df.loc[result_df["auc"].lt(auc_step), ["n_removed", "n_remaining"]].iloc[0].to_list()
        agg_results[f"{auc_step}_n_removed"] = n_removed
        agg_results[f"{auc_step}_n_remaining"] = n_remaining
    agg_results[f"{exit_auc_score}_n_removed"] = len(removed_features)
    agg_results[f"{exit_auc_score}_n_remaining"] = len(X.columns)
    return pd.Series(agg_results)


df = load_shuffle()
correlations = calculate_correlations(df)
most_varied_pairs = compare_correlations(correlations)
print(most_varied_pairs.head(10))
