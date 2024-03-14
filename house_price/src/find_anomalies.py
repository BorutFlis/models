import os

import numpy as np
import pandas as pd


def load_shuffle():
    shuffle_dir = "../house_price_data/interim/shuffled"
    gather_dfs = []
    for f in os.listdir(shuffle_dir):
        if f[-3:] == "csv":
            df = pd.read_csv(os.path.join(shuffle_dir, f))
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


df = load_shuffle()
correlations = calculate_correlations(df)
most_varied_pairs = compare_correlations(correlations)
print(most_varied_pairs.head(10))
