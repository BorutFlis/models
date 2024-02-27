import numpy as np
import pandas as pd


def assign_unique_random_numbers(group):
    size = min(len(group), 5)  # Ensure the size does not exceed the range of numbers
    unique_numbers = np.arange(size)
    np.random.shuffle(unique_numbers)
    group['fold'] = unique_numbers[:len(group)]  # Assign to the group
    return group


def sort_stratified_regression_group_k(df, target, k=5):
    """
    Ensures balance of regression target variable by
    splitting sorted values
    """

    df = (
        df.reset_index()  # we need indices prior to sort
          .sort_values(by=target)
    )
    # temporary group assigned by position in sorted target variable
    df["group_tmp"] = [int(np.floor(i/k)) for i in range(len(df))]

    # group-by temporary group and assign fold number
    df = df.groupby('group_tmp', group_keys=False).apply(assign_unique_random_numbers)
    df.drop(["group_tmp"], axis=1)

    gather_indices = []
    for fold_step in range(k):
        val_indices = df.query(f"fold == {fold_step}")["index"].tolist()
        gather_indices.append([
            list(set(df["index"].tolist()).difference(val_indices)),
            val_indices
        ])
    return gather_indices
