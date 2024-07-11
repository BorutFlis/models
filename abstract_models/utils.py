import numpy as np
import pandas as pd

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        # Transformer for float columns
        ('num', SimpleImputer(strategy='median'), make_column_selector(dtype_include='float')),
        # Transformer for categorical columns
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), make_column_selector(dtype_include='category')),
    ]
)


def get_most_important_feature(df, target):
    df = df.dropna(subset=target)
    y = df[target]
    X = preprocessor.fit_transform(df.drop(target, axis=1))
    rf = RandomForestClassifier(max_depth=5)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=preprocessor.get_feature_names_out()).sort_values()


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
