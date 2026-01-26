import numpy as np
import pandas as pd

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score

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

def get_most_important_feature_reg(df, target):
    df = df.dropna(subset=target)
    y = df[target]
    X = preprocessor.fit_transform(df.drop(target, axis=1))
    rf = RandomForestRegressor(max_depth=5)
    rf.fit(X, y)

    return pd.Series(rf.feature_importances_, index=preprocessor.get_feature_names_out()).sort_values()


def get_preliminary_accuracy(df, target):
    df = df.dropna(subset=target)
    y = df[target]
    X = preprocessor.fit_transform(df.drop(target, axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    rf = RandomForestClassifier(max_depth=5)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))


def get_preliminary_accuracy_reg(df, target):
    df = df.dropna(subset=target)
    y = df[target]
    X = preprocessor.fit_transform(df.drop(target, axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    rf = RandomForestClassifier(max_depth=5)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(r2_score(y_test, y_pred))


def assign_unique_random_numbers(group):
    size = min(len(group), 5)  # Ensure the size does not exceed the range of numbers
    unique_numbers = np.arange(size)
    np.random.shuffle(unique_numbers)
    group['fold'] = unique_numbers[:len(group)]  # Assign to the group
    return group


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


