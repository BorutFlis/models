from typing import Callable

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, BaseCrossValidator
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from metric_utils import compute_binary_classification_metrics


def run_imputation_classifier_grid_search(X, y, preprocessor: ColumnTransformer, clf, param_grid, cv=None, n_jobs=2):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )

    grid_search.fit(X, y)
    return grid_search


def run_imputation_classifier_random_search(X, y, preprocessor: ColumnTransformer, clf, param_grid,  cv=None, n_iter=20, n_jobs=2):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
        n_iter=n_iter
    )

    grid_search.fit(X, y)
    return grid_search

def balance_classes_undersample(df, target_col, random_state=None):
    """
    Balance classes in a DataFrame by undersampling all classes
    to match the size of the minority class while preserving
    the original index.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Name of the target/class column.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Balanced dataframe with original indices preserved.
    """

    # Size of the minority class
    min_count = df[target_col].value_counts().min()

    # Sample each class down to min_count
    balanced_df = (
        df.groupby(target_col, group_keys=False)
          .apply(lambda x: x.sample(n=min_count, random_state=random_state))
    )

    return balanced_df


def run_time_dependent_cv_experiment(
    X,
    y,
    time_period_col,
    preprocessor: ColumnTransformer,
    clf,
    param_grid,
    n_splits=5,
    min_train_periods=2,
    test_periods=1,
    search_type="grid",
    n_iter=20,
    n_jobs=2,
):
    """
    Run forward-only (time-dependent) cross validation with increasing temporal training windows.

    The data is sorted by `time_period_col`. Each fold trains on earlier time periods and
    validates on the immediately following period block, so no future information leaks into
    training.
    """

    if time_period_col not in X.columns:
        raise ValueError(f"Column '{time_period_col}' not found in X.")

    if min_train_periods < 1:
        raise ValueError("min_train_periods must be at least 1.")
    if test_periods < 1:
        raise ValueError("test_periods must be at least 1.")

    # Keep y aligned with X while enforcing temporal order.
    order = X[time_period_col].sort_values().index
    X_sorted = X.loc[order].reset_index(drop=True)
    y_sorted = y.loc[order].reset_index(drop=True) if hasattr(y, "loc") else np.asarray(y)[order]

    unique_periods = X_sorted[time_period_col].dropna().sort_values().unique()
    if len(unique_periods) < (min_train_periods + test_periods):
        raise ValueError(
            "Not enough distinct time periods for the requested split configuration."
        )

    max_splits = (len(unique_periods) - min_train_periods) // test_periods
    if max_splits < 1:
        raise ValueError("No valid temporal split can be created with current parameters.")
    if n_splits > max_splits:
        n_splits = max_splits

    cv_splits = []
    for fold_idx in range(n_splits):
        train_end_period_idx = min_train_periods + fold_idx * test_periods
        test_start_period_idx = train_end_period_idx
        test_end_period_idx = test_start_period_idx + test_periods

        train_periods = unique_periods[:train_end_period_idx]
        test_periods_values = unique_periods[test_start_period_idx:test_end_period_idx]
        if len(test_periods_values) == 0:
            break

        train_mask = X_sorted[time_period_col].isin(train_periods)
        test_mask = X_sorted[time_period_col].isin(test_periods_values)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        cv_splits.append((train_idx, test_idx))

    if not cv_splits:
        raise ValueError("Temporal CV produced zero valid folds.")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_splits,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=cv_splits,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1,
            n_iter=n_iter,
        )
    else:
        raise ValueError("search_type must be either 'grid' or 'random'.")

    search.fit(X_sorted, y_sorted)
    return search, cv_splits


def run_time_dependent_cv_experiment_with_series(
    X,
    y,
    time_period_series: pd.Series,
    preprocessor: ColumnTransformer,
    clf,
    param_grid,
    n_splits=5,
    min_train_periods=2,
    test_periods=1,
    search_type="grid",
    n_iter=20,
    n_jobs=2,
):
    """
    Run forward-only temporal CV where time periods are provided as a separate pandas Series.

    Returns the fitted search object, generated cv splits, and per-fold test results with
    the specific time period(s) used as the test set.
    """

    if not isinstance(time_period_series, pd.Series):
        raise ValueError("time_period_series must be a pandas Series.")
    if len(time_period_series) != len(X):
        raise ValueError("time_period_series must have the same length as X.")
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be at least 1.")
    if test_periods < 1:
        raise ValueError("test_periods must be at least 1.")

    # Keep X, y, and time labels aligned in temporal order.
    time_series = time_period_series.reset_index(drop=True)
    order = time_series.sort_values(kind="mergesort").index
    X_sorted = X.iloc[order].reset_index(drop=True)
    time_sorted = time_series.iloc[order].reset_index(drop=True)
    if hasattr(y, "iloc"):
        y_sorted = y.iloc[order].reset_index(drop=True)
    else:
        y_sorted = pd.Series(np.asarray(y)[order]).reset_index(drop=True)

    unique_periods = time_sorted.dropna().sort_values().unique()
    if len(unique_periods) < (min_train_periods + test_periods):
        raise ValueError(
            "Not enough distinct time periods for the requested split configuration."
        )

    max_splits = (len(unique_periods) - min_train_periods) // test_periods
    if max_splits < 1:
        raise ValueError("No valid temporal split can be created with current parameters.")
    if n_splits > max_splits:
        n_splits = max_splits

    cv_splits = []
    fold_periods = []
    for fold_idx in range(n_splits):
        train_end_period_idx = min_train_periods + fold_idx * test_periods
        test_start_period_idx = train_end_period_idx
        test_end_period_idx = test_start_period_idx + test_periods

        train_period_values = unique_periods[:train_end_period_idx]
        test_period_values = unique_periods[test_start_period_idx:test_end_period_idx]
        if len(test_period_values) == 0:
            break

        train_mask = time_sorted.isin(train_period_values)
        test_mask = time_sorted.isin(test_period_values)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        cv_splits.append((train_idx, test_idx))
        fold_periods.append(list(test_period_values))

    if not cv_splits:
        raise ValueError("Temporal CV produced zero valid folds.")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_splits,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=cv_splits,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1,
            n_iter=n_iter,
        )
    else:
        raise ValueError("search_type must be either 'grid' or 'random'.")

    search.fit(X_sorted, y_sorted)

    # Evaluate best params per temporal fold and expose each test period result explicitly.
    period_test_results = []
    for fold_idx, ((train_idx, test_idx), test_period_values) in enumerate(
        zip(cv_splits, fold_periods)
    ):
        fold_estimator = clone(search.best_estimator_)
        fold_estimator.fit(X_sorted.iloc[train_idx], y_sorted.iloc[train_idx])

        y_test = y_sorted.iloc[test_idx]
        y_pred = fold_estimator.predict(X_sorted.iloc[test_idx])
        fold_accuracy = accuracy_score(y_test, y_pred)

        period_test_results.append(
            {
                "fold": fold_idx,
                "test_periods": test_period_values,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(fold_accuracy),
                "y_true": y_test.to_numpy(),
                "y_pred": np.asarray(y_pred),
            }
        )

    return search, cv_splits, period_test_results


def run_walk_forward_validation(
    X,
    y,
    time_period_series: pd.Series,
    base_model: BaseEstimator,
    metric_function: Callable = compute_binary_classification_metrics
):
    """
    Walk-forward validation: train on past time periods, test on the next block.

    Time periods are provided as a separate Series (aligned with X). No hyperparameter
    search — the pipeline is fit once per fold with the given preprocessor and classifier.
    """
    if not isinstance(time_period_series, pd.Series):
        raise ValueError("time_period_series must be a pandas Series.")
    if len(time_period_series) != len(X):
        raise ValueError("time_period_series must have the same length as X.")

    unique_periods = time_period_series.dropna().sort_values().unique()
    if len(unique_periods) < 2:
        raise ValueError(
            "Not enough distinct time periods for the requested split configuration."
        )

    period_test_results = []

    sorted_unique_periods = sorted(unique_periods)

    for i, test_tp in enumerate(sorted_unique_periods[1:]):
        fold_estimator = clone(base_model)
        train_filter = time_period_series.isin(sorted_unique_periods[:i + 1])
        test_filter = time_period_series.eq(test_tp)

        fold_estimator.fit(X.loc[train_filter], y.loc[train_filter])

        y_test = y.loc[test_filter]
        y_pred = fold_estimator.predict(X.loc[test_filter])
        y_proba = fold_estimator.predict_proba(X.loc[test_filter])[:, 1]

        tp_metrics = metric_function(y_test, y_pred, y_proba)

        period_test_results.append(
            {
                **{
                    "test_periods": test_tp,
                    "n_train": train_filter.sum(),
                    "n_test": test_filter.sum()
                },
                **tp_metrics.to_dict()
            }
        )

    return period_test_results


def run_cross_validation(
    cv_split: BaseCrossValidator,
    estimator: BaseEstimator,
    metric_function: Callable,
    X,
    y,
) -> pd.DataFrame:
    scores = []

    for train_idx, test_idx in cv_split(X, y):
        model = clone(estimator)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fold_metrics = metric_function(y_test, y_pred, y_proba)
        scores.append(fold_metrics)

    return pd.DataFrame(scores)

