from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


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

