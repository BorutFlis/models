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

