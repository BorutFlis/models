import os
import json
from functools import partial

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, GroupKFold
from xgboost import XGBClassifier

from abstract_models.param_grid import rf_param_grid, xgb_param_grid
from abstract_models.experiment_utils import run_walk_forward_validation, run_cross_validation
from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import compute_binary_classification_metrics_adjusted, mean_std_metrics_output
from early_diagnosis.data_loader.loader import load_data


DATA_DIR = "../data"

experiments_to_run = ["sliding_window_validation", "per_practice_cv"]
attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))


if "per_practice_cv" in experiments_to_run:
    df = load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))
    df = df.dropna(subset="pracid")
    df["pracid"] = df["pracid"].astype(int)

    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']

    cv_method = GroupKFold(n_splits=10)

    split_func = partial(cv_method.split, groups=df["pracid"])

    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"]
    pipeline = Pipeline(
        steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
    )

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=None,
        scoring='accuracy',
        n_jobs=10,
        verbose=1,
        n_iter=5,
    )
    per_practice_cv_results = run_cross_validation(split_func, base_model_process, compute_binary_classification_metrics_adjusted, X, y)
    k_fold = KFold(n_splits=10, shuffle=True)
    normal_cv_results = run_cross_validation(
        k_fold.split, base_model_process, compute_binary_classification_metrics_adjusted, X, y
    )

    per_practice_cmp_df = pd.concat([
        per_practice_cv_results.assign(Validation_Type="By Practice"),
        normal_cv_results.assign(Validation_Type="K-Fold")
    ])
    per_practice_cmp_df.to_csv(os.path.join(DATA_DIR, "results", "per_practice_cv.csv"))

    per_practice_cv_display_df = mean_std_metrics_output(
        per_practice_cmp_df,
        groupby_col="Validation_Type"
    )
    per_practice_cv_display_df.to_csv(os.path.join(DATA_DIR, "results", "per_practice_cv_display.csv"))

if "sliding_window_validation" in experiments_to_run:
    df = load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))

    attrs = list(set(attr_selections["expert"]).intersection(df.columns)) + ['Med_LD_permanent']

    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"]

    year = pd.to_datetime(df["date"]).dt.year
    year_group = pd.cut(year, bins=[year.min(), 2008, 2012, 2016, 2020, year.max()], include_lowest=True)


    pipeline = Pipeline(
        steps=[('preprocessor', median_imputer), ('classifier', RandomForestClassifier())]
    )

    base_model_process = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=None,
        scoring='accuracy',
        n_jobs=2,
        verbose=1,
        n_iter=30,
    )

    time_cv_results = run_walk_forward_validation(
        X, y, year_group, base_model_process, metric_function=compute_binary_classification_metrics_adjusted
    )
    time_cv_df = pd.DataFrame(time_cv_results)
    time_cv_df.to_csv(os.path.join(DATA_DIR, "results", "sliding_window_validation.csv"))

if "survival_analysis_dp" in experiments_to_run:
    pass

