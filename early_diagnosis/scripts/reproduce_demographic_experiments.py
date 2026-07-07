import json
import os

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer
from abstract_models.metric_utils import (
    binary_metrics_by_age_cohort,
    compute_binary_classification_metrics_adjusted,
)
from abstract_models.param_grid import rf_param_grid
from early_diagnosis.data_loader.loader import load_data


DATA_DIR = "../data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
RANDOM_STATE = 42

experiments_to_run = ["per_gender_evaluation"]
attr_selections = json.load(open(os.path.join(DATA_DIR, "expert_attr_selection.json")))


def get_expert_attrs(df, exclude_attrs=None):
    exclude_attrs = set(exclude_attrs or [])
    attrs = [
        attr for attr in attr_selections["expert"]
        if attr in df.columns and attr not in exclude_attrs
    ]
    if "Med_LD_permanent" in df.columns and "Med_LD_permanent" not in attrs:
        attrs.append("Med_LD_permanent")
    return attrs


def make_base_model_process(n_jobs=10, n_iter=5, cv=None):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", median_imputer),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rf_param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=1,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
    )


def load_balanced_dataset():
    return load_data(os.path.join(DATA_DIR, "processed", "balanced_ED_NT.csv"))


def xy_for_expert_model(df, exclude_attrs=None):
    attrs = get_expert_attrs(df, exclude_attrs=exclude_attrs)
    X = df.loc[:, attrs]
    y = df["Dia_HFD_12M"].astype(int)
    return X, y


def make_inner_cv(y, max_splits=5):
    n_splits = min(max_splits, y.value_counts().min())
    if n_splits < 2:
        raise ValueError("At least two examples from each class are required for inner CV.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def collect_cv_predictions(cv_split, estimator, X, y, metadata=None):
    predictions = []

    metadata = metadata if metadata is not None else pd.DataFrame(index=X.index)

    for fold, (train_idx, test_idx) in enumerate(cv_split(X, y)):
        model = clone(estimator)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fold_predictions = metadata.iloc[test_idx].copy()
        fold_predictions["fold"] = fold
        fold_predictions["y_true"] = y_test.to_numpy()
        fold_predictions["y_pred"] = y_pred
        fold_predictions["y_proba"] = y_proba
        predictions.append(fold_predictions)

    return pd.concat(predictions, axis=0)


def metrics_by_group(prediction_df, group_col):
    results = []
    for group, group_df in prediction_df.groupby(group_col):
        metrics = compute_binary_classification_metrics_adjusted(
            group_df["y_true"],
            group_df["y_pred"],
            group_df["y_proba"],
        )
        metrics[group_col] = group
        metrics["n"] = len(group_df)

        results.append(metrics)
    return pd.DataFrame(results)


if "per_age_cohort_evaluation" in experiments_to_run:
    df = load_balanced_dataset().dropna(subset=["Phy_Age", "Dia_HFD_12M"])
    X, y = xy_for_expert_model(df)

    base_model_process = make_base_model_process()
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    age_prediction_df = collect_cv_predictions(
        k_fold.split,
        base_model_process,
        X,
        y,
        metadata=df.loc[:, ["Phy_Age"]],
    )

    age_prediction_df.to_csv(
        os.path.join(RESULTS_DIR, "per_age_cohort_evaluation_predictions.csv"),
        index=False,
    )

    age_cohort_results_df = binary_metrics_by_age_cohort(
        age_prediction_df["y_true"],
        age_prediction_df["y_proba"],
        age_prediction_df["Phy_Age"],
    )
    age_cohort_results_df.to_csv(
        os.path.join(RESULTS_DIR, "per_age_cohort_evaluation.csv"),
        index=False,
    )

    age_cohort_display_df = age_cohort_results_df.loc[
        :,
        [
            "cohort",
            "n",
            "n_pos",
            "n_neg",
            "prevalence",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
        ],
    ].round(3)
    age_cohort_display_df.to_csv(
        os.path.join(RESULTS_DIR, "per_age_cohort_evaluation_display.csv"),
        index=False,
    )


if "per_gender_cv" in experiments_to_run:
    df = load_balanced_dataset().dropna(subset=["Phy_Sex", "Dia_HFD_12M"])

    gender_results = []
    gender_predictions = []
    for train_gender, test_gender in [("Male", "Female"), ("Female", "Male")]:
        train_df = df.loc[df["Phy_Sex"].eq(train_gender)]
        test_df = df.loc[df["Phy_Sex"].eq(test_gender)]
        if train_df.empty or test_df.empty:
            continue

        X_train, y_train = xy_for_expert_model(train_df, exclude_attrs={"Phy_Sex"})
        X_test, y_test = xy_for_expert_model(test_df, exclude_attrs={"Phy_Sex"})

        model = make_base_model_process(cv=make_inner_cv(y_train))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_binary_classification_metrics_adjusted(y_test, y_pred, y_proba)
        metrics["Train_Gender"] = train_gender
        metrics["Test_Gender"] = test_gender
        metrics["n_train"] = len(train_df)
        metrics["n_test"] = len(test_df)
        gender_results.append(metrics)

        prediction_df = test_df.loc[:, ["Phy_Sex"]].copy()
        prediction_df["Train_Gender"] = train_gender
        prediction_df["Test_Gender"] = test_gender
        prediction_df["y_true"] = y_test.to_numpy()
        prediction_df["y_pred"] = y_pred
        prediction_df["y_proba"] = y_proba
        gender_predictions.append(prediction_df)

    per_gender_cv_results = pd.DataFrame(gender_results)
    per_gender_cv_results.to_csv(os.path.join(RESULTS_DIR, "per_gender_cv.csv"), index=False)

    per_gender_cv_predictions = pd.concat(gender_predictions, axis=0)
    per_gender_cv_predictions.to_csv(
        os.path.join(RESULTS_DIR, "per_gender_cv_predictions.csv"),
        index=False,
    )

    per_gender_cv_display_df = per_gender_cv_results.loc[
        :,
        [
            "Train_Gender",
            "Test_Gender",
            "n_train",
            "n_test",
            "Accuracy",
            "AUC",
            "Precision",
            "Sensitivity (Recall)",
            "Specificity",
            "F1 Score",
        ],
    ].round(3)
    per_gender_cv_display_df.to_csv(os.path.join(RESULTS_DIR, "per_gender_cv_display.csv"), index=False)


if "per_gender_evaluation" in experiments_to_run:
    df = load_balanced_dataset().dropna(subset=["Phy_Sex", "Dia_HFD_12M"])
    X, y = xy_for_expert_model(df)

    base_model_process = make_base_model_process()
    n_splits = min(10, y.value_counts().min())
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    gender_prediction_df = collect_cv_predictions(
        k_fold.split,
        base_model_process,
        X,
        y,
        metadata=df.loc[:, ["Phy_Sex"]],
    )
    gender_prediction_df.to_csv(
        os.path.join(RESULTS_DIR, "per_gender_evaluation_predictions.csv"),
        index=False,
    )

    per_gender_evaluation_results = metrics_by_group(gender_prediction_df, "Phy_Sex")
    per_gender_evaluation_results.to_csv(
        os.path.join(RESULTS_DIR, "per_gender_evaluation.csv"),
        index=False,
    )
    per_gender_evaluation_results["Prevalence"] = (
        (per_gender_evaluation_results["tp"] + per_gender_evaluation_results["fp"])/per_gender_evaluation_results["n"]
    )
    per_gender_evaluation_display_df = per_gender_evaluation_results.loc[
        :,
        [
            "Phy_Sex",
            "n",
            "Prevalence"
            "Accuracy",
            "AUC",
            "Precision",
            "Sensitivity (Recall)",
            "Specificity",
            "F1 Score",
        ],
    ].round(3)
    per_gender_evaluation_display_df.to_csv(
        os.path.join(RESULTS_DIR, "per_gender_evaluation_display.csv"),
        index=False,
    )
