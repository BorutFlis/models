#!/usr/bin/env python3

import argparse
from functools import partial
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from abstract_models.imputation import median_imputer, median_imputer_missing
from abstract_models.param_grid import rf_param_grid, xgb_param_grid, lgb_param_grid, rf_imbalanced_param_grid, lgb_imbalanced_param_grid, nn_param_grid, svm_param_grid

# Classifiers
classifiers = {
    "RandomForest": (RandomForestClassifier(), rf_param_grid),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid),
    "LightGBM": (LGBMClassifier(random_state=42), lgb_imbalanced_param_grid),
    "SVM": (SVC(probability=True, random_state=42), svm_param_grid)
}

imputers = {
    "median": median_imputer,
    "median_missing": median_imputer_missing
}

model_name = "RandomForest"
model = classifiers[model_name][0]
model_grid = classifiers[model_name][1]



imputer_name = "median"
imputer = imputers[imputer_name]

pipeline = Pipeline(steps=[('preprocessor', imputer), ('classifier', model)])


def run_experiment(n_splits=5, random_state=42):
    data = pd.read_csv("../data/processed/prospective_merged.csv")
    data = data.dropna(subset="diagnosis_hf")
    X = data.drop(columns=['diagnosis_hf', "minnesota_living", "date_diagnosis"])

    expert_columns = [
        'v1_sex',
         'age',
         'v1_height',
         'v1_weight',
         'creatinine_value',
         'ntprobnp_value',
         'egfr_value',
         'myocardial_infarction',
         'cabg',
         'hypertension',
         'atrial_fibrillation2',
         'diabetes_mellitus',
         'shortness_of_breath',
         'peripheral_oedema',
         'chest_pain2'
     ]

    X = X.loc[:, expert_columns]
    binary_cols = ['myocardial_infarction', 'cabg', 'hypertension',
                   'atrial_fibrillation2', 'diabetes_mellitus', 'shortness_of_breath',
                   'peripheral_oedema', 'chest_pain2']
    X["ntprobnp_value"] = X["ntprobnp_value"].apply(pd.to_numeric, errors="coerce")
    X["egfr_value"] = X["egfr_value"].apply(pd.to_numeric, errors="coerce")
    X['creatinine_value'] = X['creatinine_value'].apply(pd.to_numeric, errors="coerce")

    X.loc[:, binary_cols] = X.loc[:, binary_cols].map(lambda x: {1: "Y", 0: "N"}.get(x, x))
    X["v1_sex"] = X["v1_sex"].map(lambda x: {1: "Male", 2: "Female"}.get(x, x))

    y = data['diagnosis_hf'].loc[X.index]

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    metrics = (
        pd.DataFrame(cv_results)
        .filter(regex="^test_")
        .rename(columns=lambda c: c.replace("test_", ""))
    )

    summary = metrics.agg(["mean", "std"]).T

    model.fit(imputer.fit_transform(X.dropna(how="all", axis=1)), y)

    importances = (
        pd.DataFrame({
            "feature": imputer.get_feature_names_out(),
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return metrics, summary, importances



parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--random-state", type=int, default=42)
args = parser.parse_args()

fold_metrics, metric_summary, feature_importances = run_experiment(
    n_splits=args.folds,
    random_state=args.random_state
)

print("\nCross-validation metrics by fold:")
print(fold_metrics.round(4).to_string(index=False))

print("\nCross-validation metric summary:")
print(metric_summary.round(4).to_string())

print("\nRandom forest feature importances:")
print(feature_importances.round(6).to_string(index=False))


